//! Kaggle launcher for distributed training.
//!
//! `plan` is offline: it writes one kernel folder per account (kernel-metadata.json and a
//! notebook). Each notebook writes the training package, then starts one DiLoCo worker per
//! T4 (two per account). No API keys are written anywhere in a plan.
//!
//! `push` talks to Kaggle. It needs each account's API token and a coordinator that the
//! notebooks can reach over the public internet. It runs only when every confirmation flag
//! is given, and otherwise prints what it would do.
//!
//! Accounts file: one `alias,username` per line. Keys file (never committed): `alias=KGAT_...`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;

/// The training package written into every notebook. Nothing else is needed on Kaggle.
pub const TRAINING_FILES: &[&str] = &["__init__.py", "pwdata.py", "train.py", "arch.py", "pack.py", "logparse.py", "group_worker.py"];

/// transformers release the architecture code targets. Pinned in every notebook.
pub const TRANSFORMERS_PIN: &str = "5.19.0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Account {
    pub alias: String,
    pub username: String,
}

pub fn parse_accounts(text: &str) -> Result<Vec<Account>> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (alias, username) = line
            .split_once(',')
            .with_context(|| format!("line {}: expected alias,username", i + 1))?;
        let (alias, username) = (alias.trim(), username.trim());
        if alias.is_empty() || username.is_empty() || !alias.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            bail!("line {}: alias must be letters, digits or '-', and username must be set", i + 1);
        }
        if username.chars().any(|c| c.is_whitespace() || c == '/') {
            bail!("line {}: a Kaggle username has no spaces or slashes", i + 1);
        }
        out.push(Account { alias: alias.into(), username: username.into() });
    }
    anyhow::ensure!(!out.is_empty(), "no accounts listed");
    let mut seen = BTreeMap::new();
    for a in &out {
        if seen.insert(a.alias.clone(), ()).is_some() {
            bail!("alias `{}` is listed twice", a.alias);
        }
    }
    Ok(out)
}

/// Kaggle notebooks cannot reach private addresses, so the coordinator needs a public URL.
pub fn check_coordinator_url(url: &str) -> Result<()> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .with_context(|| format!("coordinator URL `{url}` must start with https:// (or http://)"))?;
    let host = rest.split(['/', ':']).next().unwrap_or("");
    let private = matches!(host, "localhost" | "0.0.0.0" | "::1")
        || host.starts_with("127.")
        || host.starts_with("10.")
        || host.starts_with("192.168.")
        || host.starts_with("169.254.");
    anyhow::ensure!(!host.is_empty() && !private, "coordinator host `{host}` is not reachable from Kaggle; use a public tunnel or host");
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// One replica per group. The group's GPUs train it together with DistributedDataParallel.
    Ddp,
    /// One replica per GPU, with no communication inside a group. Use it if DDP fails on a notebook.
    Independent,
}

#[derive(Clone, Debug)]
pub struct PlanConfig {
    pub coordinator_url: String,
    pub accounts: Vec<Account>,
    pub gpus_per_account: usize,
    pub mode: Mode,
    /// Optimizer steps between sends.
    pub inner_steps: usize,
    /// How often a worker checks for a decision (send, merge, stop).
    pub sync_every: usize,
    /// Sends per run before the worker stops (0 = run until the time budget ends).
    pub rounds: usize,
    pub micro_batch: usize,
    pub grad_accum: usize,
    pub lr: f64,
    pub warmup_steps: usize,
    /// Planned optimizer steps for the cosine schedule (0 = inner_steps times rounds).
    pub total_steps: usize,
    pub max_minutes: f64,
    pub training_dir: PathBuf,
}

impl PlanConfig {
    /// DiLoCo replicas the coordinator waits for: groups in DDP mode, GPUs in independent mode.
    pub fn replicas(&self) -> usize {
        match self.mode {
            Mode::Ddp => self.accounts.len(),
            Mode::Independent => self.accounts.len() * self.gpus_per_account,
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct KernelPlan {
    pub alias: String,
    pub username: String,
    pub slug: String,
    pub dir: PathBuf,
    pub ranks: Vec<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct KernelMetadata<'a> {
    id: String,
    title: String,
    code_file: &'a str,
    language: &'a str,
    kernel_type: &'a str,
    is_private: bool,
    enable_gpu: bool,
    enable_tpu: bool,
    enable_internet: bool,
    dataset_sources: Vec<String>,
    competition_sources: Vec<String>,
    kernel_sources: Vec<String>,
    model_sources: Vec<String>,
}

fn source_lines(text: &str) -> Vec<String> {
    let mut lines: Vec<String> = text.split_inclusive('\n').map(String::from).collect();
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn code_cell(text: &str) -> serde_json::Value {
    serde_json::json!({
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    })
}

fn markdown_cell(text: &str) -> serde_json::Value {
    serde_json::json!({ "cell_type": "markdown", "metadata": {}, "source": source_lines(text) })
}

/// The notebook for one account. It installs the pinned libraries, writes the training package,
/// and trains. In DDP mode one torchrun process per GPU shares a replica. In independent mode each
/// GPU is its own replica. Every line of worker output is printed, and the same lines are kept in
/// pw_run.log, so the notebook output is the monitoring log.
pub fn build_notebook(alias: &str, account_index: usize, cfg: &PlanConfig, files: &[(String, String)]) -> serde_json::Value {
    let mode = match cfg.mode {
        Mode::Ddp => "ddp",
        Mode::Independent => "independent",
    };
    let mut cells = vec![markdown_cell(&format!(
        "# PatchWork group `{alias}`\n\nGenerated by `pw train kaggle plan`. This notebook is one DiLoCo group on {} GPUs (mode `{mode}`). \
         Its output is the monitoring log: lines starting with `PW` are JSON events. No credentials are stored here.\n",
        cfg.gpus_per_account
    ))];
    cells.push(code_cell(&format!(
        "# Pinned library versions. The architecture code targets this transformers release.\n!pip install -q \"transformers=={TRANSFORMERS_PIN}\"\n"
    )));
    cells.push(code_cell(
        "import os, pathlib\npathlib.Path('training').mkdir(exist_ok=True)\n\
         # Coordinator token, if the run uses one. Set it as a Kaggle secret named PW_TOKEN.\n\
         try:\n    from kaggle_secrets import UserSecretsClient\n    os.environ['PW_TOKEN'] = UserSecretsClient().get_secret('PW_TOKEN')\n\
         except Exception:\n    pass\n",
    ));
    for (name, text) in files {
        let mut body = format!("%%writefile training/{name}\n");
        body.push_str(text);
        cells.push(code_cell(&body));
    }
    let (replica, replicas) = match cfg.mode {
        Mode::Ddp => (account_index, cfg.replicas()),
        Mode::Independent => (account_index * cfg.gpus_per_account, cfg.replicas()),
    };
    let launch = LAUNCH_TEMPLATE
        .replace("__URL__", &cfg.coordinator_url)
        .replace("__ALIAS__", alias)
        .replace("__MODE__", mode)
        .replace("__GPUS__", &cfg.gpus_per_account.to_string())
        .replace("__REPLICA__", &replica.to_string())
        .replace("__REPLICAS__", &replicas.to_string())
        .replace("__INNER__", &cfg.inner_steps.to_string())
        .replace("__SYNC__", &cfg.sync_every.to_string())
        .replace("__ROUNDS__", &cfg.rounds.to_string())
        .replace("__MB__", &cfg.micro_batch.to_string())
        .replace("__GA__", &cfg.grad_accum.to_string())
        .replace("__LR__", &cfg.lr.to_string())
        .replace("__WARM__", &cfg.warmup_steps.to_string())
        .replace("__TOTAL__", &cfg.total_steps.to_string())
        .replace("__MINUTES__", &cfg.max_minutes.to_string());
    cells.push(code_cell(&launch));
    serde_json::json!({
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    })
}

/// The launch cell. Placeholders (__URL__ etc.) are filled in by build_notebook.
const LAUNCH_TEMPLATE: &str = r#"import os, subprocess, sys, threading, pathlib
os.environ['PYTHONUNBUFFERED'] = '1'
URL = '__URL__'
GROUP = '__ALIAS__'
MODE = '__MODE__'
GPUS = __GPUS__
REPLICA = __REPLICA__
REPLICAS = __REPLICAS__
PROBE_STEPS = 20
COMMON = ['--coordinator', URL, '--inner-steps', '__INNER__', '--sync-every', '__SYNC__',
          '--max-rounds', '__ROUNDS__', '--micro-batch', '__MB__', '--grad-accum', '__GA__',
          '--lr', '__LR__', '--warmup-steps', '__WARM__', '--total-steps', '__TOTAL__',
          '--max-minutes', '__MINUTES__', '--log-every-s', '60', '--global-dtype', 'f32',
          '--eval-every', '1', '--final-wait-minutes', '10']
LOG = pathlib.Path('pw_run.log')
LOG.write_text('')
lock = threading.Lock()
captured = []

def emit(text):
    with lock:
        print(text, end='', flush=True)
        with LOG.open('a') as f:
            f.write(text)

def pump(proc, prefix):
    for line in proc.stdout:
        captured.append(line)
        emit(prefix + line)
    return proc.wait()

def build_jobs(extra):
    jobs = []
    if MODE == 'ddp':
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(GPUS),
               '-m', 'training.group_worker', '--group', GROUP, '--replica', str(REPLICA),
               '--replicas', str(REPLICAS), '--work-dir', 'work'] + COMMON + extra
        jobs.append((cmd, None, ''))
    else:
        for g in range(GPUS):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            cmd = [sys.executable, '-m', 'training.group_worker', '--group', f'{GROUP}-gpu{g}',
                   '--replica', str(REPLICA + g), '--replicas', str(REPLICAS),
                   '--work-dir', f'work-gpu{g}'] + COMMON + extra
            jobs.append((cmd, env, f'[gpu{g}] '))
    return jobs

def run_jobs(jobs):
    procs = [(subprocess.Popen(cmd, env=env or dict(os.environ), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1), prefix)
             for cmd, env, prefix in jobs]
    codes = []
    threads = [threading.Thread(target=lambda p=p, pre=pre: codes.append(pump(p, pre))) for p, pre in procs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return codes

# Probe first: fp16 can overflow on Gemma-style activations, and T4s have no bf16.
precision = 'auto'
if PROBE_STEPS > 0:
    emit('--- probing numerical health (' + str(PROBE_STEPS) + ' steps)\n')
    run_jobs(build_jobs(['--probe-only', '--probe-steps', str(PROBE_STEPS)]))
    finite = [l for l in captured if '"ev":"probe"' in l or '"ev": "probe"' in l]
    healthy = any('"finite":true' in l or '"finite": true' in l for l in finite) and not any('"finite":false' in l or '"finite": false' in l for l in finite)
    if not healthy:
        precision = 'fp32'
        emit('--- probe found non-finite values under fp16; training in fp32 instead (slower on a T4)\n')
    captured.clear()

codes = run_jobs(build_jobs(['--precision', precision]))
print('worker exit codes', codes, flush=True)
from training.logparse import parse, render
print(render(parse([LOG])), flush=True)
"#;

fn read_training_files(dir: &Path) -> Result<Vec<(String, String)>> {
    TRAINING_FILES
        .iter()
        .map(|name| {
            let text = fs::read_to_string(dir.join(name)).with_context(|| format!("reading {}", dir.join(name).display()))?;
            Ok((name.to_string(), text))
        })
        .collect()
}

/// Write one kernel folder per account. Returns the plans; nothing is sent anywhere.
pub fn write_plan(cfg: &PlanConfig, out: &Path) -> Result<Vec<KernelPlan>> {
    check_coordinator_url(&cfg.coordinator_url)?;
    anyhow::ensure!(cfg.gpus_per_account >= 1, "at least one GPU per account");
    anyhow::ensure!(cfg.micro_batch >= 1 && cfg.grad_accum >= 1, "micro-batch and grad-accum must be at least 1");
    anyhow::ensure!(cfg.sync_every >= 1 && cfg.inner_steps >= cfg.sync_every, "inner steps must be at least the sync interval");
    let files = read_training_files(&cfg.training_dir)?;
    fs::create_dir_all(out)?;
    let mut plans = Vec::new();
    for (a, account) in cfg.accounts.iter().enumerate() {
        let ranks: Vec<usize> = match cfg.mode {
            Mode::Ddp => vec![a],
            Mode::Independent => (0..cfg.gpus_per_account).map(|g| a * cfg.gpus_per_account + g).collect(),
        };
        let slug = format!("pw-worker-{}", account.alias);
        let dir = out.join(&account.alias);
        fs::create_dir_all(&dir)?;
        let meta = KernelMetadata {
            id: format!("{}/{}", account.username, slug),
            title: slug.clone(),
            code_file: "pw_worker.ipynb",
            language: "python",
            kernel_type: "notebook",
            is_private: true,
            enable_gpu: true,
            enable_tpu: false,
            enable_internet: true,
            dataset_sources: vec![],
            competition_sources: vec![],
            kernel_sources: vec![],
            model_sources: vec![],
        };
        fs::write(dir.join("kernel-metadata.json"), serde_json::to_string_pretty(&meta)?)?;
        let nb = build_notebook(&account.alias, a, cfg, &files);
        fs::write(dir.join("pw_worker.ipynb"), serde_json::to_string_pretty(&nb)?)?;
        plans.push(KernelPlan { alias: account.alias.clone(), username: account.username.clone(), slug, dir, ranks });
    }
    let mode = match cfg.mode {
        Mode::Ddp => "ddp (one replica per group, both GPUs share it)",
        Mode::Independent => "independent (one replica per GPU)",
    };
    let summary = format!(
        "# PatchWork Kaggle plan\n\n- accounts (groups): {}\n- GPUs per group: {}\n- mode: {}\n- DiLoCo replicas: {}\n- total GPUs: {}\n- coordinator: {}\n- inner steps between sends: {}\n- sync every: {} steps\n- sends per run: {}\n- micro-batch: {}, grad accumulation: {}\n- learning rate: {}, warmup: {} steps, planned total steps: {}\n- time budget per notebook: {} minutes\n\nNothing here contains credentials. `push` needs each account's API token, supplied at run time.\n",
        cfg.accounts.len(),
        cfg.gpus_per_account,
        mode,
        cfg.replicas(),
        cfg.accounts.len() * cfg.gpus_per_account,
        cfg.coordinator_url,
        cfg.inner_steps,
        cfg.sync_every,
        if cfg.rounds == 0 { "until the time budget ends".to_string() } else { cfg.rounds.to_string() },
        cfg.micro_batch,
        cfg.grad_accum,
        cfg.lr,
        cfg.warmup_steps,
        cfg.total_steps,
        cfg.max_minutes
    );
    fs::write(out.join("PLAN.md"), summary)?;
    Ok(plans)
}

/// Read the plans written by `write_plan` (one folder per account).
pub fn load_plans(out: &Path) -> Result<Vec<KernelPlan>> {
    let mut plans = Vec::new();
    let mut dirs: Vec<PathBuf> = fs::read_dir(out)
        .with_context(|| format!("reading plan folder {}", out.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("kernel-metadata.json").exists())
        .collect();
    dirs.sort();
    for dir in dirs {
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join("kernel-metadata.json"))?)?;
        let id = meta["id"].as_str().context("kernel-metadata.json has no id")?;
        let (username, slug) = id.split_once('/').context("kernel id must be user/slug")?;
        let alias = dir.file_name().and_then(|n| n.to_str()).context("plan folder has no name")?.to_string();
        plans.push(KernelPlan { alias, username: username.into(), slug: slug.into(), dir, ranks: Vec::new() });
    }
    anyhow::ensure!(!plans.is_empty(), "no kernel folders found in {}", out.display());
    Ok(plans)
}

/// Parse `alias=KGAT_...` lines. Values are never printed.
pub fn parse_keys(text: &str) -> Result<BTreeMap<String, String>> {
    let mut keys = BTreeMap::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (alias, key) = line.split_once('=').with_context(|| format!("keys line {}: expected alias=KEY", i + 1))?;
        anyhow::ensure!(!key.trim().is_empty(), "keys line {}: empty key", i + 1);
        keys.insert(alias.trim().to_string(), key.trim().to_string());
    }
    Ok(keys)
}

/// The push gate. Every flag must be set, and the number must match the plan.
pub fn push_allowed(plans: usize, confirm_accounts: usize, acknowledged_rules: bool) -> Result<()> {
    anyhow::ensure!(
        acknowledged_rules,
        "push refused: pass --i-have-checked-kaggle-rules after confirming that running these accounts is allowed by Kaggle's terms and your quotas"
    );
    anyhow::ensure!(
        confirm_accounts == plans,
        "push refused: --confirm-accounts {confirm_accounts} does not match the {plans} accounts in the plan"
    );
    Ok(())
}

/// Run `kaggle kernels push` for each plan. `dry_run` prints the commands and sends nothing.
pub fn push(plans: &[KernelPlan], keys: &BTreeMap<String, String>, dry_run: bool) -> Result<()> {
    for plan in plans {
        let key = keys.get(&plan.alias).with_context(|| format!("no key for account `{}`", plan.alias))?;
        println!("kaggle kernels push -p {} (account {}, key from keys file)", plan.dir.display(), plan.alias);
        if dry_run {
            continue;
        }
        let status = std::process::Command::new("kaggle")
            .args(["kernels", "push", "-p"])
            .arg(&plan.dir)
            .env("KAGGLE_API_TOKEN", key)
            .status()
            .context("running the kaggle CLI")?;
        anyhow::ensure!(status.success(), "kaggle push failed for account {}", plan.alias);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(accounts: usize, training_dir: PathBuf, mode: Mode) -> PlanConfig {
        PlanConfig {
            coordinator_url: "https://coord.example.org".into(),
            accounts: (0..accounts).map(|i| Account { alias: format!("acct{i}"), username: format!("user{i}") }).collect(),
            gpus_per_account: 2,
            mode,
            inner_steps: 500,
            sync_every: 10,
            rounds: 0,
            micro_batch: 8,
            grad_accum: 2,
            lr: 6e-4,
            warmup_steps: 200,
            total_steps: 20000,
            max_minutes: 540.0,
            training_dir,
        }
    }

    fn training_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../training")
    }

    fn scratch(name: &str) -> PathBuf {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp").join(format!("kaggle-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }

    fn notebook_text(path: &Path) -> String {
        let nb: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        nb["cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["source"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn accounts_file_is_validated() {
        let ok = parse_accounts("# comment\nacct1,alice\nacct2,bob\n").unwrap();
        assert_eq!(ok.len(), 2);
        assert!(parse_accounts("acct1 alice").is_err());
        assert!(parse_accounts("acct1,al ice").is_err());
        assert!(parse_accounts("a,x\na,y").is_err());
        assert!(parse_accounts("# nothing\n").is_err());
    }

    #[test]
    fn coordinator_must_be_public() {
        assert!(check_coordinator_url("https://coord.example.org").is_ok());
        assert!(check_coordinator_url("http://127.0.0.1:8765").is_err());
        assert!(check_coordinator_url("http://192.168.1.4:8765").is_err());
        assert!(check_coordinator_url("ftp://x.example").is_err());
    }

    #[test]
    fn ddp_plan_has_one_replica_per_group_and_the_expected_count() {
        let out = scratch("ddp");
        let config = cfg(3, training_dir(), Mode::Ddp);
        assert_eq!(config.replicas(), 3);
        let plans = write_plan(&config, &out).unwrap();
        assert_eq!(plans.len(), 3);
        assert_eq!(plans[2].ranks, vec![2]);
        let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(out.join("acct2/kernel-metadata.json")).unwrap()).unwrap();
        assert_eq!(meta["id"], "user2/pw-worker-acct2");
        assert_eq!(meta["enableGpu"], true);
        assert_eq!(meta["enableInternet"], true);
        assert_eq!(meta["isPrivate"], true);
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn independent_plan_has_one_replica_per_gpu() {
        let config = cfg(3, training_dir(), Mode::Independent);
        assert_eq!(config.replicas(), 6);
        let out = scratch("indep");
        let plans = write_plan(&config, &out).unwrap();
        assert_eq!(plans[2].ranks, vec![4, 5]);
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn generated_files_never_contain_key_material() {
        let out = scratch("nokeys");
        write_plan(&cfg(2, training_dir(), Mode::Ddp), &out).unwrap();
        for entry in walk(&out) {
            let text = fs::read_to_string(&entry).unwrap();
            assert!(!text.contains("KGAT_"), "{}", entry.display());
            assert!(!text.contains("KAGGLE_API_TOKEN="), "{}", entry.display());
        }
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn ddp_notebook_writes_the_package_and_launches_torchrun_on_both_gpus() {
        let out = scratch("nb-ddp");
        write_plan(&cfg(1, training_dir(), Mode::Ddp), &out).unwrap();
        let text = notebook_text(&out.join("acct0/pw_worker.ipynb"));
        for name in TRAINING_FILES {
            assert!(text.contains(&format!("%%writefile training/{name}")), "missing {name}");
        }
        assert!(text.contains("transformers==5.19.0"));
        assert!(text.contains("'torch.distributed.run'") || text.contains("torch.distributed.run"));
        assert!(text.contains("'--nproc_per_node', str(GPUS)"));
        assert!(text.contains("GPUS = 2"));
        assert!(text.contains("'--replicas', str(REPLICAS)"));
        assert!(text.contains("pw_run.log"));
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn independent_notebook_pins_each_gpu_to_its_own_process() {
        let out = scratch("nb-indep");
        write_plan(&cfg(2, training_dir(), Mode::Independent), &out).unwrap();
        let text = notebook_text(&out.join("acct1/pw_worker.ipynb"));
        assert!(text.contains("CUDA_VISIBLE_DEVICES"));
        assert!(!text.contains("GPUS = 2\nREPLICA = 2\n") || text.contains("REPLICA = 2"));
        assert!(text.contains("REPLICA = 2"), "account 1 starts at replica 2 in independent mode");
        let _ = fs::remove_dir_all(&out);
    }

    #[test]
    fn push_is_refused_without_both_confirmations() {
        assert!(push_allowed(14, 14, false).is_err());
        assert!(push_allowed(14, 13, true).is_err());
        assert!(push_allowed(14, 14, true).is_ok());
    }

    #[test]
    fn keys_file_parses_without_echoing_values() {
        let keys = parse_keys("acct1=KGAT_abc\n# c\nacct2 = KGAT_def\n").unwrap();
        assert_eq!(keys.get("acct1").map(String::as_str), Some("KGAT_abc"));
        assert_eq!(keys.get("acct2").map(String::as_str), Some("KGAT_def"));
        assert!(parse_keys("acct1=").is_err());
    }

    fn walk(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for e in fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
        out
    }
}
