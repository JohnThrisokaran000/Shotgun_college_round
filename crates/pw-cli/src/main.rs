//! `pw`: patch live websites without rebuilding them.
//!
//! Exit codes: 0 success; 1 error or a blocked release or failed assertion;
//! 2 `check` found patches that are broken or need a human.

mod browser;
mod project;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use pw_core::audit::GENESIS;
use pw_core::conflict;
use pw_core::relocate::Status;
use pw_core::release;
use serde_json::json;

use project::{parse_pair, Project};

#[derive(Parser)]
#[command(name = "pw", version, about = "PatchWork: patch live websites without rebuilding them")]
struct Cli {
    /// Project directory (the one holding pw.toml)
    #[arg(long, global = true, default_value = ".")]
    project: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a project in the --project directory
    Init {
        #[arg(long)]
        name: String,
        /// Origin of the live site, for example http://127.0.0.1:8080
        #[arg(long)]
        upstream: String,
    },
    /// Print the JSON Schema of the patch DSL (the AUTHOR model's output format)
    DslSchema,
    /// Re-anchor every patch against its page snapshot (no browser needed)
    Check {
        /// PAGE=FILE, for example /cart=snapshots/cart.html (repeatable)
        #[arg(long = "snapshot", value_parser = parse_pair)]
        snapshots: Vec<(String, PathBuf)>,
        /// Rewrite relocated patches so their anchors point at the new element
        #[arg(long)]
        write: bool,
        #[arg(long)]
        json: bool,
    },
    /// Check this project's patches against another branch's patches for conflicts
    Merge {
        /// Directory with the other branch's patch files
        #[arg(long)]
        with: PathBuf,
        #[arg(long = "snapshot", value_parser = parse_pair)]
        snapshots: Vec<(String, PathBuf)>,
    },
    /// Publish verified patches to a channel
    Release {
        #[arg(long, default_value = "production")]
        channel: String,
        #[arg(long = "snapshot", value_parser = parse_pair)]
        snapshots: Vec<(String, PathBuf)>,
        #[arg(long)]
        dry_run: bool,
        /// Allow patches that were relocated by the heuristic. Run `pw verify` on a preview
        /// channel first: a relocation is only as good as its assertions.
        #[arg(long)]
        accept_relocated: bool,
    },
    /// Point a channel back to the previous release
    Rollback {
        #[arg(long, default_value = "production")]
        channel: String,
    },
    /// Show the most recent audit events
    Log {
        #[arg(long, default_value_t = 20)]
        tail: usize,
    },
    /// Check the audit chain for edits, gaps, and truncation
    AuditVerify,
    /// Serve the site through the patch proxy
    Proxy {
        #[arg(long, default_value = "127.0.0.1:8787")]
        listen: SocketAddr,
    },
    /// Playwright management
    Browser {
        #[command(subcommand)]
        cmd: BrowserCmd,
    },
    /// Capture a page (HTML and element rectangles) for check and release
    Snapshot {
        url: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        headed: bool,
    },
    /// Open a page with the visual editor; each Save writes a patch file
    Edit {
        url: String,
        #[arg(long, default_value = "dev")]
        author: String,
    },
    /// Model training: data generation, the coordinator, and the Kaggle plan
    Train {
        #[command(subcommand)]
        cmd: TrainCmd,
    },
    /// Run a channel's assertions in a real browser through the proxy
    Verify {
        /// Proxy base URL, for example http://127.0.0.1:8787
        #[arg(long)]
        base: String,
        #[arg(long, default_value = "production")]
        channel: String,
    },
}

#[derive(Subcommand)]
enum KaggleCmd {
    /// Write one kernel folder per account (offline; no credentials are written)
    Plan {
        /// Accounts file: one `alias,username` per line
        #[arg(long)]
        accounts: PathBuf,
        /// Public URL of the coordinator (Kaggle cannot reach private addresses)
        #[arg(long)]
        coordinator_url: String,
        #[arg(long, default_value = "kaggle/plan")]
        out: PathBuf,
        /// GPUs per group (a Kaggle group has two T4s)
        #[arg(long, default_value_t = 2)]
        gpus: usize,
        /// ddp: one replica per group, both GPUs share it. independent: one replica per GPU
        #[arg(long, default_value = "ddp")]
        mode: String,
        /// Optimizer steps between sends
        #[arg(long, default_value_t = 1000)]
        inner_steps: usize,
        /// How often a worker checks for a decision
        #[arg(long, default_value_t = 10)]
        sync_every: usize,
        /// Sends per run (0 = until the time budget ends)
        #[arg(long, default_value_t = 0)]
        rounds: usize,
        #[arg(long, default_value_t = 8)]
        micro_batch: usize,
        #[arg(long, default_value_t = 2)]
        grad_accum: usize,
        #[arg(long, default_value_t = 6e-4)]
        lr: f64,
        #[arg(long, default_value_t = 200)]
        warmup_steps: usize,
        /// Planned optimizer steps for the cosine schedule (set this for a real run)
        #[arg(long, default_value_t = 0)]
        total_steps: usize,
        /// Training minutes per notebook. Kaggle start-up, the install and the probe take the rest of the window.
        #[arg(long, default_value_t = 220.0)]
        max_minutes: f64,
        #[arg(long, default_value = "training")]
        training_dir: PathBuf,
    },
    /// Push the plan to Kaggle. Dry run unless every confirmation flag is given
    Push {
        #[arg(long, default_value = "kaggle/plan")]
        plan: PathBuf,
        /// Keys file with `alias=KEY` lines (kept out of version control)
        #[arg(long, default_value = "kaggle/secrets/keys.env")]
        keys: PathBuf,
        /// Actually push. Without this flag the command only prints what it would run
        #[arg(long)]
        execute: bool,
        /// Must equal the number of accounts in the plan
        #[arg(long)]
        confirm_accounts: Option<usize>,
        /// Confirms you have checked Kaggle's terms and your quotas for these accounts
        #[arg(long)]
        i_have_checked_kaggle_rules: bool,
    },
}

#[derive(Subcommand)]
enum TrainCmd {
    /// Distributed training on Kaggle: plan and push one notebook per account
    Kaggle {
        #[command(subcommand)]
        cmd: KaggleCmd,
    },
    /// Held-out benchmark for re-anchoring: heuristic vs selector-only, on unseen synthetic sites
    Bench {
        #[arg(long, default_value_t = 200)]
        sites: usize,
        #[arg(long, default_value_t = 6)]
        variants: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long)]
        json: bool,
    },
    /// Run the distributed-training coordinator (workers send deltas, it averages and steps)
    Serve {
        /// init.safetensors written by `python -m training.prepare`
        #[arg(long)]
        init: PathBuf,
        /// Where the global model and round state are kept between restarts
        #[arg(long)]
        state: PathBuf,
        /// Directory served to workers at /v1/files/<name> (config, tokenizer, training rows)
        #[arg(long)]
        serve: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1:8765")]
        listen: SocketAddr,
        #[arg(long)]
        workers: usize,
        /// Deltas needed to close a round (default: all workers)
        #[arg(long)]
        quorum: Option<usize>,
        /// A round with at least one delta closes after this long
        #[arg(long, default_value_t = 30.0)]
        timeout_minutes: f64,
        #[arg(long, default_value_t = 0.7)]
        outer_lr: f32,
        #[arg(long, default_value_t = 0.9)]
        outer_momentum: f32,
    },
    /// Generate RELOCATE and AUTHOR training data and a held-out split
    GenData {
        #[arg(long, default_value = "data/patchwork")]
        out: PathBuf,
        #[arg(long, default_value_t = 2000)]
        sites: usize,
        #[arg(long, default_value_t = 6)]
        variants: usize,
        #[arg(long, default_value_t = 3)]
        author: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Share of RELOCATE examples labelled null (deleted targets)
        #[arg(long, default_value_t = 0.15)]
        null_fraction: f64,
    },
}

#[derive(Subcommand)]
enum BrowserCmd {
    /// Download the Playwright Chromium build
    Install {
        /// Also install the system libraries Chromium needs (may ask for sudo)
        #[arg(long)]
        with_deps: bool,
    },
    /// Launch Chromium once and print its version
    Status,
}

fn status_word(s: Status) -> &'static str {
    project::status_label(s)
}

fn cmd_check(project: &Project, pairs: &[(String, PathBuf)], write: bool, as_json: bool) -> Result<bool> {
    let snaps = project::load_snapshots(pairs)?;
    let patches = project.load_patches()?;
    let mut rows = Vec::new();
    let mut needs_human = false;
    for patch in &patches {
        let Some(snap) = snaps.get(&patch.page) else {
            rows.push(json!({"id": patch.id, "page": patch.page, "status": "skipped", "reason": "no snapshot for this page"}));
            continue;
        };
        let out = project::relocate_one(patch, snap)?;
        if matches!(out.status, Status::Ambiguous | Status::Broken) {
            needs_human = true;
        }
        if write && out.status == Status::Relocated {
            if let Some(el) = out.element {
                project.overwrite_patch(&project::rebase(patch, snap, el)?)?;
            }
        }
        rows.push(json!({
            "id": patch.id,
            "page": patch.page,
            "status": status_word(out.status),
            "element": out.element,
            "confidence": (out.confidence * 1000.0).round() / 1000.0,
            "reason": out.reason,
            "candidates": out.top,
        }));
    }
    if as_json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else if rows.is_empty() {
        println!("no patches in {}", project.patches_dir().display());
    } else {
        for r in &rows {
            println!(
                "{:<28} {:<10} {:<10} conf {:<6} {}",
                r["id"].as_str().unwrap_or(""),
                r["page"].as_str().unwrap_or(""),
                r["status"].as_str().unwrap_or(""),
                r["confidence"].as_f64().map_or("-".to_string(), |c| format!("{c:.3}")),
                r["reason"].as_str().unwrap_or("")
            );
        }
    }
    Ok(needs_human)
}

fn cmd_merge(project: &Project, with: &PathBuf, pairs: &[(String, PathBuf)]) -> Result<bool> {
    let snaps = project::load_snapshots(pairs)?;
    let mut all = project.load_patches()?;
    let theirs = project::load_patch_dir(with)?;
    let theirs_ids: Vec<String> = theirs.iter().map(|p| p.id.clone()).collect();
    all.retain(|p| !theirs_ids.contains(&p.id));
    all.extend(theirs);
    let mut hard = false;
    for (page, snap) in &snaps {
        let targets = project::release_targets(&all, snap, page);
        for c in conflict::detect(&targets, snap) {
            hard |= c.hard;
            println!(
                "{} {:?} on {}: {} vs {} ({})",
                if c.hard { "CONFLICT" } else { "warning " },
                c.kind,
                page,
                c.patches[0],
                c.patches[1],
                c.detail
            );
        }
    }
    if !hard {
        println!("no hard conflicts across {} patches", all.len());
    }
    Ok(hard)
}

async fn cmd_release(project: &Project, channel: &str, pairs: &[(String, PathBuf)], dry_run: bool, accept_relocated: bool) -> Result<()> {
    let snaps = project::load_snapshots(pairs)?;
    let plan = project::plan_release(project, channel, &snaps)?;
    for id in &plan.relocated {
        println!("relocated  {id} (anchor will be rewritten in the release)");
    }
    for (id, why) in &plan.refused {
        println!("refused    {id}: {why}");
    }
    for c in &plan.conflicts {
        println!(
            "{} {:?}: {} vs {} ({})",
            if c.hard { "conflict" } else { "warning " },
            c.kind,
            c.patches[0],
            c.patches[1],
            c.detail
        );
    }
    if !plan.is_clear() {
        bail!("release blocked: fix the refused patches and hard conflicts above, then run again");
    }
    if let Some(current) = release::current(&project.root, channel)? {
        if release::read_release(&project.root, &current)?.patches == plan.manifest.patches {
            println!("{channel} already serves this content as release {current}; nothing to do");
            return Ok(());
        }
    }
    if !plan.relocated.is_empty() && !accept_relocated && !dry_run {
        bail!(
            "release blocked: {} patch(es) were relocated by the heuristic ({}). Verify them on a preview channel with `pw verify`, then pass --accept-relocated",
            plan.relocated.len(),
            plan.relocated.join(", ")
        );
    }
    if dry_run {
        println!("dry run: would publish release {} to {channel}", plan.manifest.id);
        return Ok(());
    }
    release::write_release(&project.root, &plan.manifest)?;
    release::set_channel(&project.root, channel, &plan.manifest.id)?;
    let ids: Vec<&str> = plan.manifest.patches.iter().map(|p| p.id.as_str()).collect();
    project.audit().append(
        "cli",
        "release",
        json!({"channel": channel, "id": plan.manifest.id, "parent": plan.manifest.parent, "patches": ids, "relocated": plan.relocated}),
    )?;
    println!("released {} to {channel} ({} patches)", plan.manifest.id, ids.len());
    Ok(())
}

fn cmd_rollback(project: &Project, channel: &str) -> Result<()> {
    let back = release::rollback(&project.root, channel)?;
    project.audit().append("cli", "rollback", json!({"channel": channel, "to": back}))?;
    println!("{channel} now serves release {back}");
    Ok(())
}

fn cmd_log(project: &Project, tail: usize) -> Result<()> {
    let events = project.audit().read_all()?;
    let start = events.len().saturating_sub(tail);
    for e in &events[start..] {
        println!("{:>5} {:>14} {:<8} {:<14} {}", e.seq, e.ts_ms, e.actor, e.kind, e.data);
    }
    if events.is_empty() {
        println!("(no events)");
    }
    Ok(())
}

fn cmd_audit_verify(project: &Project) -> Result<()> {
    let report = project.audit().verify()?;
    if report.events == 0 {
        println!("audit log is empty (head {GENESIS})");
    } else {
        println!("audit ok: {} events, head {}", report.events, report.head);
    }
    Ok(())
}

async fn cmd_verify(project: &Project, base: &str, channel: &str) -> Result<bool> {
    let id = release::current(&project.root, channel)?
        .with_context(|| format!("channel `{channel}` has no release; run `pw release` first"))?;
    let manifest = release::read_release(&project.root, &id)?;
    let results = browser::verify(base, channel, &manifest.patches).await?;
    let mut failed = false;
    for r in &results {
        let mark = match r.passed {
            Some(true) => "pass",
            Some(false) => {
                failed = true;
                "FAIL"
            }
            None => "skip",
        };
        println!("{mark}  {:<28} {}", r.patch, r.label);
    }
    if results.is_empty() {
        println!("no assertions in release {id}");
    }
    Ok(failed)
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    let cli = Cli::parse();
    let root = cli.project.clone();
    match cli.cmd {
        Cmd::Init { name, upstream } => {
            Project::init(&root, &name, &upstream)?;
            println!("created PatchWork project `{name}` in {}", root.display());
        }
        Cmd::DslSchema => {
            println!("{}", serde_json::to_string_pretty(&pw_core::dsl::op_schema())?);
        }
        Cmd::Check { snapshots, write, json } => {
            let project = Project::open(&root)?;
            if cmd_check(&project, &snapshots, write, json)? {
                return Ok(ExitCode::from(2));
            }
        }
        Cmd::Merge { with, snapshots } => {
            let project = Project::open(&root)?;
            if cmd_merge(&project, &with, &snapshots)? {
                return Ok(ExitCode::from(1));
            }
        }
        Cmd::Release { channel, snapshots, dry_run, accept_relocated } => {
            let project = Project::open(&root)?;
            cmd_release(&project, &channel, &snapshots, dry_run, accept_relocated).await?;
        }
        Cmd::Rollback { channel } => cmd_rollback(&Project::open(&root)?, &channel)?,
        Cmd::Log { tail } => cmd_log(&Project::open(&root)?, tail)?,
        Cmd::AuditVerify => cmd_audit_verify(&Project::open(&root)?)?,
        Cmd::Proxy { listen } => {
            let project = Project::open(&root)?;
            let cfg = pw_proxy::Config {
                upstream: project.config.upstream.clone(),
                root: project.root.clone(),
                policy: project.policy(),
            };
            println!("PatchWork proxy on http://{listen} -> {}", project.config.upstream);
            pw_proxy::serve(cfg, listen).await?;
        }
        Cmd::Browser { cmd } => match cmd {
            BrowserCmd::Install { with_deps } => {
                browser::install(with_deps).await?;
                println!("Chromium is installed for Playwright.");
            }
            BrowserCmd::Status => println!("Chromium {}", browser::status().await?),
        },
        Cmd::Snapshot { url, out, headed } => {
            let count = browser::snapshot(&url, &out, headed).await?;
            println!("captured {count} elements to {}", out.display());
        }
        Cmd::Edit { url, author } => {
            let project = Project::open(&root)?;
            browser::edit(&url, &project.root, &author).await?;
        }
        Cmd::Train { cmd } => match cmd {
            TrainCmd::Kaggle { cmd } => match cmd {
                KaggleCmd::Plan {
                    accounts, coordinator_url, out, gpus, mode, inner_steps, sync_every, rounds, micro_batch,
                    grad_accum, lr, warmup_steps, total_steps, max_minutes, training_dir,
                } => {
                    let text = std::fs::read_to_string(&accounts).with_context(|| format!("reading {}", accounts.display()))?;
                    let mode = match mode.as_str() {
                        "ddp" => pw_train::kaggle::Mode::Ddp,
                        "independent" => pw_train::kaggle::Mode::Independent,
                        other => anyhow::bail!("unknown mode `{other}`; use ddp or independent"),
                    };
                    let cfg = pw_train::kaggle::PlanConfig {
                        coordinator_url,
                        accounts: pw_train::kaggle::parse_accounts(&text)?,
                        gpus_per_account: gpus,
                        mode,
                        inner_steps,
                        sync_every,
                        rounds,
                        micro_batch,
                        grad_accum,
                        lr,
                        warmup_steps,
                        total_steps,
                        max_minutes,
                        training_dir,
                        token: std::env::var("PW_TOKEN").ok().filter(|t| !t.is_empty())
                            .context("set PW_TOKEN to the coordinator token (at least 16 characters)")?,
                    };
                    let plans = pw_train::kaggle::write_plan(&cfg, &out)?;
                    println!(
                        "planned {} groups, {} DiLoCo replicas, {} GPUs in {} (mode {:?})",
                        plans.len(),
                        cfg.replicas(),
                        plans.len() * gpus,
                        out.display(),
                        cfg.mode
                    );
                    println!("nothing was sent to Kaggle. Review {}/PLAN.md, then run `pw train kaggle push`.", out.display());
                }
                KaggleCmd::Push { plan, keys, execute, confirm_accounts, i_have_checked_kaggle_rules } => {
                    let plans = pw_train::kaggle::load_plans(&plan)?;
                    if execute {
                        pw_train::kaggle::push_allowed(plans.len(), confirm_accounts.unwrap_or(0), i_have_checked_kaggle_rules)?;
                        let text = std::fs::read_to_string(&keys).with_context(|| format!("reading {}", keys.display()))?;
                        let map = pw_train::kaggle::parse_keys(&text)?;
                        pw_train::kaggle::push(&plans, &map, false)?;
                    } else {
                        println!("dry run: {} kernels would be pushed. Add --execute with the confirmations to push.", plans.len());
                        // Dry run prints the commands and never reads the keys file.
                        let placeholder: std::collections::BTreeMap<String, String> =
                            plans.iter().map(|p| (p.alias.clone(), "<from keys file>".to_string())).collect();
                        pw_train::kaggle::push(&plans, &placeholder, true)?;
                    }
                }
            },
            TrainCmd::Bench { sites, variants, seed, json } => {
                let r = pw_train::bench::run(sites, variants, seed)?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&r)?);
                } else {
                    let line = |name: &str, t: &pw_train::bench::Tally| {
                        println!(
                            "{name:<34} cases {:>6}  correct {:>6.1}%  wrong-match {:>6.2}%  abstain {:>6.1}%",
                            t.cases,
                            100.0 * t.accuracy(),
                            100.0 * t.wrong_rate(),
                            100.0 * t.abstain_rate()
                        )
                    };
                    println!("held-out sites {}, variants per site {}", r.sites, r.variants);
                    line("heuristic, target present", &r.present);
                    line("selector-only, target present", &r.selector_only_present);
                    line("heuristic, target deleted", &r.deleted);
                    line("selector-only, target deleted", &r.selector_only_deleted);
                    println!(
                        "wrong matches a text-identity assertion would catch: heuristic {}/{} (deleted) and {}/{} (present); selector-only {}/{} (deleted)",
                        r.deleted.wrong_caught_by_identity,
                        r.deleted.wrong_match,
                        r.present.wrong_caught_by_identity,
                        r.present.wrong_match,
                        r.selector_only_deleted.wrong_caught_by_identity,
                        r.selector_only_deleted.wrong_match
                    );
                }
            }
            TrainCmd::Serve { init, state, serve, listen, workers, quorum, timeout_minutes, outer_lr, outer_momentum } => {
                let initial = pw_train::coordinator::parse_safetensors(&std::fs::read(&init).with_context(|| format!("reading {}", init.display()))?)?;
                // Public URLs are reachable by anyone, so the token is mandatory. Every worker sends it too.
                let token = std::env::var("PW_TOKEN").ok().filter(|t| !t.is_empty()).context("set PW_TOKEN (at least 16 random characters)")?;
                anyhow::ensure!(token.len() >= 16, "PW_TOKEN must be at least 16 characters");
                let cfg = pw_train::coordinator::Config {
                    workers,
                    quorum: quorum.unwrap_or(workers),
                    timeout: std::time::Duration::from_secs_f64(timeout_minutes * 60.0),
                    outer: pw_train::coordinator::OuterOpt { lr: outer_lr, momentum: outer_momentum },
                    token: Some(token),
                    state_dir: state,
                    serve_dir: serve,
                };
                println!("coordinator on http://{listen} for {workers} workers, {} parameters", initial.len());
                pw_train::coordinator::serve(cfg, initial, listen).await?;
            }
            TrainCmd::GenData { out, sites, variants, author, seed, null_fraction } => {
                let cfg = pw_train::datagen::GenConfig {
                    seed,
                    sites,
                    variants_per_site: variants,
                    author_per_site: author,
                    heldout_every: 10,
                    null_fraction,
                };
                let stats = pw_train::datagen::write_dataset(&out, &cfg)?;
                println!("{}", serde_json::to_string_pretty(&stats)?);
                println!("wrote {}", out.display());
            }
        },
        Cmd::Verify { base, channel } => {
            let project = Project::open(&root)?;
            if cmd_verify(&project, &base, &channel).await? {
                return Ok(ExitCode::from(1));
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}
