//! `pw`: patch live websites without rebuilding them.
//!
//! Exit codes: 0 success; 1 error or a blocked release or failed assertion;
//! 2 `check` found patches that are broken or need a human.

mod browser;
mod gate;
mod project;

use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use pw_core::audit::GENESIS;
use pw_core::conflict;
use pw_core::patch::{Patch, Status as ReviewStatus};
use pw_core::relocate::Status;
use pw_core::release;
use pw_core::review::{self, Approval};
use pw_core::verify::{self, Outcome};
use serde_json::json;

use gate::Waivers;
use project::{parse_pair, Project};

#[derive(Parser)]
#[command(name = "pw", version, about = "PatchWork: patch live websites without rebuilding them")]
struct Cli {
    /// Project directory (the one holding pw.toml)
    #[arg(long, global = true, default_value = ".")]
    project: PathBuf,
    /// Who is acting, for the audit trail. Defaults to your login name. It is declared, not checked.
    #[arg(long, global = true)]
    actor: Option<String>,
    /// A trained PatchWork model folder (config.json, model.safetensors, tokenizer.json). When given, the model
    /// decides relocations it is confident about, and the heuristic decides the rest. Experimental: on the
    /// synthetic benchmark it mis-matched deleted targets more often than the heuristic (docs/DESIGN.md, section 6.5).
    #[arg(long, global = true, value_name = "DIR")]
    model: Option<PathBuf>,
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
        /// Copy the other branch's new patches into this project. Refused when there are hard conflicts.
        /// Copied patches keep their status, but they need approval in this log before a protected release.
        #[arg(long)]
        write: bool,
    },
    /// Publish patches to a channel. Protected channels (production by default) need approved patches and a
    /// passing verification of the same content
    Release {
        #[arg(long, default_value = "production")]
        channel: String,
        #[arg(long = "snapshot", value_parser = parse_pair)]
        snapshots: Vec<(String, PathBuf)>,
        /// Show what would happen, including every reason it would be blocked. Publishes nothing
        #[arg(long)]
        dry_run: bool,
        /// Allow patches that were relocated by the heuristic. Verify them on a preview channel first:
        /// a relocation is only as good as its assertions.
        #[arg(long)]
        accept_relocated: bool,
        /// Publish to a protected channel without a passing verification of this content. The reason is audited.
        #[arg(long, value_name = "REASON")]
        waive_verification: Option<String>,
        /// Publish to a protected channel with patches that are not approved. The reason is audited.
        #[arg(long, value_name = "REASON")]
        waive_review: Option<String>,
    },
    /// Review patches: propose, approve, reject, and list their state
    Review {
        #[command(subcommand)]
        cmd: ReviewCmd,
    },
    /// Point a channel back to an earlier release: its parent (default), a release you name, or the newest
    /// release whose patches passed verification
    Rollback {
        #[arg(long, default_value = "production")]
        channel: String,
        /// Serve this release (see `pw log` for ids)
        #[arg(long, value_name = "RELEASE_ID", conflicts_with = "last_verified")]
        to: Option<String>,
        /// Serve the newest earlier release with a complete, passing verification record
        #[arg(long)]
        last_verified: bool,
    },
    /// Conflicts among active patches, and how to resolve them
    Conflicts {
        #[command(subcommand)]
        cmd: ConflictCmd,
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
    /// Open a page with the visual editor; each Save writes a draft patch file
    Edit {
        url: String,
        /// The patch's author field. Defaults to the actor
        #[arg(long)]
        author: Option<String>,
    },
    /// Model training: data generation, the coordinator, and the Kaggle plan
    Train {
        #[command(subcommand)]
        cmd: TrainCmd,
    },
    /// Check a channel's release in a real browser through the proxy, and write a verification record
    Verify {
        /// Proxy base URL, for example http://127.0.0.1:8787
        #[arg(long)]
        base: String,
        #[arg(long, default_value = "production")]
        channel: String,
        /// Run click checks (clickNavigatesTo). A click can submit a form or trigger a real action, so use a
        /// copy of the site where that is safe
        #[arg(long)]
        interact: bool,
        /// Skip visual checks. A record without them cannot pass a protected channel's gate
        #[arg(long)]
        no_visual: bool,
    },
}

#[derive(Subcommand)]
enum ConflictCmd {
    /// List conflicts between active (not rejected) patches. Exit 1 when a hard conflict remains
    List {
        #[arg(long = "snapshot", value_parser = parse_pair)]
        snapshots: Vec<(String, PathBuf)>,
    },
    /// Resolve a conflict by rejecting the patch you drop. A rejected patch is left out of every release
    Resolve {
        /// The patch that stays
        #[arg(long)]
        keep: String,
        /// The patch that is dropped (rejected)
        #[arg(long)]
        drop: String,
        #[arg(long)]
        reason: String,
    },
}

#[derive(Subcommand)]
enum ReviewCmd {
    /// Ask for a review: a draft or rejected patch becomes proposed
    Propose { id: String },
    /// Approve a proposed patch. The approver must not be its author, unless --allow-self-review is given
    Approve {
        id: String,
        /// Approve your own patch. The audit trail records that it was self-reviewed
        #[arg(long)]
        allow_self_review: bool,
    },
    /// Reject a patch. This also withdraws an approval
    Reject {
        id: String,
        #[arg(long)]
        reason: String,
    },
    /// Show each patch's state, and whether its approval still holds
    List,
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
        /// Also score the trained model (an exported folder). Slower: the model runs on the CPU.
        #[arg(long, value_name = "DIR")]
        model: Option<PathBuf>,
        /// Confidence bars to sweep, for example 0.5,0.8,0.95 (needs --model). The model is asked once per case,
        /// and each bar is applied to the same answers.
        #[arg(long, value_name = "BARS", value_delimiter = ',', requires = "model")]
        accept: Vec<f64>,
        /// Relocation settings to use, as key=value pairs, for example floor=0.4,heuristic_margin=0.1.
        /// Names are those of pw_core::relocate::Config. A misspelt name is an error.
        #[arg(long, value_name = "K=V,...")]
        set: Option<String>,
        /// How targets drift. `mixed` is the protocol's default. `compound` drops each target's id, changes its
        /// text and moves it: the case the release gate cannot see on its own. Not with --accept.
        #[arg(long, default_value = "mixed", value_parser = ["mixed", "compound"])]
        drift: String,
        /// Score the matcher model (a folder from `training/match.py`) instead of the heuristic. Uses the
        /// thresholds from --set (matcher_accept, matcher_margin, matcher_review).
        #[arg(long, value_name = "DIR")]
        matcher: Option<PathBuf>,
        /// With --matcher: run the pre-registered threshold grid and select one setting by the protocol's rule
        #[arg(long, requires = "matcher")]
        matcher_grid: bool,
        #[arg(long)]
        json: bool,
    },
    /// Score the combined tracker (heuristic, then matcher) on slices of synthetic sites. With --grid, select a
    /// setting by the dev objective summed over the slices (docs/DESIGN.md, section 6.10)
    Track {
        /// The matcher folder (from `training/match.py`)
        #[arg(long, value_name = "DIR")]
        matcher: PathBuf,
        /// Slices as SEED:mixed or SEED:compound, comma-separated
        #[arg(long, value_delimiter = ',', required = true)]
        slices: Vec<String>,
        #[arg(long, default_value_t = 300)]
        sites: usize,
        #[arg(long, default_value_t = 3)]
        variants: usize,
        /// Run the whole grid and select one setting
        #[arg(long, conflicts_with = "setting")]
        grid: bool,
        /// One setting, as accept,margin,review,veto
        #[arg(long, value_name = "A,M,R,V")]
        setting: Option<String>,
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
        /// Outer Nesterov momentum. 0 is plain averaging. 0.9 (the DiLoCo value) overshot in our run.
        #[arg(long, default_value_t = 0.0)]
        outer_momentum: f32,
    },
    /// Generate (anchor, candidate) pairs for the matcher model, with hard negatives
    MatchData {
        #[arg(long, default_value = "data/match")]
        out: PathBuf,
        #[arg(long, default_value_t = 2000)]
        sites: usize,
        #[arg(long, default_value_t = 3)]
        variants: usize,
        #[arg(long, default_value_t = 5001)]
        seed: u64,
        /// Prefilter candidates per variant (the heuristic sees the same list)
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        /// Every n-th site goes to the validation file
        #[arg(long, default_value_t = 10)]
        validation_every: usize,
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

fn cmd_check(project: &Project, actor: &str, pairs: &[(String, PathBuf)], write: bool, as_json: bool) -> Result<bool> {
    let snaps = project::load_snapshots(pairs)?;
    let files = project.load_patch_files()?;
    let mut rows = Vec::new();
    let mut needs_human = false;
    for (path, patch) in &files {
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
                // The file keeps its name. The new anchor changes the content, so an approval of the old one no longer counts.
                project::write_patch_file(path, &project::rebase(patch, snap, el)?)?;
                let confidence = (out.confidence * 1000.0).round() / 1000.0;
                project.audit().append(actor, "patch.relocate", json!({"patch": patch.id, "page": patch.page, "element": el, "confidence": confidence}))?;
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

/// Compare this project's patches with another branch's. Hard conflicts are reported and block a write.
/// With `write`, the other branch's new patches are copied in. A patch with the same id and different
/// content is a conflict, because one branch would silently replace the other's patch.
fn cmd_merge(project: &Project, actor: &str, with: &PathBuf, pairs: &[(String, PathBuf)], write: bool) -> Result<bool> {
    let snaps = project::load_snapshots(pairs)?;
    let mine = project.load_patch_files()?;
    let theirs = project::load_patch_files(with)?;
    let theirs_ids: Vec<String> = theirs.iter().map(|(_, p)| p.id.clone()).collect();
    let mut all: Vec<_> = mine.iter().map(|(_, p)| p.clone()).filter(|p| !theirs_ids.contains(&p.id)).collect();
    all.extend(theirs.iter().map(|(_, p)| p.clone()));

    let mut hard = 0usize;
    let mut added: Vec<(PathBuf, String)> = Vec::new();
    let mut unchanged = Vec::new();
    for (path, theirs_patch) in &theirs {
        let Some((_, mine_patch)) = mine.iter().find(|(_, m)| m.id == theirs_patch.id) else {
            let name = path.file_name().context("a patch file without a name")?;
            let dest = project.patches_dir().join(name);
            if dest.exists() {
                hard += 1;
                println!("CONFLICT file name: {} is already used by another patch", dest.display());
            } else {
                added.push((path.clone(), theirs_patch.id.clone()));
            }
            continue;
        };
        if mine_patch.sha256()? == theirs_patch.sha256()? {
            unchanged.push(theirs_patch.id.clone());
        } else {
            hard += 1;
            println!("CONFLICT same id on {}: patch `{}` differs between the two branches", theirs_patch.page, theirs_patch.id);
        }
    }

    let mut warnings = 0usize;
    for (page, snap) in &snaps {
        let targets = project::release_targets(&all, snap, page);
        for c in conflict::detect(&targets, snap) {
            if c.hard {
                hard += 1;
            } else {
                warnings += 1;
            }
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
    if hard == 0 {
        println!("no hard conflicts across {} patches", all.len());
    }

    let written = write && hard == 0;
    if written {
        fs::create_dir_all(project.patches_dir())?;
        for (path, id) in &added {
            let name = path.file_name().context("a patch file without a name")?;
            let dest = project.patches_dir().join(name);
            fs::copy(path, &dest).with_context(|| format!("copying {}", path.display()))?;
            println!("added {id} ({})", dest.display());
        }
    } else if write {
        println!("nothing written: resolve the conflicts above first");
    }
    let added_ids: Vec<&str> = added.iter().map(|(_, id)| id.as_str()).collect();
    project.audit().append(
        actor,
        "merge",
        json!({"with": with.display().to_string(), "written": written, "added": added_ids, "unchanged": unchanged, "hard_conflicts": hard, "warnings": warnings}),
    )?;
    Ok(hard > 0)
}

/// Publish the plan to `channel`. Blockers are listed together, so a dry run shows everything that would stop
/// the real run. A protected channel's gate (see `gate`) can be waived with a reason, and the waiver is audited.
async fn cmd_release(project: &Project, actor: &str, channel: &str, pairs: &[(String, PathBuf)], dry_run: bool, accept_relocated: bool, waivers: Waivers) -> Result<()> {
    let snaps = project::load_snapshots(pairs)?;
    let plan = project::plan_release(project, channel, &snaps)?;
    for id in &plan.relocated {
        println!("relocated  {id} (anchor will be rewritten in the release)");
    }
    for (id, why) in &plan.refused {
        println!("refused    {id}: {why}");
    }
    for id in &plan.withdrawn {
        println!("withdrawn  {id} (rejected: left out of this release)");
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
    if let Some(current) = release::current(&project.root, channel)? {
        if release::read_release(&project.root, &current)?.patches == plan.manifest.patches {
            println!("{channel} already serves this content as release {current}; nothing to do");
            return Ok(());
        }
    }

    let mut blockers = Vec::new();
    if !plan.is_clear() {
        blockers.push("fix the refused patches and hard conflicts listed above".to_string());
    }
    if !plan.relocated.is_empty() && !accept_relocated {
        blockers.push(format!(
            "{} patch(es) were relocated by the heuristic ({}); verify them on a preview channel, then pass --accept-relocated",
            plan.relocated.len(),
            plan.relocated.join(", ")
        ));
    }
    // The gate only has a meaning for a release that is otherwise clear.
    let gate = if plan.is_clear() { gate::evaluate(project, channel, &plan, &waivers)? } else { gate::Gate::default() };
    blockers.extend(gate.blockers.iter().cloned());

    if !blockers.is_empty() {
        if dry_run {
            println!("dry run: the release to {channel} would be BLOCKED by {} reason(s):", blockers.len());
            for b in &blockers {
                println!("  - {b}");
            }
            bail!("dry run: nothing was published");
        }
        project.audit().append(actor, "release.blocked", json!({"channel": channel, "id": plan.manifest.id, "reasons": blockers}))?;
        bail!("release blocked:\n  - {}", blockers.join("\n  - "));
    }
    if dry_run {
        println!("dry run: would publish release {} to {channel} ({} patches)", plan.manifest.id, plan.manifest.patches.len());
        for (name, reason) in &gate.waived {
            println!("  would waive the {name} gate: {reason}");
        }
        return Ok(());
    }

    release::write_release(&project.root, &plan.manifest)?;
    release::set_channel(&project.root, channel, &plan.manifest.id)?;
    let ids: Vec<&str> = plan.manifest.patches.iter().map(|p| p.id.as_str()).collect();
    let digest = release::patches_digest(&plan.manifest.patches)?;
    let audit = project.audit();
    audit.append(
        actor,
        "release",
        json!({
            "channel": channel,
            "id": plan.manifest.id,
            "parent": plan.manifest.parent,
            "patches": ids,
            "relocated": plan.relocated,
            "withdrawn": plan.withdrawn,
            "digest": digest,
            "verified_ms": gate.verified_ms,
        }),
    )?;
    for (name, reason) in &gate.waived {
        audit.append(actor, "release.waiver", json!({"channel": channel, "id": plan.manifest.id, "gate": name, "reason": reason}))?;
    }
    println!("released {} to {channel} ({} patches)", plan.manifest.id, ids.len());
    for (name, _) in &gate.waived {
        println!("  waived the {name} gate; the waiver is in the audit log");
    }
    Ok(())
}

/// The newest release before `from` whose patches have a complete, passing verification record.
fn last_verified_release(project: &Project, from: &str) -> Result<String> {
    let mut id = from.to_string();
    loop {
        let manifest = release::read_release(&project.root, &id)?;
        if id != from {
            let digest = release::patches_digest(&manifest.patches)?;
            if verify::latest(&project.root, &digest)?.is_some_and(|r| r.complete()) {
                return Ok(id);
            }
        }
        match manifest.parent {
            Some(parent) => id = parent,
            None => bail!("no earlier release has a complete verification record; run `pw verify` on it first"),
        }
    }
}

fn cmd_rollback(project: &Project, actor: &str, channel: &str, to: Option<String>, last_verified: bool) -> Result<()> {
    let from = release::current(&project.root, channel)?.with_context(|| format!("channel `{channel}` has no release"))?;
    let (target, mode) = match (to, last_verified) {
        (Some(id), _) => {
            release::read_release(&project.root, &id)?;
            (id, "to")
        }
        (None, true) => (last_verified_release(project, &from)?, "last_verified"),
        (None, false) => {
            let parent = release::read_release(&project.root, &from)?
                .parent
                .with_context(|| format!("release `{from}` has no parent to roll back to"))?;
            (parent, "parent")
        }
    };
    release::set_channel(&project.root, channel, &target)?;
    project.audit().append(actor, "rollback", json!({"channel": channel, "from": from, "to": target, "mode": mode}))?;
    println!("{channel} now serves release {target}");
    Ok(())
}

/// Conflicts among active patches. `resolve` rejects the dropped patch, so no release includes it.
fn cmd_conflicts(project: &Project, actor: &str, cmd: ConflictCmd) -> Result<bool> {
    match cmd {
        ConflictCmd::List { snapshots } => {
            let snaps = project::load_snapshots(&snapshots)?;
            let active: Vec<Patch> = project.load_patches()?.into_iter().filter(|p| p.status != ReviewStatus::Rejected).collect();
            let mut hard = false;
            let mut found = 0usize;
            for (page, snap) in &snaps {
                let targets = project::release_targets(&active, snap, page);
                for c in conflict::detect(&targets, snap) {
                    found += 1;
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
            if found == 0 {
                println!("no conflicts among {} active patches", active.len());
            }
            Ok(hard)
        }
        ConflictCmd::Resolve { keep, drop, reason } => {
            anyhow::ensure!(keep != drop, "--keep and --drop must name different patches");
            let (_, kept) = project.find_patch(&keep)?;
            anyhow::ensure!(kept.status != ReviewStatus::Rejected, "`{keep}` is rejected; keep an active patch");
            let (path, mut dropped) = project.find_patch(&drop)?;
            dropped.status = ReviewStatus::Rejected;
            dropped.reviewer = None;
            project::write_patch_file(&path, &dropped)?;
            let digest = dropped.review_digest()?;
            let reason = format!("superseded by {keep}: {reason}");
            project.audit().append(actor, review::REJECT, json!({"patch": drop, "digest": digest, "reason": reason}))?;
            project.audit().append(actor, "conflict.resolve", json!({"keep": keep, "drop": drop, "reason": reason}))?;
            println!("resolved: `{drop}` is rejected and is no longer part of any release (kept `{keep}`)");
            Ok(false)
        }
    }
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

/// Check the channel's release in a real browser, write a verification record, and audit it. Returns true
/// when the record is not complete (a check failed or did not run).
async fn cmd_verify(project: &Project, actor: &str, base: &str, channel: &str, interact: bool, no_visual: bool) -> Result<bool> {
    let id = release::current(&project.root, channel)?
        .with_context(|| format!("channel `{channel}` has no release; run `pw release` first"))?;
    let manifest = release::read_release(&project.root, &id)?;
    let digest = release::patches_digest(&manifest.patches)?;
    let visual = (!no_visual).then(|| browser::Visual {
        baseline: project.config.baseline_channel.clone(),
        tolerance: project.config.visual_tolerance,
    });
    let options = browser::Options { interact, visual };
    let checks = browser::verify(base, channel, &manifest.patches, &options).await?;
    let record = verify::Record {
        digest: digest.clone(),
        release: id.clone(),
        channel: channel.into(),
        base: base.into(),
        actor: actor.into(),
        ts_ms: now_ms(),
        interact,
        visual: !no_visual,
        checks,
    };
    for c in &record.checks {
        let mark = match c.outcome {
            Outcome::Pass => "pass",
            Outcome::Fail => "FAIL",
            Outcome::NotRun => "skip",
        };
        let detail = if c.detail.is_empty() { String::new() } else { format!("  ({})", c.detail) };
        println!("{mark}  {:<28} {}{detail}", c.patch, c.check);
    }
    if record.checks.is_empty() {
        println!("no checks: release {id} has no patches");
    }
    let path = verify::write(&project.root, &record)?;
    let complete = record.complete();
    let (passed, failed, not_run) = (record.count(Outcome::Pass), record.count(Outcome::Fail), record.count(Outcome::NotRun));
    println!("verification {}: {passed} passed, {failed} failed, {not_run} not run", if complete { "complete" } else { "INCOMPLETE" });
    println!("record {}", path.display());
    project.audit().append(
        actor,
        "verify",
        json!({"channel": channel, "release": id, "digest": digest, "complete": complete, "passed": passed, "failed": failed, "not_run": not_run, "interact": interact, "visual": !no_visual}),
    )?;
    Ok(!complete)
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Review state changes. Every change is an audit event, and an approval names the content it approves.
fn cmd_review(project: &Project, actor: &str, cmd: ReviewCmd) -> Result<()> {
    let audit = project.audit();
    match cmd {
        ReviewCmd::Propose { id } => {
            let (path, mut patch) = project.find_patch(&id)?;
            if !matches!(patch.status, ReviewStatus::Draft | ReviewStatus::Rejected) {
                bail!("`{id}` is {}; only a draft or rejected patch can be proposed", patch.status.label());
            }
            patch.status = ReviewStatus::Proposed;
            patch.reviewer = None;
            project::write_patch_file(&path, &patch)?;
            let digest = patch.review_digest()?;
            audit.append(actor, review::PROPOSE, json!({"patch": id, "digest": digest}))?;
            println!("`{id}` is proposed for review (content {digest})");
        }
        ReviewCmd::Approve { id, allow_self_review } => {
            let (path, mut patch) = project.find_patch(&id)?;
            if patch.status != ReviewStatus::Proposed {
                bail!("`{id}` is {}; propose it first with `pw review propose {id}`", patch.status.label());
            }
            let self_review = patch.author == actor;
            if self_review && !allow_self_review {
                bail!("`{id}` was written by {actor}. Another reviewer should approve it; --allow-self-review records that you did not");
            }
            patch.status = ReviewStatus::Approved;
            patch.reviewer = Some(actor.to_string());
            project::write_patch_file(&path, &patch)?;
            let digest = patch.review_digest()?;
            audit.append(
                actor,
                review::APPROVE,
                json!({"patch": id, "digest": digest, "reviewer": actor, "self_review": self_review}),
            )?;
            let note = if self_review { " (self-review, recorded)" } else { "" };
            println!("`{id}` approved by {actor}{note}");
        }
        ReviewCmd::Reject { id, reason } => {
            let (path, mut patch) = project.find_patch(&id)?;
            if patch.status == ReviewStatus::Rejected {
                bail!("`{id}` is already rejected");
            }
            patch.status = ReviewStatus::Rejected;
            patch.reviewer = None;
            project::write_patch_file(&path, &patch)?;
            let digest = patch.review_digest()?;
            audit.append(actor, review::REJECT, json!({"patch": id, "digest": digest, "reason": reason}))?;
            println!("`{id}` rejected: {reason}");
        }
        ReviewCmd::List => {
            let events = audit.read_all()?;
            let files = project.load_patch_files()?;
            if files.is_empty() {
                println!("no patches in {}", project.patches_dir().display());
            }
            for (_, patch) in files {
                let state = match review::approval(&patch, &events)? {
                    Approval::Approved => "approved".to_string(),
                    Approval::Unproven(reason) => format!("NOT VALID: {reason}"),
                    Approval::NotApproved(status) => status.label().to_string(),
                };
                println!("{:<28} {:<9} author {:<12} {state}", patch.id, patch.status.label(), patch.author);
            }
        }
    }
    Ok(())
}

/// `SEED:mixed` or `SEED:compound`.
fn parse_slice(spec: &str) -> Result<(u64, pw_train::bench::Drift)> {
    let (seed, drift) = spec.split_once(':').with_context(|| format!("a slice is SEED:mixed or SEED:compound, got `{spec}`"))?;
    let drift = match drift {
        "mixed" => pw_train::bench::Drift::Mixed,
        "compound" => pw_train::bench::Drift::Compound,
        other => bail!("unknown drift `{other}`; use mixed or compound"),
    };
    Ok((seed.parse().with_context(|| format!("seed `{seed}` is not a number"))?, drift))
}

/// `accept,margin,review,veto`, four numbers.
fn parse_setting(spec: &str) -> Result<(f64, f64, f64, f64)> {
    let parts: Vec<f64> = spec.split(',').map(|p| p.trim().parse::<f64>()).collect::<std::result::Result<_, _>>().with_context(|| format!("`{spec}` is not four numbers"))?;
    match parts.as_slice() {
        [a, m, r, v] => Ok((*a, *m, *r, *v)),
        _ => bail!("expected accept,margin,review,veto, got `{spec}`"),
    }
}

/// One benchmark table, labelled with who made the choices.
fn print_bench(r: &pw_train::bench::Report, who: &str) {
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
    line(&format!("{who}, target present"), &r.present);
    line("selector-only, target present", &r.selector_only_present);
    line(&format!("{who}, target deleted"), &r.deleted);
    line("selector-only, target deleted", &r.selector_only_deleted);
    println!("objective {}", pw_train::bench::objective(r));
}

/// The benchmark at several confidence bars. The model's answers are the same at every bar. The bar decides
/// whether an answer is applied.
fn print_sweep(s: &pw_train::bench::Sweep) {
    let r = &s.report;
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
    for (title, first) in [("model then heuristic (what --model does)", true), ("model alone", false)] {
        println!("\n{title}");
        println!("{:>6} {:>14} {:>13} {:>15} {:>14} {:>15}", "bar", "present correct", "present wrong", "present abstain", "deleted wrong", "deleted abstain");
        for bar in &s.bars {
            let (present, deleted) = if first {
                (&bar.model.first_present, &bar.model.first_deleted)
            } else {
                (&bar.model.only_present, &bar.model.only_deleted)
            };
            println!(
                "{:>6.3} {:>13.1}% {:>12.2}% {:>14.1}% {:>13.1}% {:>14.1}%",
                bar.accept,
                100.0 * present.accuracy(),
                100.0 * present.wrong_rate(),
                100.0 * present.abstain_rate(),
                100.0 * deleted.wrong_rate(),
                100.0 * deleted.abstain_rate()
            );
        }
    }
}

/// Commands that rewrite patch files or publish relocated anchors. A model's relocation is only worth
/// that once it has been checked, so these print a warning when `--model` is given.
fn rewrites_or_publishes(cmd: &Cmd) -> bool {
    matches!(cmd, Cmd::Check { write: true, .. } | Cmd::Release { accept_relocated: true, dry_run: false, .. })
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    let cli = Cli::parse();
    let actor = project::resolve_actor(cli.actor.as_deref());
    if let Some(dir) = &cli.model {
        project::install_model(dir)?;
        if rewrites_or_publishes(&cli.cmd) {
            eprintln!("warning: --model relocations are experimental (docs/DESIGN.md, section 6.5); review them before you publish");
        }
    }
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
            if cmd_check(&project, &actor, &snapshots, write, json)? {
                return Ok(ExitCode::from(2));
            }
        }
        Cmd::Merge { with, snapshots, write } => {
            let project = Project::open(&root)?;
            if cmd_merge(&project, &actor, &with, &snapshots, write)? {
                return Ok(ExitCode::from(1));
            }
        }
        Cmd::Release { channel, snapshots, dry_run, accept_relocated, waive_verification, waive_review } => {
            let project = Project::open(&root)?;
            let waivers = Waivers { verification: waive_verification, review: waive_review };
            cmd_release(&project, &actor, &channel, &snapshots, dry_run, accept_relocated, waivers).await?;
        }
        Cmd::Review { cmd } => cmd_review(&Project::open(&root)?, &actor, cmd)?,
        Cmd::Rollback { channel, to, last_verified } => cmd_rollback(&Project::open(&root)?, &actor, &channel, to, last_verified)?,
        Cmd::Conflicts { cmd } => {
            if cmd_conflicts(&Project::open(&root)?, &actor, cmd)? {
                return Ok(ExitCode::from(1));
            }
        }
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
            let author = author.unwrap_or_else(|| actor.clone());
            browser::edit(&url, &project.root, &author, &actor).await?;
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
            TrainCmd::Track { matcher, slices, sites, variants, grid, setting, json } => {
                let loaded = pw_model::Matcher::load(&matcher).with_context(|| format!("loading the matcher in {}", matcher.display()))?;
                let parsed: Vec<(u64, pw_train::bench::Drift)> = slices.iter().map(|s| parse_slice(s)).collect::<Result<_>>()?;
                let base = pw_core::relocate::Config::default();
                if grid {
                    let settings = pw_train::bench::combined_grid();
                    // (objective, correct present, minus the work for a person), summed over the slices
                    let mut totals = vec![(0i64, 0i64, 0i64); settings.len()];
                    let mut runs_by_slice = Vec::new();
                    for (seed, drift) in &parsed {
                        let runs = pw_train::bench::run_combined_grid(sites, variants, *seed, *drift, &loaded, base, &settings)?;
                        for (i, (_, _, r)) in runs.iter().enumerate() {
                            totals[i].0 += pw_train::bench::objective(r);
                            totals[i].1 += r.present.correct as i64;
                            totals[i].2 -= (r.present.ambiguous + r.deleted.ambiguous) as i64;
                        }
                        runs_by_slice.push(runs);
                    }
                    let best = (0..settings.len())
                        .max_by_key(|&i| (totals[i].0, totals[i].1, totals[i].2, -(i as i64)))
                        .context("the grid is empty")?;
                    if json {
                        let picks: Vec<_> = runs_by_slice.iter().map(|runs| &runs[best].2).collect();
                        let (a, m, r, v) = settings[best];
                        println!("{}", serde_json::to_string_pretty(&json!({"selected": {"accept": a, "margin": m, "review": r, "veto": v}, "objective_sum": totals[best].0, "per_slice": picks}))?);
                    } else {
                        let mut order: Vec<usize> = (0..settings.len()).collect();
                        order.sort_by_key(|&i| std::cmp::Reverse((totals[i].0, totals[i].1, totals[i].2, -(i as i64))));
                        println!("combined grid over {} settings and {} slices; top 8 by summed objective (selected: *)", settings.len(), parsed.len());
                        println!("{:>1} {:>6} {:>6} {:>6} {:>5} {:>12} {:>12}", "", "accept", "margin", "review", "veto", "objective", "pres correct");
                        for &i in order.iter().take(8) {
                            let (a, m, r, v) = settings[i];
                            let mark = if i == best { "*" } else { " " };
                            println!("{mark} {a:>6.2} {m:>6.2} {r:>6.2} {v:>5.2} {:>12} {:>12}", totals[i].0, totals[i].1);
                        }
                        let (a, m, r, v) = settings[best];
                        println!("selected: accept={a} margin={m} review={r} veto={v}");
                        for ((seed, drift), runs) in parsed.iter().zip(&runs_by_slice) {
                            println!("\nseed {seed} {drift:?}:");
                            print_bench(&runs[best].2, "combined");
                        }
                    }
                    return Ok(ExitCode::SUCCESS);
                }
                let (a, m, r, v) = parse_setting(setting.as_deref().context("give --setting A,M,R,V, or --grid")?)?;
                let settings = [(a, m, r, v)];
                for (seed, drift) in &parsed {
                    let runs = pw_train::bench::run_combined_grid(sites, variants, *seed, *drift, &loaded, base, &settings)?;
                    let (_, _, report) = &runs[0];
                    if json {
                        println!("{}", serde_json::to_string_pretty(report)?);
                    } else {
                        println!("\nseed {seed} {drift:?}:");
                        print_bench(report, "combined");
                    }
                }
                return Ok(ExitCode::SUCCESS);
            }
            TrainCmd::Bench { sites, variants, seed, model, accept, set, drift, matcher, matcher_grid, json } => {
                let drift = match drift.as_str() {
                    "compound" => pw_train::bench::Drift::Compound,
                    _ => pw_train::bench::Drift::Mixed,
                };
                let mut cfg = pw_core::relocate::Config::default();
                if let Some(spec) = &set {
                    cfg.apply(spec)?;
                }
                if let Some(dir) = &matcher {
                    let loaded = pw_model::Matcher::load(dir).with_context(|| format!("loading the matcher in {}", dir.display()))?;
                    if matcher_grid {
                        let runs = pw_train::bench::run_matcher_grid(sites, variants, seed, drift, &loaded, cfg, &pw_train::bench::matcher_grid())?;
                        let best = pw_train::bench::select_setting(&runs).context("the grid is empty")?;
                        if json {
                            let rows: Vec<_> = runs
                                .iter()
                                .map(|(c, r)| json!({"accept": c.matcher_accept, "margin": c.matcher_margin, "review": c.matcher_review, "objective": pw_train::bench::objective(r), "report": r}))
                                .collect();
                            println!("{}", serde_json::to_string_pretty(&json!({"selected": best, "settings": rows}))?);
                        } else {
                            let mut order: Vec<usize> = (0..runs.len()).collect();
                            order.sort_by_key(|&i| std::cmp::Reverse(pw_train::bench::objective(&runs[i].1)));
                            println!("matcher grid, {} settings, seed {seed}; top 10 by objective (selected: row marked *)", runs.len());
                            println!("{:>1} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12}", "", "accept", "margin", "review", "objective", "pres corr", "pres wrong", "pres amb", "dele wrong", "dele amb");
                            for &i in order.iter().take(10).chain(std::iter::once(&best).filter(|b| !order.iter().take(10).any(|o| o == *b))) {
                                let (c, r) = &runs[i];
                                let mark = if i == best { "*" } else { " " };
                                println!(
                                    "{mark} {:>6.2} {:>6.2} {:>6.2} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12}",
                                    c.matcher_accept, c.matcher_margin, c.matcher_review, pw_train::bench::objective(r),
                                    r.present.correct, r.present.wrong_match, r.present.ambiguous, r.deleted.wrong_match, r.deleted.ambiguous
                                );
                            }
                            let (c, r) = &runs[best];
                            println!("selected: matcher_accept={} matcher_margin={} matcher_review={}", c.matcher_accept, c.matcher_margin, c.matcher_review);
                            print_bench(r, "matcher");
                        }
                        return Ok(ExitCode::SUCCESS);
                    }
                    let settings = [(cfg.matcher_accept, cfg.matcher_margin, cfg.matcher_review)];
                    let runs = pw_train::bench::run_matcher_grid(sites, variants, seed, drift, &loaded, cfg, &settings)?;
                    let (_, r) = &runs[0];
                    if json {
                        println!("{}", serde_json::to_string_pretty(r)?);
                    } else {
                        print_bench(r, "matcher");
                    }
                    return Ok(ExitCode::SUCCESS);
                }
                let chooser = match model {
                    Some(dir) => {
                        let loaded = pw_model::Model::load(&dir).with_context(|| format!("loading {}", dir.display()))?;
                        Some(pw_model::ModelChooser::new(loaded, cfg.model_accept))
                    }
                    None => None,
                };
                if !accept.is_empty() {
                    anyhow::ensure!(drift == pw_train::bench::Drift::Mixed, "--drift compound is not available with --accept");
                    let chooser = chooser.as_ref().context("--accept needs --model")?;
                    let sweep = pw_train::bench::run_sweep(sites, variants, seed, chooser, &accept, cfg)?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&sweep)?);
                    } else {
                        print_sweep(&sweep);
                    }
                    return Ok(ExitCode::SUCCESS);
                }
                let r = pw_train::bench::run_drift(sites, variants, seed, chooser.as_ref(), cfg, drift)?;
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
                    if let Some(m) = &r.model {
                        line("model alone, target present", &m.only_present);
                        line("model alone, target deleted", &m.only_deleted);
                        line("model then heuristic, present", &m.first_present);
                        line("model then heuristic, deleted", &m.first_deleted);
                    }
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
            TrainCmd::MatchData { out, sites, variants, seed, candidates, validation_every } => {
                let cfg = pw_train::match_data::Settings { sites, variants, seed, candidates, validation_every };
                let stats = pw_train::match_data::write(&out, &cfg)?;
                println!("{}", serde_json::to_string_pretty(&stats)?);
                println!("wrote {}", out.display());
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
        Cmd::Verify { base, channel, interact, no_visual } => {
            let project = Project::open(&root)?;
            if cmd_verify(&project, &actor, &base, &channel, interact, no_visual).await? {
                return Ok(ExitCode::from(1));
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_commands_that_rewrite_or_publish_warn_about_the_model() {
        let check = |write| Cmd::Check { snapshots: vec![], write, json: false };
        let release = |dry_run, accept_relocated| Cmd::Release {
            channel: "production".into(),
            snapshots: vec![],
            dry_run,
            accept_relocated,
            waive_verification: None,
            waive_review: None,
        };
        assert!(rewrites_or_publishes(&check(true)));
        assert!(!rewrites_or_publishes(&check(false)));
        assert!(rewrites_or_publishes(&release(false, true)));
        assert!(!rewrites_or_publishes(&release(true, true)), "a dry run publishes nothing");
        assert!(!rewrites_or_publishes(&release(false, false)), "without accept_relocated, relocated patches are refused");
    }
}
