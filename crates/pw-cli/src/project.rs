//! Project layout and the browser-free operations on it: patch files, snapshots,
//! re-anchoring, conflict checks, and release planning.
//!
//! Layout:
//!   pw.toml          config (upstream, beacon policy)
//!   patches/*.yaml   one patch per file
//!   releases/*.json  immutable manifests
//!   channels/*.txt   channel pointers
//!   audit/           hash-chained event log
//!   snapshots/       captured pages (<name>.html + <name>.rects.json)

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use pw_core::anchor::Anchor;
use pw_core::audit::AuditLog;
use pw_core::conflict::{self, Conflict, Target};
use pw_core::patch::{Patch, Status as ReviewStatus};
use pw_core::relocate::{self, Config as RelocateConfig, Heuristic, Outcome, Status};
use pw_core::release::{self, BeaconPolicy, Manifest};
use pw_core::snapshot::{Rect, Snapshot};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    pub upstream: String,
    pub beacon_window_ms: u64,
    pub beacon_min_errors: usize,
    pub beacon_min_sessions: usize,
    /// Channels that publish only approved patches, and only with a passing verification.
    #[serde(default = "default_protected")]
    pub protected_channels: Vec<String>,
    /// What visual checks compare against: the channel users see now.
    #[serde(default = "default_baseline")]
    pub baseline_channel: String,
    /// Share of pixels that may differ outside the patched elements before a visual check fails. Exact by default:
    /// raise it for pages with animation or ads, and say why in the project's notes.
    #[serde(default = "default_tolerance")]
    pub visual_tolerance: f64,
}

fn default_protected() -> Vec<String> {
    vec!["production".into()]
}

fn default_baseline() -> String {
    "production".into()
}

fn default_tolerance() -> f64 {
    0.0
}

/// The name in the audit trail. A flag or `PW_ACTOR` wins, then the login name. It is declared, not checked.
pub fn resolve_actor(flag: Option<&str>) -> String {
    flag.map(str::to_string)
        .or_else(|| std::env::var("PW_ACTOR").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("USER").ok().filter(|u| !u.is_empty()))
        .unwrap_or_else(|| "unknown".into())
}

pub struct Project {
    pub root: PathBuf,
    pub config: Config,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

impl Project {
    pub fn init(root: &Path, name: &str, upstream: &str) -> Result<()> {
        fs::create_dir_all(root)?;
        let cfg_path = root.join("pw.toml");
        if cfg_path.exists() {
            bail!("{} already exists", cfg_path.display());
        }
        let config = Config {
            name: name.into(),
            upstream: upstream.into(),
            beacon_window_ms: 60_000,
            beacon_min_errors: 5,
            beacon_min_sessions: 3,
            protected_channels: default_protected(),
            baseline_channel: default_baseline(),
            visual_tolerance: default_tolerance(),
        };
        fs::write(&cfg_path, toml::to_string_pretty(&config)?)?;
        for dir in ["patches", "releases", "channels", "audit", "snapshots"] {
            fs::create_dir_all(root.join(dir))?;
        }
        Ok(())
    }

    pub fn open(root: &Path) -> Result<Project> {
        let path = root.join("pw.toml");
        let text = fs::read_to_string(&path)
            .with_context(|| format!("no PatchWork project at {} (run `pw init`)", root.display()))?;
        let config: Config = toml::from_str(&text).with_context(|| format!("reading {}", path.display()))?;
        Ok(Project { root: root.to_path_buf(), config })
    }

    pub fn policy(&self) -> BeaconPolicy {
        BeaconPolicy {
            window_ms: self.config.beacon_window_ms,
            min_errors: self.config.beacon_min_errors,
            min_sessions: self.config.beacon_min_sessions,
        }
    }

    pub fn audit(&self) -> AuditLog {
        AuditLog::open(self.root.join("audit"))
    }

    pub fn patches_dir(&self) -> PathBuf {
        self.root.join("patches")
    }

    /// Protected channels publish only approved patches, with a passing verification (see `gate`).
    pub fn is_protected(&self, channel: &str) -> bool {
        self.config.protected_channels.iter().any(|c| c == channel)
    }

    /// Every patch in `patches/`, validated. One bad file stops the load and names the file.
    pub fn load_patches(&self) -> Result<Vec<Patch>> {
        load_patch_dir(&self.patches_dir())
    }

    /// Every patch with the file it came from, so a change is written back to that file.
    pub fn load_patch_files(&self) -> Result<Vec<(PathBuf, Patch)>> {
        load_patch_files(&self.patches_dir())
    }

    pub fn find_patch(&self, id: &str) -> Result<(PathBuf, Patch)> {
        self.load_patch_files()?
            .into_iter()
            .find(|(_, p)| p.id == id)
            .with_context(|| format!("no patch `{id}` in {}", self.patches_dir().display()))
    }

    pub fn save_patch(&self, patch: &Patch) -> Result<PathBuf> {
        fs::create_dir_all(self.patches_dir())?;
        let path = self.patches_dir().join(format!("{}.yaml", patch.id));
        if path.exists() {
            bail!("patch `{}` already exists at {}", patch.id, path.display());
        }
        fs::write(&path, patch.to_yaml()?)?;
        Ok(path)
    }
}

pub fn write_patch_file(path: &Path, patch: &Patch) -> Result<()> {
    fs::write(path, patch.to_yaml()?).with_context(|| format!("writing {}", path.display()))
}

pub fn load_patch_dir(dir: &Path) -> Result<Vec<Patch>> {
    Ok(load_patch_files(dir)?.into_iter().map(|(_, patch)| patch).collect())
}

pub fn load_patch_files(dir: &Path) -> Result<Vec<(PathBuf, Patch)>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "yaml" || x == "yml"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let patch = Patch::read(&p).with_context(|| format!("invalid patch {}", p.display()))?;
            Ok((p, patch))
        })
        .collect()
}

/// Parse `/cart=path/to/cart.html` into a page path and a file path.
pub fn parse_pair(s: &str) -> std::result::Result<(String, PathBuf), String> {
    let (page, path) = s.split_once('=').ok_or_else(|| format!("expected PAGE=FILE, got `{s}`"))?;
    if !page.starts_with('/') {
        return Err(format!("page `{page}` must start with `/`"));
    }
    Ok((page.to_string(), PathBuf::from(path)))
}

/// Element rectangles, one per element in document order: `[x, y, w, h]` or `null`.
pub fn parse_rects(text: &str) -> Result<Vec<Option<Rect>>> {
    let value: serde_json::Value = serde_json::from_str(text).context("rects file is not JSON")?;
    let items = value.as_array().context("rects file must be a JSON array")?;
    items
        .iter()
        .map(|item| match item {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::Array(v) if v.len() == 4 => {
                let n: Vec<f64> = v.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
                if n.iter().any(|x| x.is_nan()) {
                    bail!("rect entries must be numbers");
                }
                Ok(Some(Rect { x: n[0], y: n[1], w: n[2], h: n[3] }))
            }
            _ => bail!("each rect must be null or [x, y, w, h]"),
        })
        .collect()
}

/// Load a snapshot. Rectangles come from `<name>.rects.json` next to the HTML when present.
pub fn load_snapshot(html: &Path) -> Result<Snapshot> {
    let text = fs::read_to_string(html).with_context(|| format!("reading {}", html.display()))?;
    let snap = Snapshot::parse(&text);
    let rects_path = html.with_extension("rects.json");
    if rects_path.exists() {
        let rects = parse_rects(&fs::read_to_string(&rects_path)?)?;
        if rects.len() != snap.len() {
            bail!(
                "{} lists {} rectangles but the page has {} elements; recapture the snapshot",
                rects_path.display(),
                rects.len(),
                snap.len()
            );
        }
        return Ok(snap.with_rects(&rects));
    }
    Ok(snap)
}

pub fn load_snapshots(pairs: &[(String, PathBuf)]) -> Result<BTreeMap<String, Snapshot>> {
    let mut out = BTreeMap::new();
    for (page, path) in pairs {
        out.insert(page.clone(), load_snapshot(path)?);
    }
    Ok(out)
}

/// The model, when the command was given `--model`. Loaded once per process.
static MODEL: std::sync::OnceLock<pw_model::ModelChooser> = std::sync::OnceLock::new();

/// Load the model for this process. Refuses a folder that does not load, so a typo cannot silently fall back.
pub fn install_model(dir: &Path) -> Result<()> {
    let model = pw_model::Model::load(dir).with_context(|| format!("loading the model in {}", dir.display()))?;
    let accept = RelocateConfig::default().model_accept;
    let _ = MODEL.set(pw_model::ModelChooser::new(model, accept));
    Ok(())
}

pub fn relocate_one(patch: &Patch, snap: &Snapshot) -> Result<Outcome> {
    let cfg = RelocateConfig::default();
    let heuristic = Heuristic { cfg };
    match MODEL.get() {
        Some(model) => {
            let chooser = pw_model::ModelFirst { model, fallback: &heuristic };
            Ok(relocate::relocate(&patch.anchor, snap, &chooser, &cfg)?)
        }
        None => Ok(relocate::relocate(&patch.anchor, snap, &heuristic, &cfg)?),
    }
}

pub fn status_label(s: Status) -> &'static str {
    match s {
        Status::Valid => "valid",
        Status::Relocated => "relocated",
        Status::Ambiguous => "ambiguous",
        Status::Broken => "broken",
    }
}

/// Patch with its anchor rebuilt from the element it now resolves to.
pub fn rebase(patch: &Patch, snap: &Snapshot, element: u32) -> Result<Patch> {
    let el = snap.element(element).context("element index outside the snapshot")?;
    let mut next = patch.clone();
    next.anchor = Anchor::from_element(el, snap);
    Ok(next)
}

pub struct ReleasePlan {
    pub manifest: Manifest,
    pub relocated: Vec<String>,
    pub refused: Vec<(String, String)>,
    pub conflicts: Vec<Conflict>,
    /// Rejected patches. They stay in the project, but they are not part of any release.
    pub withdrawn: Vec<String>,
}

impl ReleasePlan {
    pub fn is_clear(&self) -> bool {
        self.refused.is_empty() && !self.conflicts.iter().any(|c| c.hard)
    }
}

/// Re-anchor every patch against its page's snapshot, look for conflicts, and build the manifest.
/// Anything that is not valid or relocated, and any hard conflict, blocks the release.
pub fn plan_release(project: &Project, channel: &str, snapshots: &BTreeMap<String, Snapshot>) -> Result<ReleasePlan> {
    let patches = project.load_patches()?;
    if patches.is_empty() {
        bail!("no patches to release");
    }
    let mut refused = Vec::new();
    let mut relocated = Vec::new();
    let mut rebased: Vec<Patch> = Vec::new();
    let mut withdrawn = Vec::new();
    let mut targets_by_page: BTreeMap<String, Vec<Target>> = BTreeMap::new();

    for patch in &patches {
        if patch.status == ReviewStatus::Rejected {
            withdrawn.push(patch.id.clone());
            continue;
        }
        let Some(snap) = snapshots.get(&patch.page) else {
            refused.push((patch.id.clone(), format!("no snapshot for {} (pass --snapshot {}=FILE)", patch.page, patch.page)));
            continue;
        };
        let outcome = relocate_one(patch, snap)?;
        match (outcome.status, outcome.element) {
            (Status::Valid, Some(el)) => {
                targets_by_page.entry(patch.page.clone()).or_default().push(Target {
                    patch: patch.id.clone(),
                    node: el,
                    fingerprint: patch.anchor.fingerprint.clone(),
                    ops: patch.ops.clone(),
                });
                rebased.push(patch.clone());
            }
            (Status::Relocated, Some(el)) => {
                relocated.push(patch.id.clone());
                let next = rebase(patch, snap, el)?;
                targets_by_page.entry(patch.page.clone()).or_default().push(Target {
                    patch: patch.id.clone(),
                    node: el,
                    fingerprint: next.anchor.fingerprint.clone(),
                    ops: patch.ops.clone(),
                });
                rebased.push(next);
            }
            (status, _) => refused.push((patch.id.clone(), format!("{} ({})", status_label(status), outcome.reason))),
        }
    }

    let mut conflicts = Vec::new();
    for (page, targets) in &targets_by_page {
        if let Some(snap) = snapshots.get(page) {
            conflicts.extend(conflict::detect(targets, snap));
        }
    }

    if rebased.is_empty() && refused.is_empty() {
        bail!("no patches to release: every patch is rejected");
    }
    let parent = release::current(&project.root, channel)?;
    let manifest = Manifest::build(channel, parent, now_ms(), &rebased)?;
    Ok(ReleasePlan { manifest, relocated, refused, conflicts, withdrawn })
}

pub fn release_targets(patches: &[Patch], snap: &Snapshot, page: &str) -> Vec<Target> {
    patches
        .iter()
        .filter(|p| p.page == page)
        .filter_map(|p| {
            let outcome = relocate_one(p, snap).ok()?;
            match (outcome.status, outcome.element) {
                (Status::Valid | Status::Relocated, Some(el)) => Some(Target {
                    patch: p.id.clone(),
                    node: el,
                    fingerprint: p.anchor.fingerprint.clone(),
                    ops: p.ops.clone(),
                }),
                _ => None,
            }
        })
        .collect()
}
