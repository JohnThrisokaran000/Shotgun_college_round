//! Distributed training coordinator (DiLoCo-style).
//!
//! Workers train locally for H steps, then send `delta = global - local` as fp16 safetensors.
//! When a quorum of deltas has arrived, the coordinator averages them and applies an outer
//! Nesterov-momentum step:
//!
//!   m     <- mu * m + avg
//!   theta <- theta - lr * (avg + mu * m)
//!
//! With lr = 1 and mu = 0 this is plain model averaging. The defaults are lr 0.7 and mu 0. The DiLoCo
//! paper uses mu 0.9, but with 2000-step rounds and deltas that arrive one version late, that momentum
//! kept adding the same progress and overshot (docs/KAGGLE.md). A round also closes on a timeout once at
//! least one delta has arrived.
//! A delta based on a version more than MAX_STALENESS behind is refused (409), and the worker
//! resyncs from the current global.
//!
//! Memory: each pending delta is kept as raw fp16 bits, half the size of f32. The global model
//! and the momentum are f32. Workers download the global as int8 (one f16 scale per row of each
//! matrix), about a quarter of the f32 bytes. f16 and f32 are available on request.
//!
//! When a token is configured, every /v1 route needs `Authorization: Bearer <token>`. Workers
//! post monitoring events to /v1/events. The coordinator prints each one as a `PW {json}` line
//! and appends it to `events.jsonl` in the state directory.
//!
//! RoundState is written to `state_dir` after every round, so the coordinator can restart.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use half::f16;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

/// Largest accepted delta or payload. A fp16 delta is about 2 bytes per parameter, so this
/// covers models far larger than the smoke test. Axum's default (2 MB) is far too small.
pub const MAX_BODY_BYTES: usize = 512 * 1024 * 1024;

/// A delta may be based on the current global version or up to this many versions earlier.
/// Workers keep training while an update is in flight, so their deltas can arrive one version late.
pub const MAX_STALENESS: u64 = 1;

/// Events per POST to /v1/events, and the body size limit for that route.
pub const MAX_EVENTS_PER_POST: usize = 500;
pub const MAX_EVENT_BODY_BYTES: usize = 1024 * 1024;

/// Name suffix of the per-row scale tensor in an int8 payload.
pub const SCALE_SUFFIX: &str = ".__scale";

/// A flat f32 copy of every tensor, keyed by name, with its shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Params {
    pub fn len(&self) -> usize {
        self.tensors.values().map(|(_, v)| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element-wise `self - other`, keeping names and shapes.
    pub fn minus(&self, other: &Params) -> Result<Params> {
        check_same_layout(self, other)?;
        let mut out = self.clone();
        for (name, (_, v)) in out.tensors.iter_mut() {
            let o = &other.tensors[name].1;
            for (a, b) in v.iter_mut().zip(o) {
                *a -= *b;
            }
        }
        Ok(out)
    }
}

/// A delta as raw fp16 bits per tensor. Half the memory of `Params` while it waits for its round.
pub type Delta16 = BTreeMap<String, (Vec<usize>, Vec<u16>)>;

fn check_same_layout(a: &Params, b: &Params) -> Result<()> {
    if a.tensors.len() != b.tensors.len() {
        bail!("tensor count differs: {} vs {}", a.tensors.len(), b.tensors.len());
    }
    for (name, (shape, v)) in &a.tensors {
        match b.tensors.get(name) {
            Some((s2, v2)) if s2 == shape && v2.len() == v.len() => {}
            _ => bail!("tensor `{name}` differs in shape or is missing"),
        }
    }
    Ok(())
}

fn delta_matches(global: &Params, delta: &Delta16) -> bool {
    global.tensors.len() == delta.len()
        && global.tensors.iter().all(|(name, (shape, v))| {
            matches!(delta.get(name), Some((s2, bits)) if s2 == shape && bits.len() == v.len())
        })
}

/// Parse a safetensors payload. Accepts f32, f16 and bf16 and converts to f32.
pub fn parse_safetensors(bytes: &[u8]) -> Result<Params> {
    let st = SafeTensors::deserialize(bytes).context("not a safetensors payload")?;
    let mut tensors = BTreeMap::new();
    for (name, view) in st.tensors() {
        let raw = view.data();
        let values: Vec<f32> = match view.dtype() {
            Dtype::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Dtype::F16 => raw.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            Dtype::BF16 => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            other => bail!("tensor `{name}` has unsupported dtype {other:?}"),
        };
        if values.iter().any(|v| !v.is_finite()) {
            bail!("tensor `{name}` contains NaN or infinity");
        }
        tensors.insert(name.to_string(), (view.shape().to_vec(), values));
    }
    Ok(Params { tensors })
}

/// Serialise as f32 (exact) or f16 (half the bytes).
pub fn to_safetensors(params: &Params, dtype: Dtype) -> Result<Vec<u8>> {
    let mut stored: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, (shape, v)) in &params.tensors {
        let bytes: Vec<u8> = match dtype {
            Dtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Dtype::F16 => v.iter().flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes()).collect(),
            other => bail!("cannot write dtype {other:?}"),
        };
        stored.push((name.clone(), shape.clone(), bytes));
    }
    let views: Vec<(String, TensorView)> = stored
        .iter()
        .map(|(name, shape, bytes)| Ok((name.clone(), TensorView::new(dtype, shape.clone(), bytes)?)))
        .collect::<Result<_>>()?;
    Ok(safetensors::serialize(views, None)?)
}

/// int8 for matrices, one f16 scale per row (named `<tensor>.__scale`). f16 for the 1-D tensors.
/// The worker dequantises as `q * scale`. The scale is rounded to f16 before the values are
/// quantised, so both sides use the same number.
pub fn to_int8_safetensors(params: &Params) -> Result<Vec<u8>> {
    let mut stored: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, (shape, v)) in &params.tensors {
        if shape.len() == 2 && shape[1] > 0 {
            let (rows, cols) = (shape[0], shape[1]);
            let mut q: Vec<u8> = Vec::with_capacity(v.len());
            let mut scales: Vec<u8> = Vec::with_capacity(rows * 2);
            for row in v.chunks(cols) {
                let max = row.iter().fold(0f32, |m, x| m.max(x.abs()));
                let scale = if max > 1e-6 { f16::from_f32(max / 127.0) } else { f16::ONE };
                let s = scale.to_f32();
                scales.extend_from_slice(&scale.to_bits().to_le_bytes());
                q.extend(row.iter().map(|x| ((x / s).round().clamp(-127.0, 127.0) as i8) as u8));
            }
            stored.push((name.clone(), Dtype::I8, shape.clone(), q));
            stored.push((format!("{name}{SCALE_SUFFIX}"), Dtype::F16, vec![rows], scales));
        } else {
            let bytes: Vec<u8> = v.iter().flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes()).collect();
            stored.push((name.clone(), Dtype::F16, shape.clone(), bytes));
        }
    }
    let views: Vec<(String, TensorView)> = stored
        .iter()
        .map(|(name, dtype, shape, bytes)| Ok((name.clone(), TensorView::new(*dtype, shape.clone(), bytes)?)))
        .collect::<Result<_>>()?;
    Ok(safetensors::serialize(views, None)?)
}

/// Inverse of `to_int8_safetensors`, for tests and tools. Refuses non-finite values.
pub fn parse_int8_safetensors(bytes: &[u8]) -> Result<Params> {
    let st = SafeTensors::deserialize(bytes).context("not a safetensors payload")?;
    let mut tensors = BTreeMap::new();
    for (name, view) in st.tensors() {
        if name.ends_with(SCALE_SUFFIX) {
            continue;
        }
        let shape = view.shape().to_vec();
        let values: Vec<f32> = match view.dtype() {
            Dtype::I8 => {
                let key = format!("{name}{SCALE_SUFFIX}");
                let scales_view = st.tensor(&key).map_err(|e| anyhow::anyhow!("tensor `{name}` has no scales: {e}"))?;
                let scales: Vec<f32> = scales_view
                    .data()
                    .chunks_exact(2)
                    .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect();
                let cols = shape.get(1).copied().unwrap_or(0);
                if cols == 0 || scales.len() != shape[0] {
                    bail!("tensor `{name}` has a scale layout that does not match its shape");
                }
                view.data().iter().enumerate().map(|(i, b)| (*b as i8) as f32 * scales[i / cols]).collect()
            }
            Dtype::F16 => view.data().chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            Dtype::F32 => view.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            other => bail!("tensor `{name}` has unsupported dtype {other:?}"),
        };
        if values.iter().any(|v| !v.is_finite()) {
            bail!("tensor `{name}` contains NaN or infinity");
        }
        tensors.insert(name.to_string(), (shape, values));
    }
    Ok(Params { tensors })
}

/// Parse an fp16 delta as raw bits. Other dtypes are refused, so pending deltas stay half-size.
pub fn parse_delta16(bytes: &[u8]) -> Result<Delta16> {
    let st = SafeTensors::deserialize(bytes).context("not a safetensors payload")?;
    let mut out = BTreeMap::new();
    for (name, view) in st.tensors() {
        if view.dtype() != Dtype::F16 {
            bail!("tensor `{name}` must be fp16 (got {:?})", view.dtype());
        }
        let bits: Vec<u16> = view.data().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        if bits.iter().any(|b| !f16::from_bits(*b).is_finite()) {
            bail!("tensor `{name}` contains NaN or infinity");
        }
        out.insert(name.to_string(), (view.shape().to_vec(), bits));
    }
    Ok(out)
}

/// sha256 over the float32 little-endian values of every tensor, in name order. The Python side
/// (training/initrng.canonical_sha256) computes the same digest, so a worker can check its starting weights.
pub fn canonical_sha256(params: &Params) -> String {
    let mut hasher = Sha256::new();
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 20);
    for (_, (_, values)) in &params.tensors {
        for chunk in values.chunks(1 << 18) {
            buf.clear();
            for x in chunk {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            hasher.update(&buf);
        }
    }
    hex::encode(hasher.finalize())
}

/// Element-wise mean of several f32 deltas.
pub fn average(deltas: &[Params]) -> Result<Params> {
    let first = deltas.first().context("no deltas to average")?;
    let mut out = first.clone();
    for d in &deltas[1..] {
        check_same_layout(&out, d)?;
        for (name, (_, v)) in out.tensors.iter_mut() {
            let o = &d.tensors[name].1;
            for (a, b) in v.iter_mut().zip(o) {
                *a += *b;
            }
        }
    }
    let n = deltas.len() as f32;
    for (_, v) in out.tensors.values_mut() {
        for a in v.iter_mut() {
            *a /= n;
        }
    }
    Ok(out)
}

/// Mean of fp16 deltas, accumulated in f32 one tensor at a time.
fn average16(template: &Params, deltas: &[&Delta16]) -> Result<Params> {
    anyhow::ensure!(!deltas.is_empty(), "no deltas to average");
    let mut out = zeros_like(template);
    for delta in deltas {
        for (name, (_, acc)) in out.tensors.iter_mut() {
            let (_, bits) = delta.get(name).context("delta is missing a tensor")?;
            for (a, b) in acc.iter_mut().zip(bits) {
                *a += f16::from_bits(*b).to_f32();
            }
        }
    }
    let n = deltas.len() as f32;
    for (_, v) in out.tensors.values_mut() {
        for a in v.iter_mut() {
            *a /= n;
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OuterOpt {
    pub lr: f32,
    pub momentum: f32,
}

impl Default for OuterOpt {
    fn default() -> Self {
        // Momentum 0 on purpose. The DiLoCo value (0.9) overshot in our run: see docs/KAGGLE.md, "Outer momentum".
        OuterOpt { lr: 0.7, momentum: 0.0 }
    }
}

/// One Nesterov outer step. `momentum` must have the same layout as `params`.
pub fn outer_step(params: &mut Params, momentum: &mut Params, avg_delta: &Params, opt: OuterOpt) -> Result<()> {
    check_same_layout(params, avg_delta)?;
    check_same_layout(params, momentum)?;
    for (name, (_, theta)) in params.tensors.iter_mut() {
        let d = &avg_delta.tensors[name].1;
        let m = &mut momentum.tensors.get_mut(name).context("momentum tensor missing")?.1;
        for i in 0..theta.len() {
            m[i] = opt.momentum * m[i] + d[i];
            theta[i] -= opt.lr * (d[i] + opt.momentum * m[i]);
        }
    }
    Ok(())
}

pub fn zeros_like(params: &Params) -> Params {
    Params {
        tensors: params.tensors.iter().map(|(k, (s, v))| (k.clone(), (s.clone(), vec![0.0; v.len()]))).collect(),
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub workers: usize,
    pub quorum: usize,
    pub timeout: Duration,
    pub outer: OuterOpt,
    pub token: Option<String>,
    pub state_dir: PathBuf,
    pub serve_dir: Option<PathBuf>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Status {
    pub round: u64,
    pub workers: usize,
    pub quorum: usize,
    pub received_this_round: usize,
    pub params: usize,
    /// sha256 over the float32 values in name order (see `canonical_sha256`).
    pub global_sha256: String,
    /// Unix time each worker last posted a delta or an event.
    #[serde(default)]
    pub last_seen: BTreeMap<String, u64>,
}

/// A delta being uploaded in parts. A part can be lost to a dropped connection, so each part is
/// stored by index and a repeated part just overwrites the same slot.
struct PartialDelta {
    seq: u64,
    round: u64,
    parts: Vec<Option<Vec<u8>>>,
}

struct RoundState {
    round: u64,
    global: Params,
    momentum: Params,
    /// Each delta with the global version it was computed from.
    pending: BTreeMap<String, (Delta16, u64)>,
    /// Last accepted (sequence number, version it joined) per worker. Makes retries idempotent.
    seen: BTreeMap<String, (u64, u64)>,
    last_seen: BTreeMap<String, u64>,
    round_started: Instant,
    partial: BTreeMap<String, PartialDelta>,
    global_f16: Arc<Vec<u8>>,
    global_i8: Arc<Vec<u8>>,
    sha: String,
}

struct Shared {
    cfg: Config,
    state: Mutex<RoundState>,
}

impl RoundState {
    fn refresh_bytes(&mut self) -> Result<()> {
        self.global_f16 = Arc::new(to_safetensors(&self.global, Dtype::F16)?);
        self.global_i8 = Arc::new(to_int8_safetensors(&self.global)?);
        self.sha = canonical_sha256(&self.global);
        Ok(())
    }
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn pending_dir(state_dir: &Path) -> PathBuf {
    state_dir.join("pending")
}

/// Keep an accepted delta on disk until its round closes, so a coordinator restart does not lose it.
fn save_pending(state_dir: &Path, worker: &str, body: &[u8], seq: u64, base: u64, accepted_into: u64) -> Result<()> {
    let dir = pending_dir(state_dir);
    std::fs::create_dir_all(&dir)?;
    write_atomic(&dir.join(format!("{worker}.delta")), body)?;
    let meta = serde_json::json!({"seq": seq, "base": base, "accepted_into": accepted_into});
    write_atomic(&dir.join(format!("{worker}.json")), meta.to_string().as_bytes())
}

/// Reload the deltas that were waiting when the coordinator last stopped. Only deltas accepted into the
/// current round are restored: an older one was already averaged into the global model.
fn load_pending(state_dir: &Path, state: &mut RoundState) -> usize {
    let dir = pending_dir(state_dir);
    let Ok(entries) = std::fs::read_dir(&dir) else { return 0 };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(worker) = path.file_stem().and_then(|n| n.to_str()).map(String::from) else { continue };
        let Some(meta) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { continue };
        if meta["accepted_into"].as_u64() != Some(state.round) {
            continue;
        }
        let Ok(bytes) = std::fs::read(dir.join(format!("{worker}.delta"))) else { continue };
        let Ok(delta) = parse_delta16(&bytes) else { continue };
        if !delta_matches(&state.global, &delta) {
            continue;
        }
        let seq = meta["seq"].as_u64().unwrap_or(0);
        let base = meta["base"].as_u64().unwrap_or(state.round);
        state.seen.insert(worker.clone(), (seq, state.round));
        state.pending.insert(worker, (delta, base));
    }
    state.pending.len()
}

/// Write the latest global and momentum, then the status. Only the newest copy is kept.
fn persist(dir: &Path, state: &RoundState, cfg: &Config) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    write_atomic(&dir.join("global.safetensors"), &to_safetensors(&state.global, Dtype::F32)?)?;
    write_atomic(&dir.join("momentum.safetensors"), &to_safetensors(&state.momentum, Dtype::F32)?)?;
    write_atomic(&dir.join("status.json"), serde_json::to_string_pretty(&status_of(state, cfg))?.as_bytes())?;
    Ok(())
}

/// Start from a saved global model when one exists, otherwise from `initial`.
pub fn open_state(cfg: &Config, initial: Params) -> Result<(u64, Params, Params)> {
    let path = cfg.state_dir.join("global.safetensors");
    let status_path = cfg.state_dir.join("status.json");
    if path.exists() && status_path.exists() {
        let global = parse_safetensors(&std::fs::read(&path)?)?;
        let status: Status = serde_json::from_str(&std::fs::read_to_string(&status_path)?)?;
        check_same_layout(&initial, &global).context("saved state does not match this model")?;
        let momentum_path = cfg.state_dir.join("momentum.safetensors");
        let momentum = if momentum_path.exists() {
            parse_safetensors(&std::fs::read(momentum_path)?)?
        } else {
            zeros_like(&global)
        };
        return Ok((status.round, global, momentum));
    }
    let momentum = zeros_like(&initial);
    Ok((0, initial, momentum))
}

fn status_of(state: &RoundState, cfg: &Config) -> Status {
    Status {
        round: state.round,
        workers: cfg.workers,
        quorum: cfg.quorum,
        received_this_round: state.pending.len(),
        params: state.global.len(),
        global_sha256: state.sha.clone(),
        last_seen: state.last_seen.clone(),
    }
}

/// Close the round: average, outer step, persist, advance.
fn aggregate(shared: &Shared, state: &mut RoundState) -> Result<()> {
    let stale = state.pending.values().filter(|(_, base)| *base < state.round).count();
    let deltas: Vec<&Delta16> = state.pending.values().map(|(d, _)| d).collect();
    let count = deltas.len();
    let avg = average16(&state.global, &deltas)?;
    outer_step(&mut state.global, &mut state.momentum, &avg, shared.cfg.outer)?;
    drop(avg);
    state.pending.clear();
    state.round += 1;
    state.round_started = Instant::now();
    state.refresh_bytes()?;
    println!(
        "PW {}",
        serde_json::json!({"ev": "outer", "version": state.round, "deltas": count, "stale": stale, "global_sha": state.sha})
    );
    persist(&shared.cfg.state_dir, state, &shared.cfg)?;
    let _ = std::fs::remove_dir_all(pending_dir(&shared.cfg.state_dir)); // their round is closed
    Ok(())
}

fn authorised(cfg: &Config, headers: &HeaderMap) -> bool {
    let Some(token) = &cfg.token else { return true };
    let Some(given) = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) else { return false };
    constant_time_eq(given.as_bytes(), format!("Bearer {token}").as_bytes())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn valid_worker_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

async fn status_handler(State(sh): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if !authorised(&sh.cfg, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let mut st = sh.state.lock().await;
    // A round that has waited long enough closes with the deltas it has.
    if !st.pending.is_empty() && st.round_started.elapsed() >= sh.cfg.timeout {
        if let Err(e) = aggregate(&sh, &mut st) {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    }
    axum::Json(status_of(&st, &sh.cfg)).into_response()
}

#[derive(Deserialize)]
struct GlobalQuery {
    /// "f32" (default, exact), "f16", or "int8" (one quarter of the f32 bytes; workers request this).
    dtype: Option<String>,
}

/// Parse a single `bytes=a-b` (or `bytes=a-`) range against a body of `len` bytes.
fn parse_range(value: &str, len: usize) -> Option<(usize, usize)> {
    let spec = value.strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (a, b) = spec.split_once('-')?;
    let start: usize = a.trim().parse().ok()?;
    if start >= len {
        return None;
    }
    // Like HTTP itself: a range that runs past the end is clipped to the end of the body.
    let end: usize = if b.trim().is_empty() { len - 1 } else { b.trim().parse::<usize>().ok()?.min(len - 1) };
    if start > end {
        return None;
    }
    Some((start, end))
}

async fn global_handler(State(sh): State<Arc<Shared>>, headers: HeaderMap, Query(q): Query<GlobalQuery>) -> Response {
    if !authorised(&sh.cfg, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let st = sh.state.lock().await;
    let body: Arc<Vec<u8>> = match q.dtype.as_deref().unwrap_or("f32") {
        "int8" => st.global_i8.clone(),
        "f16" => st.global_f16.clone(),
        "f32" => match to_safetensors(&st.global, Dtype::F32) {
            Ok(bytes) => Arc::new(bytes),
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        },
        other => return (StatusCode::BAD_REQUEST, format!("unknown dtype `{other}`; use int8, f16 or f32")).into_response(),
    };
    let common = [
        (header::CONTENT_TYPE, "application/octet-stream".to_string()),
        (header::ETAG, st.sha.clone()),
        (header::ACCEPT_RANGES, "bytes".to_string()),
    ];
    let round = st.round.to_string();
    let sha = st.sha.clone();
    drop(st);
    match headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        None => (
            common,
            [("x-pw-round", round), ("x-pw-sha256", sha)],
            body.as_ref().clone(),
        )
            .into_response(),
        Some(range) => match parse_range(range, body.len()) {
            Some((a, b)) => (
                StatusCode::PARTIAL_CONTENT,
                common,
                [
                    ("x-pw-round", round),
                    ("x-pw-sha256", sha),
                    (header::CONTENT_RANGE.as_str(), format!("bytes {a}-{b}/{}", body.len())),
                ],
                body[a..=b].to_vec(),
            )
                .into_response(),
            None => (StatusCode::RANGE_NOT_SATISFIABLE, [(header::CONTENT_RANGE.as_str(), format!("bytes */{}", body.len()))]).into_response(),
        },
    }
}

#[derive(Deserialize)]
struct DeltaQuery {
    worker: String,
    round: u64,
    /// Per-worker sequence number, increasing. A repeat is acknowledged, never counted twice.
    #[serde(default)]
    seq: u64,
}

/// Accept a delta that has fully arrived. Shared by the single-request and the chunked upload.
fn accept_delta(sh: &Shared, st: &mut RoundState, worker: String, seq: u64, round: u64, body: &[u8]) -> Response {
    st.last_seen.insert(worker.clone(), unix_now());
    let delta = match parse_delta16(body) {
        Ok(d) => d,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("bad delta: {e:#}")).into_response(),
    };
    if let Some(&(last, joined)) = st.seen.get(&worker) {
        if seq <= last {
            let mut body = serde_json::to_value(status_of(st, &sh.cfg)).unwrap_or(serde_json::Value::Null);
            body["accepted_into"] = serde_json::json!(joined);
            body["duplicate"] = serde_json::json!(true);
            return (StatusCode::ACCEPTED, axum::Json(body)).into_response();
        }
    }
    if round > st.round {
        return (StatusCode::CONFLICT, format!("delta is based on version {}, but the newest is {}", round, st.round)).into_response();
    }
    if round + MAX_STALENESS < st.round {
        return (
            StatusCode::CONFLICT,
            format!("delta is based on version {}, more than {MAX_STALENESS} behind version {}; fetch the global and resync", round, st.round),
        )
            .into_response();
    }
    if !delta_matches(&st.global, &delta) {
        return (StatusCode::BAD_REQUEST, "delta layout does not match the global model").into_response();
    }
    // The version this delta lands in. If its arrival closes the round, that round is the one it joined.
    let accepted_into = st.round;
    if let Err(e) = save_pending(&sh.cfg.state_dir, &worker, body, seq, round, accepted_into) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("could not keep the delta: {e}")).into_response();
    }
    st.seen.insert(worker.clone(), (seq, accepted_into));
    st.pending.insert(worker, (delta, round));
    if st.pending.len() >= sh.cfg.quorum {
        if let Err(e) = aggregate(sh, st) {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    }
    let mut body = serde_json::to_value(status_of(st, &sh.cfg)).unwrap_or(serde_json::Value::Null);
    body["accepted_into"] = serde_json::json!(accepted_into);
    (StatusCode::ACCEPTED, axum::Json(body)).into_response()
}

async fn delta_handler(State(sh): State<Arc<Shared>>, headers: HeaderMap, Query(q): Query<DeltaQuery>, body: Bytes) -> Response {
    if !authorised(&sh.cfg, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if !valid_worker_name(&q.worker) {
        return (StatusCode::BAD_REQUEST, "worker id must be 1-64 letters, digits, - or _").into_response();
    }
    let mut st = sh.state.lock().await;
    accept_delta(&sh, &mut st, q.worker, q.seq, q.round, &body)
}

#[derive(Deserialize)]
struct PartQuery {
    worker: String,
    round: u64,
    seq: u64,
    part: usize,
    parts: usize,
}

/// One chunk of a delta. Each request is small, so a dropped connection costs one chunk, not the whole upload.
async fn delta_part_handler(State(sh): State<Arc<Shared>>, headers: HeaderMap, Query(q): Query<PartQuery>, body: Bytes) -> Response {
    if !authorised(&sh.cfg, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if !valid_worker_name(&q.worker) {
        return (StatusCode::BAD_REQUEST, "worker id must be 1-64 letters, digits, - or _").into_response();
    }
    if q.parts == 0 || q.parts > 4096 || q.part >= q.parts {
        return (StatusCode::BAD_REQUEST, "part must be below parts, and parts between 1 and 4096").into_response();
    }
    let mut st = sh.state.lock().await;
    if let Some(&(last, joined)) = st.seen.get(&q.worker) {
        if q.seq <= last {
            let mut body = serde_json::to_value(status_of(&st, &sh.cfg)).unwrap_or(serde_json::Value::Null);
            body["accepted_into"] = serde_json::json!(joined);
            body["duplicate"] = serde_json::json!(true);
            return (StatusCode::ACCEPTED, axum::Json(body)).into_response();
        }
    }
    let entry = st.partial.entry(q.worker.clone()).or_insert_with(|| PartialDelta {
        seq: q.seq,
        round: q.round,
        parts: vec![None; q.parts],
    });
    if entry.seq != q.seq || entry.round != q.round || entry.parts.len() != q.parts {
        *entry = PartialDelta { seq: q.seq, round: q.round, parts: vec![None; q.parts] };
    }
    entry.parts[q.part] = Some(body.to_vec());
    if entry.parts.iter().all(|p| p.is_some()) {
        let done = st.partial.remove(&q.worker).expect("entry was just used");
        let whole: Vec<u8> = done.parts.into_iter().flatten().flatten().collect();
        return accept_delta(&sh, &mut st, q.worker, q.seq, q.round, &whole);
    }
    let received = st.partial.get(&q.worker).map(|p| p.parts.iter().filter(|x| x.is_some()).count()).unwrap_or(0);
    (StatusCode::ACCEPTED, axum::Json(serde_json::json!({"complete": false, "part": q.part, "received": received, "parts": q.parts}))).into_response()
}

async fn events_handler(State(sh): State<Arc<Shared>>, headers: HeaderMap, body: Bytes) -> Response {
    if !authorised(&sh.cfg, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if body.len() > MAX_EVENT_BODY_BYTES {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }
    let events: Vec<serde_json::Value> = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("events must be a JSON array: {e}")).into_response(),
    };
    if events.len() > MAX_EVENTS_PER_POST {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }
    let now = unix_now();
    let mut lines = String::new();
    {
        let mut st = sh.state.lock().await;
        for event in &events {
            if let Some(group) = event.get("group").and_then(|g| g.as_str()) {
                if valid_worker_name(group) {
                    st.last_seen.insert(group.to_string(), now);
                }
            }
        }
    }
    for mut event in events {
        if let Some(obj) = event.as_object_mut() {
            obj.insert("received".to_string(), serde_json::json!(now));
        }
        let line = event.to_string();
        println!("PW {line}");
        lines.push_str(&line);
        lines.push('\n');
    }
    let accepted = lines.lines().count();
    if let Err(e) = std::fs::create_dir_all(&sh.cfg.state_dir).and_then(|_| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(sh.cfg.state_dir.join("events.jsonl"))
            .and_then(|mut f| f.write_all(lines.as_bytes()))
    }) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("could not write events: {e}")).into_response();
    }
    (StatusCode::ACCEPTED, axum::Json(serde_json::json!({"accepted": accepted}))).into_response()
}

async fn file_handler(State(sh): State<Arc<Shared>>, headers: HeaderMap, axum::extract::Path(name): axum::extract::Path<String>) -> Response {
    if !authorised(&sh.cfg, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Some(dir) = &sh.cfg.serve_dir else { return StatusCode::NOT_FOUND.into_response() };
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return StatusCode::BAD_REQUEST.into_response();
    }
    match tokio::fs::read(dir.join(&name)).await {
        Ok(bytes) => ([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Build the HTTP app. `initial` is used when no saved state exists.
pub fn router(cfg: Config, initial: Params) -> Result<Router> {
    anyhow::ensure!(cfg.quorum >= 1 && cfg.quorum <= cfg.workers, "quorum must be between 1 and the worker count");
    std::fs::create_dir_all(&cfg.state_dir)?;
    let (round, global, momentum) = open_state(&cfg, initial)?;
    let mut state = RoundState {
        round,
        global,
        momentum,
        pending: BTreeMap::new(),
        seen: BTreeMap::new(),
        last_seen: BTreeMap::new(),
        round_started: Instant::now(),
        partial: BTreeMap::new(),
        global_f16: Arc::new(Vec::new()),
        global_i8: Arc::new(Vec::new()),
        sha: String::new(),
    };
    state.refresh_bytes()?;
    let restored = load_pending(&cfg.state_dir, &mut state);
    if restored > 0 {
        println!("PW {}", serde_json::json!({"ev": "pending_restored", "count": restored, "round": state.round}));
    }
    let shared = Arc::new(Shared { cfg, state: Mutex::new(state) });
    Ok(Router::new()
        .route("/v1/status", get(status_handler))
        .route("/v1/global", get(global_handler))
        .route("/v1/delta", post(delta_handler))
        .route("/v1/delta-part", post(delta_part_handler))
        .route("/v1/events", post(events_handler))
        .route("/v1/files/{name}", get(file_handler))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(shared))
}

pub async fn serve(cfg: Config, initial: Params, addr: std::net::SocketAddr) -> Result<()> {
    let app = router(cfg, initial)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn params_from(tensors: Vec<(&str, Vec<usize>, Vec<f32>)>) -> Params {
    Params { tensors: tensors.into_iter().map(|(n, s, v)| (n.to_string(), (s, v))).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(values: &[f32]) -> Params {
        params_from(vec![("w", vec![values.len()], values.to_vec())])
    }

    #[test]
    fn plain_averaging_is_outer_step_with_lr_one_and_no_momentum() {
        let mut theta = p(&[1.0, 2.0]);
        let mut m = zeros_like(&theta);
        // Two workers moved the weights to [0.8, 2.2] and [0.6, 1.8]: deltas are the moves.
        let d1 = p(&[0.2, -0.2]);
        let d2 = p(&[0.4, 0.2]);
        let avg = average(&[d1, d2]).unwrap();
        outer_step(&mut theta, &mut m, &avg, OuterOpt { lr: 1.0, momentum: 0.0 }).unwrap();
        assert!((theta.tensors["w"].1[0] - 0.7).abs() < 1e-6);
        assert!((theta.tensors["w"].1[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn nesterov_momentum_accumulates_across_rounds() {
        let mut theta = p(&[0.0]);
        let mut m = zeros_like(&theta);
        let delta = p(&[1.0]);
        outer_step(&mut theta, &mut m, &delta, OuterOpt { lr: 0.5, momentum: 0.9 }).unwrap();
        // m = 1, step = 0.5 * (1 + 0.9 * 1) = 0.95
        assert!((theta.tensors["w"].1[0] + 0.95).abs() < 1e-6);
        outer_step(&mut theta, &mut m, &delta, OuterOpt { lr: 0.5, momentum: 0.9 }).unwrap();
        // m = 0.9 + 1 = 1.9, step = 0.5 * (1 + 0.9 * 1.9) = 1.355
        assert!((theta.tensors["w"].1[0] + 0.95 + 1.355).abs() < 1e-5);
    }

    #[test]
    fn safetensors_round_trip_in_f32_and_f16() {
        let original = p(&[0.5, -1.25, 3.0]);
        let back = parse_safetensors(&to_safetensors(&original, Dtype::F32).unwrap()).unwrap();
        assert_eq!(back, original);
        let half = parse_safetensors(&to_safetensors(&original, Dtype::F16).unwrap()).unwrap();
        for (a, b) in half.tensors["w"].1.iter().zip(&original.tensors["w"].1) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn int8_global_is_within_half_a_step_of_each_row_scale() {
        let rows = 4;
        let cols = 64;
        let values: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0).collect();
        let original = params_from(vec![("m", vec![rows, cols], values.clone()), ("n", vec![cols], vec![1.5; cols])]);
        let back = parse_int8_safetensors(&to_int8_safetensors(&original).unwrap()).unwrap();
        assert_eq!(back.tensors["m"].0, vec![rows, cols]);
        for r in 0..rows {
            let row = &values[r * cols..(r + 1) * cols];
            let max = row.iter().fold(0f32, |m, x| m.max(x.abs()));
            let step = max / 127.0;
            for c in 0..cols {
                let err = (back.tensors["m"].1[r * cols + c] - row[c]).abs();
                assert!(err <= step * 0.51 + 1e-6, "row {r} col {c}: error {err} above half a step {step}");
            }
        }
        for (a, b) in back.tensors["n"].1.iter().zip(&original.tensors["n"].1) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn int8_payload_is_about_a_quarter_of_f32() {
        let values: Vec<f32> = (0..128 * 256).map(|i| (i % 7) as f32 * 0.01).collect();
        let params = params_from(vec![("m", vec![128, 256], values)]);
        let f32_len = to_safetensors(&params, Dtype::F32).unwrap().len();
        let i8_len = to_int8_safetensors(&params).unwrap().len();
        assert!(i8_len * 3 < f32_len, "int8 {i8_len} bytes vs f32 {f32_len}");
    }

    #[test]
    fn canonical_hash_matches_the_python_reference() {
        // Computed with training/initrng.canonical_sha256 on {"a": [1.0, 2.0], "b": [3.0]}.
        let params = params_from(vec![("b", vec![1], vec![3.0]), ("a", vec![2], vec![1.0, 2.0])]);
        assert_eq!(canonical_sha256(&params), "8e628779e6a74ee0b36991c10158f63cafec7d340ad4e075592502c8708524dd");
    }

    #[test]
    fn deltas_must_be_fp16_and_finite() {
        let f32_delta = to_safetensors(&p(&[0.1]), Dtype::F32).unwrap();
        assert!(parse_delta16(&f32_delta).is_err());
        let f16_delta = to_safetensors(&p(&[0.1]), Dtype::F16).unwrap();
        assert!(parse_delta16(&f16_delta).is_ok());
        let nan_delta = to_safetensors(&p(&[f32::NAN]), Dtype::F16).unwrap();
        assert!(parse_delta16(&nan_delta).is_err());
    }

    #[test]
    fn non_finite_and_mismatched_payloads_are_refused() {
        let bad = p(&[f32::NAN]);
        assert!(parse_safetensors(&to_safetensors(&bad, Dtype::F32).unwrap()).is_err());
        assert!(parse_safetensors(b"not a tensor file").is_err());
        let a = p(&[1.0, 2.0]);
        let b = p(&[1.0]);
        assert!(average(&[a.clone(), b]).is_err());
        assert!(a.minus(&params_from(vec![("other", vec![2], vec![0.0, 0.0])])).is_err());
    }

    #[test]
    fn constant_time_compare_is_exact() {
        assert!(constant_time_eq(b"Bearer abc", b"Bearer abc"));
        assert!(!constant_time_eq(b"Bearer abc", b"Bearer abd"));
        assert!(!constant_time_eq(b"Bearer ab", b"Bearer abc"));
    }
}
