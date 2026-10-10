//! Distributed training coordinator (DiLoCo-style).
//!
//! Workers train locally for H steps, then send their pseudo-gradient
//! `delta = global - local` (fp16 safetensors). When a quorum of deltas arrives, the
//! coordinator averages them and applies an outer Nesterov-momentum step:
//!
//!   m     <- mu * m + avg
//!   theta <- theta - lr * (avg + mu * m)
//!
//! With lr = 1 and mu = 0 this is plain model averaging. The defaults (lr 0.7, mu 0.9) follow
//! the DiLoCo paper. Workers can be slow or disappear, so a round also closes on a timeout
//! once at least one delta has arrived. Deltas for an old round are rejected (409) and the
//! worker downloads the current global model instead.
//!
//! RoundState is written to `state_dir` after every round, so the coordinator can restart.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

/// Serialise as f32 (global model) or f16 (deltas, half the bytes on the wire).
pub fn to_safetensors(params: &Params, dtype: Dtype) -> Result<Vec<u8>> {
    let mut bytes_store: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, (shape, v)) in &params.tensors {
        let bytes: Vec<u8> = match dtype {
            Dtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Dtype::F16 => v.iter().flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes()).collect(),
            other => bail!("cannot write dtype {other:?}"),
        };
        bytes_store.push((name.clone(), shape.clone(), bytes));
    }
    let views: Vec<(String, TensorView)> = bytes_store
        .iter()
        .map(|(name, shape, bytes)| Ok((name.clone(), TensorView::new(dtype, shape.clone(), bytes)?)))
        .collect::<Result<_>>()?;
    Ok(safetensors::serialize(views, None)?)
}

/// Element-wise mean of several deltas.
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OuterOpt {
    pub lr: f32,
    pub momentum: f32,
}

impl Default for OuterOpt {
    fn default() -> Self {
        OuterOpt { lr: 0.7, momentum: 0.9 }
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
    pub global_sha256: String,
}

struct RoundState {
    round: u64,
    global: Params,
    momentum: Params,
    pending: BTreeMap<String, Params>,
    round_started: Instant,
    global_bytes: Vec<u8>,
    sha: String,
}

struct Shared {
    cfg: Config,
    state: Mutex<RoundState>,
}

fn sha_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

impl RoundState {
    fn refresh_bytes(&mut self) -> Result<()> {
        self.global_bytes = to_safetensors(&self.global, Dtype::F32)?;
        self.sha = sha_hex(&self.global_bytes);
        Ok(())
    }
}

fn persist(dir: &Path, round: u64, bytes: &[u8], status: &Status) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("global.safetensors"), bytes)?;
    std::fs::write(dir.join(format!("global-round-{round:06}.safetensors")), bytes)?;
    std::fs::write(dir.join("status.json"), serde_json::to_string_pretty(status)?)?;
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
    }
}

/// Close the round: average, outer step, persist, advance.
fn aggregate(shared: &Shared, state: &mut RoundState) -> Result<()> {
    let deltas: Vec<Params> = state.pending.values().cloned().collect();
    let avg = average(&deltas)?;
    outer_step(&mut state.global, &mut state.momentum, &avg, shared.cfg.outer)?;
    state.pending.clear();
    state.round += 1;
    state.round_started = Instant::now();
    state.refresh_bytes()?;
    let status = status_of(state, &shared.cfg);
    persist(&shared.cfg.state_dir, state.round, &state.global_bytes, &status)?;
    std::fs::write(shared.cfg.state_dir.join("momentum.safetensors"), to_safetensors(&state.momentum, Dtype::F32)?)?;
    Ok(())
}

fn authorised(cfg: &Config, headers: &HeaderMap) -> bool {
    match &cfg.token {
        None => true,
        Some(t) => headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v == format!("Bearer {t}")),
    }
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

async fn global_handler(State(sh): State<Arc<Shared>>, headers: HeaderMap) -> Response {
    if !authorised(&sh.cfg, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let st = sh.state.lock().await;
    (
        [(header::CONTENT_TYPE, "application/octet-stream".to_string()), (header::ETAG, st.sha.clone())],
        [("x-pw-round", st.round.to_string())],
        st.global_bytes.clone(),
    )
        .into_response()
}

#[derive(Deserialize)]
struct DeltaQuery {
    worker: String,
    round: u64,
}

async fn delta_handler(State(sh): State<Arc<Shared>>, headers: HeaderMap, Query(q): Query<DeltaQuery>, body: Bytes) -> Response {
    if !authorised(&sh.cfg, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if q.worker.is_empty() || q.worker.len() > 64 || !q.worker.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return (StatusCode::BAD_REQUEST, "worker id must be 1-64 letters, digits, - or _").into_response();
    }
    let delta = match parse_safetensors(&body) {
        Ok(d) => d,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("bad delta: {e:#}")).into_response(),
    };
    let mut st = sh.state.lock().await;
    if q.round != st.round {
        return (StatusCode::CONFLICT, format!("round is {}, delta was for {}", st.round, q.round)).into_response();
    }
    if check_same_layout(&st.global, &delta).is_err() {
        return (StatusCode::BAD_REQUEST, "delta layout does not match the global model").into_response();
    }
    st.pending.insert(q.worker, delta);
    if st.pending.len() >= sh.cfg.quorum {
        if let Err(e) = aggregate(&sh, &mut st) {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    }
    (StatusCode::ACCEPTED, axum::Json(status_of(&st, &sh.cfg))).into_response()
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
        round_started: Instant::now(),
        global_bytes: Vec::new(),
        sha: String::new(),
    };
    state.refresh_bytes()?;
    let shared = Arc::new(Shared { cfg, state: Mutex::new(state) });
    Ok(Router::new()
        .route("/v1/status", get(status_handler))
        .route("/v1/global", get(global_handler))
        .route("/v1/delta", post(delta_handler))
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
    fn non_finite_and_mismatched_payloads_are_refused() {
        let bad = p(&[f32::NAN]);
        assert!(parse_safetensors(&to_safetensors(&bad, Dtype::F32).unwrap()).is_err());
        assert!(parse_safetensors(b"not a tensor file").is_err());
        let a = p(&[1.0, 2.0]);
        let b = p(&[1.0]);
        assert!(average(&[a.clone(), b]).is_err());
        assert!(a.minus(&params_from(vec![("other", vec![2], vec![0.0, 0.0])])).is_err());
    }
}
