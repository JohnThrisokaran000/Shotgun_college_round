//! The PatchWork reverse proxy. It serves the upstream site with the channel's active
//! release applied, takes health beacons from visitors' browsers, and rolls a channel
//! back when error beacons cross the policy threshold.
//!
//! Routes under `/__pw/` belong to PatchWork. Everything else goes to the upstream.

pub mod html;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::{Query, Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use pw_core::audit::AuditLog;
use pw_core::release::{self, BeaconPolicy, BeaconWindow};
use serde::Deserialize;
use serde_json::json;

const MAX_REQUEST_BODY: usize = 8 * 1024 * 1024;
const MAX_BEACON_BODY: usize = 4096;
const HOP_BY_HOP: &[&str] = &[
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "host",
];

#[derive(Clone)]
pub struct Config {
    /// Origin of the live site, for example `http://127.0.0.1:8080`.
    pub upstream: String,
    /// The PatchWork project root (holds `releases/`, `channels/`, `audit/`).
    pub root: PathBuf,
    pub policy: BeaconPolicy,
}

struct AppState {
    upstream: String,
    root: PathBuf,
    client: reqwest::Client,
    beacons: Mutex<BeaconWindow>,
}

pub fn router(cfg: Config) -> anyhow::Result<Router> {
    let upstream = cfg.upstream.trim_end_matches('/').to_string();
    anyhow::ensure!(
        upstream.starts_with("http://") || upstream.starts_with("https://"),
        "upstream must start with http:// or https://"
    );
    let client = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build()?;
    let state = Arc::new(AppState {
        upstream,
        root: cfg.root,
        client,
        beacons: Mutex::new(BeaconWindow::new(cfg.policy)),
    });
    Ok(Router::new()
        .route("/__pw/runtime.js", get(runtime_js))
        .route("/__pw/editor.js", get(editor_js))
        .route("/__pw/manifest", get(manifest))
        .route("/__pw/beacon", post(beacon))
        .route("/__pw/channel", get(set_channel))
        .fallback(proxy)
        .with_state(state))
}

pub async fn serve(cfg: Config, addr: SocketAddr) -> anyhow::Result<()> {
    let app = router(cfg)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn js(body: &'static str) -> Response {
    ([(header::CONTENT_TYPE, "application/javascript; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

async fn runtime_js() -> Response {
    js(html::RUNTIME_JS)
}

async fn editor_js() -> Response {
    js(html::EDITOR_JS)
}

fn cookie_header(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::COOKIE).and_then(|v| v.to_str().ok())
}

#[derive(Deserialize)]
struct ManifestQuery {
    path: Option<String>,
}

/// The active release's patches for this page, for the channel named in the visitor's cookie.
async fn manifest(State(st): State<Arc<AppState>>, headers: HeaderMap, Query(q): Query<ManifestQuery>) -> Response {
    let channel = html::channel_from_cookie(cookie_header(&headers));
    let json_reply = |value: serde_json::Value| {
        ([(header::CACHE_CONTROL, "no-store"), (header::CONTENT_TYPE, "application/json")], value.to_string()).into_response()
    };
    let current = match release::current(&st.root, &channel) {
        Ok(c) => c,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let Some(id) = current else {
        return json_reply(json!({"release": null, "channel": channel, "patches": []}));
    };
    match release::read_release(&st.root, &id) {
        Ok(m) => {
            let patches: Vec<_> = m
                .patches
                .iter()
                .filter(|p| q.path.as_deref().map_or(true, |path| p.page == path))
                .collect();
            json_reply(json!({"release": m.id, "channel": channel, "patches": patches}))
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
struct Beacon {
    event: String,
    release: Option<String>,
    sid: String,
    patch: Option<String>,
    detail: String,
}

/// Health beacons from the runtime. Error beacons count toward auto-rollback; "ok" and "missing" do not.
async fn beacon(State(st): State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    if body.len() > MAX_BEACON_BODY {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }
    let Ok(b) = serde_json::from_slice::<Beacon>(&body) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if b.event == "error" {
        if let Some(release_id) = b.release.as_deref() {
            let tripped = match st.beacons.lock() {
                Ok(mut window) => window.record_error(release_id, &b.sid, now_ms()),
                Err(_) => false,
            };
            if tripped {
                let evidence = format!("error beacons (patch {}): {}", b.patch.as_deref().unwrap_or("-"), b.detail);
                if let Err(e) = auto_rollback(&st.root, release_id, &evidence) {
                    eprintln!("pw-proxy: auto-rollback for {release_id} failed: {e}");
                }
            }
        }
    }
    StatusCode::NO_CONTENT.into_response()
}

/// Roll the release's channel back to its parent, if the channel still points at this release.
pub fn auto_rollback(root: &std::path::Path, release_id: &str, evidence: &str) -> anyhow::Result<Option<String>> {
    let manifest = release::read_release(root, release_id)?;
    if release::current(root, &manifest.channel)?.as_deref() != Some(release_id) {
        return Ok(None);
    }
    let back = release::rollback(root, &manifest.channel)?;
    AuditLog::open(root.join("audit")).append(
        "proxy",
        "auto-rollback",
        json!({"channel": manifest.channel, "from": release_id, "to": back, "evidence": evidence}),
    )?;
    Ok(Some(back))
}

#[derive(Deserialize)]
struct ChannelQuery {
    name: String,
}

/// Switch this browser to another channel (for example a preview), then go to the home page.
async fn set_channel(Query(q): Query<ChannelQuery>) -> Response {
    if !html::valid_channel(&q.name) {
        return (StatusCode::BAD_REQUEST, "invalid channel name").into_response();
    }
    (
        StatusCode::FOUND,
        [
            (header::LOCATION, "/".to_string()),
            (header::SET_COOKIE, format!("pw_channel={}; Path=/; SameSite=Lax", q.name)),
        ],
    )
        .into_response()
}

/// Everything not under `/__pw/`: forward to the upstream and, for HTML, inject the runtime.
async fn proxy(State(st): State<Arc<AppState>>, req: Request) -> Response {
    let path = req.uri().path().to_string();
    if path.starts_with("/__pw/") {
        return StatusCode::NOT_FOUND.into_response();
    }
    let (parts, body) = req.into_parts();
    let body = match to_bytes(body, MAX_REQUEST_BODY).await {
        Ok(b) => b,
        Err(_) => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
    };
    let url = format!("{}{}", st.upstream, parts.uri.path_and_query().map_or("/", |p| p.as_str()));

    let mut headers = reqwest::header::HeaderMap::new();
    for (name, value) in parts.headers.iter() {
        let n = name.as_str();
        if HOP_BY_HOP.contains(&n) || n == "accept-encoding" || n == "content-length" {
            continue;
        }
        headers.append(name.clone(), value.clone());
    }
    headers.insert(header::ACCEPT_ENCODING, HeaderValue::from_static("gzip, deflate, br"));

    let upstream = match st.client.request(parts.method.clone(), &url).headers(headers).body(body).send().await {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream unreachable: {e}")).into_response(),
    };
    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut out = HeaderMap::new();
    for (name, value) in upstream.headers().iter() {
        let n = name.as_str();
        if HOP_BY_HOP.contains(&n) || n == "content-length" || n == "content-security-policy" {
            continue;
        }
        out.append(name.clone(), value.clone());
    }
    let encoding = upstream.headers().get(header::CONTENT_ENCODING).and_then(|v| v.to_str().ok()).map(String::from);
    let is_html = status == StatusCode::OK
        && upstream
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.trim_start().to_ascii_lowercase().starts_with("text/html"));
    let policies: Vec<String> = upstream
        .headers()
        .get_all("content-security-policy")
        .iter()
        .filter_map(|v| v.to_str().ok().map(String::from))
        .collect();
    let raw = match upstream.bytes().await {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream body: {e}")).into_response(),
    };

    if is_html {
        if let Ok(Some(decoded)) = html::decode(&raw, encoding.as_deref()) {
            let mut nonce = None;
            for policy in &policies {
                let adjusted = html::adjust_csp(policy);
                nonce = nonce.or(adjusted.nonce.clone());
                if let Ok(v) = HeaderValue::from_str(&adjusted.header) {
                    out.append(HeaderName::from_static("content-security-policy"), v);
                }
            }
            let injected = html::inject_runtime(&decoded, nonce.as_deref());
            out.remove(header::CONTENT_ENCODING);
            out.append(header::VARY, HeaderValue::from_static("Cookie"));
            return (status, out, Body::from(injected)).into_response();
        }
        // Undecodable HTML is passed through unchanged, with its original policies.
    }
    for policy in &policies {
        if let Ok(v) = HeaderValue::from_str(policy) {
            out.append(HeaderName::from_static("content-security-policy"), v);
        }
    }
    (status, out, Body::from(raw)).into_response()
}
