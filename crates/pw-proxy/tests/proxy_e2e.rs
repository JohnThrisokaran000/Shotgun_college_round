//! End-to-end: a real upstream site, a real proxy in front of it, and a real release store.

use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{header, HeaderValue};
use axum::routing::get;
use axum::Router;
use flate2::write::GzEncoder;
use flate2::Compression;
use pw_core::patch::Patch;
use pw_core::release::{self, BeaconPolicy, Manifest};
use pw_proxy::Config;

const PAGE: &str = "<!doctype html><html><head><title>Shop</title></head><body><button id=\"buy\">Buy now</button></body></html>";

const PATCH: &str = r##"
id: fix-buy
author: dev-a
page: /
anchor:
  locators:
    css: "#buy"
  fingerprint:
    tag: button
    text: "Buy now"
ops:
  - { op: setText, value: "Place order" }
assert:
  - { visible: true }
"##;

fn project_root(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-tmp")
        .join(format!("proxy-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
}

async fn spawn(app: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

/// An upstream site: gzipped HTML under a nonce-based CSP, plus a JSON API.
async fn upstream() -> SocketAddr {
    let app = Router::new()
        .route(
            "/",
            get(|| async {
                let mut enc = GzEncoder::new(Vec::new(), Compression::default());
                enc.write_all(PAGE.as_bytes()).unwrap();
                let body = enc.finish().unwrap();
                (
                    [
                        (header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8")),
                        (header::CONTENT_ENCODING, HeaderValue::from_static("gzip")),
                        (header::CONTENT_SECURITY_POLICY, HeaderValue::from_static("script-src 'nonce-abc123' 'strict-dynamic'")),
                    ],
                    body,
                )
            }),
        )
        .route("/api/data", get(|| async { ([(header::CONTENT_TYPE, "application/json")], "{\"ok\":true}") }));
    spawn(app).await
}

/// Two releases on production: r1 (no patch) is the parent of r2 (the patch).
fn seed_releases(root: &Path) -> (String, String) {
    let patch = Patch::from_yaml(PATCH).unwrap();
    let r1 = Manifest::build("production", None, now(), &[]).unwrap();
    release::write_release(root, &r1).unwrap();
    let r2 = Manifest::build("production", Some(r1.id.clone()), now(), &[patch]).unwrap();
    release::write_release(root, &r2).unwrap();
    release::set_channel(root, "production", &r2.id).unwrap();
    (r1.id, r2.id)
}

async fn start_proxy(root: &Path, upstream: SocketAddr) -> SocketAddr {
    let app = pw_proxy::router(Config {
        upstream: format!("http://{upstream}"),
        root: root.to_path_buf(),
        policy: BeaconPolicy { window_ms: 60_000, min_errors: 5, min_sessions: 3 },
    })
    .unwrap();
    spawn(app).await
}

#[tokio::test]
async fn html_is_decoded_injected_and_keeps_its_nonce() {
    let root = project_root("inject");
    seed_releases(&root);
    let proxy = start_proxy(&root, upstream().await).await;
    let client = reqwest::Client::new();

    let resp = client.get(format!("http://{proxy}/")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let csp = resp.headers().get("content-security-policy").unwrap().to_str().unwrap().to_string();
    assert!(csp.contains("'nonce-abc123'"), "{csp}");
    assert!(resp.headers().get("content-encoding").is_none(), "decoded body must not claim a content-encoding");
    let body = resp.text().await.unwrap();
    assert!(body.contains("<script src=\"/__pw/runtime.js\" nonce=\"abc123\" defer></script>"), "{body}");
    assert!(body.contains("Buy now"), "upstream content must survive: {body}");
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn non_html_passes_through_untouched() {
    let root = project_root("passthrough");
    seed_releases(&root);
    let proxy = start_proxy(&root, upstream().await).await;
    let body = reqwest::get(format!("http://{proxy}/api/data")).await.unwrap().text().await.unwrap();
    assert_eq!(body, "{\"ok\":true}");
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn manifest_serves_the_channel_release_and_filters_by_page() {
    let root = project_root("manifest");
    let (_r1, r2) = seed_releases(&root);
    let proxy = start_proxy(&root, upstream().await).await;
    let client = reqwest::Client::new();

    let v: serde_json::Value = client
        .get(format!("http://{proxy}/__pw/manifest?path=/"))
        .header("cookie", "pw_channel=production")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["release"], r2);
    assert_eq!(v["patches"][0]["id"], "fix-buy");

    let other_page: serde_json::Value = client
        .get(format!("http://{proxy}/__pw/manifest?path=/cart"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(other_page["patches"].as_array().unwrap().len(), 0);

    let preview: serde_json::Value = client
        .get(format!("http://{proxy}/__pw/manifest?path=/"))
        .header("cookie", "pw_channel=preview/nothing-yet")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(preview["release"].is_null(), "an unreleased preview channel serves no patches");
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn runtime_and_editor_are_served_and_unknown_pw_paths_are_404() {
    let root = project_root("assets");
    seed_releases(&root);
    let proxy = start_proxy(&root, upstream().await).await;
    let runtime = reqwest::get(format!("http://{proxy}/__pw/runtime.js")).await.unwrap();
    assert_eq!(runtime.status(), 200);
    assert!(runtime.text().await.unwrap().contains("window.__pw"));
    let editor = reqwest::get(format!("http://{proxy}/__pw/editor.js")).await.unwrap();
    assert!(editor.text().await.unwrap().contains("__pwSave"));
    let missing = reqwest::get(format!("http://{proxy}/__pw/nope")).await.unwrap();
    assert_eq!(missing.status(), 404);
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn error_beacons_from_several_sessions_roll_the_channel_back_and_are_audited() {
    let root = project_root("rollback");
    let (r1, r2) = seed_releases(&root);
    let proxy = start_proxy(&root, upstream().await).await;
    let client = reqwest::Client::new();

    // Too few errors: no rollback yet.
    for i in 0..3 {
        let body = format!(r#"{{"event":"error","release":"{r2}","sid":"s{i}","patch":"fix-buy","detail":"TypeError"}}"#);
        client.post(format!("http://{proxy}/__pw/beacon")).body(body).send().await.unwrap();
    }
    assert_eq!(release::current(&root, "production").unwrap().as_deref(), Some(r2.as_str()));

    // Enough errors from enough sessions: rolled back to the parent.
    for i in 0..3 {
        let body = format!(r#"{{"event":"error","release":"{r2}","sid":"s{}","patch":"fix-buy","detail":"TypeError"}}"#, i % 3);
        client.post(format!("http://{proxy}/__pw/beacon")).body(body).send().await.unwrap();
    }
    assert_eq!(release::current(&root, "production").unwrap().as_deref(), Some(r1.as_str()));

    let events = pw_core::audit::AuditLog::open(root.join("audit")).read_all().unwrap();
    assert!(events.iter().any(|e| e.kind == "auto-rollback" && e.data["from"] == r2.as_str()), "{events:?}");
    pw_core::audit::AuditLog::open(root.join("audit")).verify().unwrap();
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn ok_and_missing_beacons_never_trigger_rollback() {
    let root = project_root("nobeacon");
    let (_r1, r2) = seed_releases(&root);
    let proxy = start_proxy(&root, upstream().await).await;
    let client = reqwest::Client::new();
    for i in 0..10 {
        let body = format!(r#"{{"event":"missing","release":"{r2}","sid":"s{}","patch":"fix-buy","detail":"/"}}"#, i % 3);
        client.post(format!("http://{proxy}/__pw/beacon")).body(body).send().await.unwrap();
    }
    assert_eq!(release::current(&root, "production").unwrap().as_deref(), Some(r2.as_str()));
    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn malformed_and_oversized_beacons_are_rejected() {
    let root = project_root("badbeacon");
    seed_releases(&root);
    let proxy = start_proxy(&root, upstream().await).await;
    let client = reqwest::Client::new();
    let bad = client.post(format!("http://{proxy}/__pw/beacon")).body("not json").send().await.unwrap();
    assert_eq!(bad.status(), 400);
    let huge = client.post(format!("http://{proxy}/__pw/beacon")).body("x".repeat(5000)).send().await.unwrap();
    assert_eq!(huge.status(), 413);
    std::fs::remove_dir_all(&root).ok();
}
