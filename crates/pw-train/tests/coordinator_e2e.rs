//! The coordinator over real HTTP: staleness window, quorum, and the fp16 global download.

use std::time::Duration;

use pw_train::coordinator::{self, Config, OuterOpt, Params};
use safetensors::tensor::Dtype;

fn params(values: &[f32]) -> Params {
    coordinator::params_from(vec![("w", vec![values.len()], values.to_vec())])
}

async fn spawn(workers: usize, quorum: usize, initial: Params) -> (String, tempfile_dir::Dir) {
    spawn_with(workers, quorum, initial, None).await
}

async fn spawn_with(workers: usize, quorum: usize, initial: Params, token: Option<String>) -> (String, tempfile_dir::Dir) {
    let dir = tempfile_dir::Dir::new("coord-e2e");
    let cfg = Config {
        workers,
        quorum,
        timeout: Duration::from_secs(3600),
        outer: OuterOpt { lr: 1.0, momentum: 0.0 },
        token,
        state_dir: dir.path().join("state"),
        serve_dir: None,
    };
    let app = coordinator::router(cfg, initial).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), dir)
}

async fn post_delta(c: &reqwest::Client, base: &str, worker: &str, round: u64, delta: &Params) -> u16 {
    let body = coordinator::to_safetensors(delta, Dtype::F16).unwrap();
    c.post(format!("{base}/v1/delta?worker={worker}&round={round}"))
        .body(body)
        .send()
        .await
        .unwrap()
        .status()
        .as_u16()
}

async fn version(c: &reqwest::Client, base: &str) -> u64 {
    let v: serde_json::Value = c.get(format!("{base}/v1/status")).send().await.unwrap().json().await.unwrap();
    v["round"].as_u64().unwrap()
}

#[tokio::test]
async fn one_version_stale_is_accepted_two_versions_stale_is_refused() {
    let (base, _dir) = spawn(3, 2, params(&[0.0, 0.0])).await;
    let c = reqwest::Client::new();
    let delta = params(&[0.1, 0.1]);

    assert_eq!(post_delta(&c, &base, "a", 0, &delta).await, 202);
    assert_eq!(post_delta(&c, &base, "b", 0, &delta).await, 202); // quorum of 2: version 0 closes
    assert_eq!(version(&c, &base).await, 1);

    assert_eq!(post_delta(&c, &base, "c", 0, &delta).await, 202, "based on version 0, one behind 1: accepted");
    assert_eq!(post_delta(&c, &base, "d", 1, &delta).await, 202); // pending now 2 (c and d): version 1 closes
    assert_eq!(version(&c, &base).await, 2);

    assert_eq!(post_delta(&c, &base, "e", 0, &delta).await, 409, "two behind 2: refused");
    assert_eq!(post_delta(&c, &base, "f", 9, &delta).await, 409, "a version that does not exist yet: refused");
}

#[tokio::test]
async fn fp16_global_download_parses_and_matches_f32_closely() {
    let initial = params(&[0.25, -1.5, 3.0]);
    let (base, _dir) = spawn(1, 1, initial.clone()).await;
    let c = reqwest::Client::new();
    let f32_body = c.get(format!("{base}/v1/global")).send().await.unwrap().bytes().await.unwrap();
    let f16_body = c.get(format!("{base}/v1/global?dtype=f16")).send().await.unwrap().bytes().await.unwrap();
    let exact = coordinator::parse_safetensors(&f32_body).unwrap();
    let half = coordinator::parse_safetensors(&f16_body).unwrap();
    assert_eq!(exact, initial);
    for (a, b) in half.tensors["w"].1.iter().zip(&initial.tensors["w"].1) {
        assert!((a - b).abs() < 1e-3);
    }
    assert!(f16_body.len() < f32_body.len());
    let i8_body = c.get(format!("{base}/v1/global?dtype=int8")).send().await.unwrap().bytes().await.unwrap();
    let quantised = coordinator::parse_int8_safetensors(&i8_body).unwrap();
    for (a, b) in quantised.tensors["w"].1.iter().zip(&initial.tensors["w"].1) {
        assert!((a - b).abs() < 0.02, "int8 global strays from the f32 weights: {a} vs {b}");
    }
    let bad = c.get(format!("{base}/v1/global?dtype=bogus")).send().await.unwrap();
    assert_eq!(bad.status().as_u16(), 400);
}

#[tokio::test]
async fn every_route_needs_the_token_when_one_is_set() {
    let token = "0123456789abcdef-test".to_string();
    let (base, dir) = spawn_with(2, 2, params(&[0.0, 0.0]), Some(token.clone())).await;
    let c = reqwest::Client::new();
    assert_eq!(c.get(format!("{base}/v1/status")).send().await.unwrap().status().as_u16(), 401);
    assert_eq!(c.get(format!("{base}/v1/global")).send().await.unwrap().status().as_u16(), 401);
    let wrong = c.get(format!("{base}/v1/status")).bearer_auth("not-the-token").send().await.unwrap();
    assert_eq!(wrong.status().as_u16(), 401);
    let good = c.get(format!("{base}/v1/status")).bearer_auth(&token).send().await.unwrap();
    assert_eq!(good.status().as_u16(), 200);

    let events = serde_json::json!([{"ev": "step", "group": "g0", "step": 10, "loss": 2.5}]);
    assert_eq!(c.post(format!("{base}/v1/events")).json(&events).send().await.unwrap().status().as_u16(), 401);
    let posted = c.post(format!("{base}/v1/events")).bearer_auth(&token).json(&events).send().await.unwrap();
    assert_eq!(posted.status().as_u16(), 202);
    let log = std::fs::read_to_string(dir.path().join("state").join("events.jsonl")).unwrap();
    assert!(log.contains("\"group\":\"g0\"") && log.contains("\"received\""), "event not recorded: {log}");

    let status: serde_json::Value = c.get(format!("{base}/v1/status")).bearer_auth(&token).send().await.unwrap().json().await.unwrap();
    assert!(status["last_seen"]["g0"].as_u64().is_some(), "worker last-seen time missing: {status}");
}

#[tokio::test]
async fn a_restart_keeps_the_deltas_that_were_waiting() {
    let dir = tempfile_dir::Dir::new("coord-restart");
    let cfg = || Config {
        workers: 3,
        quorum: 2,
        timeout: Duration::from_secs(3600),
        outer: OuterOpt { lr: 1.0, momentum: 0.0 },
        token: None,
        state_dir: dir.path().join("state"),
        serve_dir: None,
    };
    let serve = |app: axum::Router| async move {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), task)
    };
    let c = reqwest::Client::new();
    let delta = params(&[0.5, 0.5]);

    let (base, first) = serve(coordinator::router(cfg(), params(&[0.0, 0.0])).unwrap()).await;
    assert_eq!(post_delta(&c, &base, "a", 0, &delta).await, 202);
    first.abort(); // the coordinator goes away with one delta pending

    let (base, second) = serve(coordinator::router(cfg(), params(&[0.0, 0.0])).unwrap()).await;
    let status: serde_json::Value = c.get(format!("{base}/v1/status")).send().await.unwrap().json().await.unwrap();
    assert_eq!(status["received_this_round"], 1, "the waiting delta was lost: {status}");
    assert_eq!(post_delta(&c, &base, "b", 0, &delta).await, 202); // quorum of 2 closes the round
    assert_eq!(version(&c, &base).await, 1);
    let global = coordinator::parse_safetensors(&c.get(format!("{base}/v1/global")).send().await.unwrap().bytes().await.unwrap()).unwrap();
    assert!((global.tensors["w"].1[0] + 0.5).abs() < 1e-3, "both deltas should be averaged in: {:?}", global.tensors["w"].1);
    assert!(!dir.path().join("state").join("pending").exists(), "closed rounds leave nothing pending on disk");
    second.abort();
}

#[tokio::test]
async fn ranged_global_download_returns_exactly_the_requested_bytes() {
    let initial = params(&[0.25, -1.5, 3.0, 0.5, 0.75]);
    let (base, _dir) = spawn(1, 1, initial).await;
    let c = reqwest::Client::new();
    let whole = c.get(format!("{base}/v1/global?dtype=int8")).send().await.unwrap().bytes().await.unwrap();
    let part = c
        .get(format!("{base}/v1/global?dtype=int8"))
        .header("Range", "bytes=0-9")
        .send()
        .await
        .unwrap();
    assert_eq!(part.status().as_u16(), 206);
    let range = part.headers().get("content-range").unwrap().to_str().unwrap().to_string();
    assert_eq!(range, format!("bytes 0-9/{}", whole.len()));
    assert_eq!(part.bytes().await.unwrap().as_ref(), &whole[0..10]);
    let tail = c
        .get(format!("{base}/v1/global?dtype=int8"))
        .header("Range", format!("bytes={}-", whole.len() - 4))
        .send()
        .await
        .unwrap();
    assert_eq!(tail.bytes().await.unwrap().as_ref(), &whole[whole.len() - 4..]);
    let clipped = c
        .get(format!("{base}/v1/global?dtype=int8"))
        .header("Range", format!("bytes={}-{}", whole.len() - 2, whole.len() + 500))
        .send()
        .await
        .unwrap();
    assert_eq!(clipped.status().as_u16(), 206);
    assert_eq!(clipped.bytes().await.unwrap().as_ref(), &whole[whole.len() - 2..]);
    let bad = c
        .get(format!("{base}/v1/global?dtype=int8"))
        .header("Range", format!("bytes={}-{}", whole.len(), whole.len() + 5))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status().as_u16(), 416);
}

#[tokio::test]
async fn a_delta_uploaded_in_parts_is_accepted_once_complete() {
    let (base, _dir) = spawn(1, 1, params(&[0.0, 0.0])).await;
    let c = reqwest::Client::new();
    let body = coordinator::to_safetensors(&params(&[0.5, 0.5]), Dtype::F16).unwrap();
    let parts: Vec<&[u8]> = body.chunks(body.len() / 3 + 1).collect();
    let n = parts.len();
    for (i, chunk) in parts.iter().enumerate().take(n - 1) {
        let r = c
            .post(format!("{base}/v1/delta-part?worker=g0&round=0&seq=5&part={i}&parts={n}"))
            .body(chunk.to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(r.status().as_u16(), 202);
        let j: serde_json::Value = r.json().await.unwrap();
        assert_eq!(j["complete"], false);
    }
    // A retried part is harmless: it overwrites the same slot.
    let again = c
        .post(format!("{base}/v1/delta-part?worker=g0&round=0&seq=5&part=0&parts={n}"))
        .body(parts[0].to_vec())
        .send()
        .await
        .unwrap();
    assert_eq!(again.status().as_u16(), 202);
    let last = c
        .post(format!("{base}/v1/delta-part?worker=g0&round=0&seq=5&part={}&parts={n}", n - 1))
        .body(parts[n - 1].to_vec())
        .send()
        .await
        .unwrap();
    assert_eq!(last.status().as_u16(), 202);
    let j: serde_json::Value = last.json().await.unwrap();
    assert_eq!(j["accepted_into"], 0, "the complete delta should be accepted: {j}");
    assert_eq!(version(&c, &base).await, 1, "quorum of one: the round should close");
    let global = coordinator::parse_safetensors(&c.get(format!("{base}/v1/global")).send().await.unwrap().bytes().await.unwrap()).unwrap();
    assert!((global.tensors["w"].1[0] + 0.5).abs() < 1e-3, "global = theta - delta with lr 1: {:?}", global.tensors["w"].1);
}

mod tempfile_dir {
    use std::path::{Path, PathBuf};
    pub struct Dir(PathBuf);
    impl Dir {
        pub fn new(name: &str) -> Dir {
            let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp").join(format!("{name}-{}-{}", std::process::id(), rand_suffix()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Dir(p)
        }
        pub fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn rand_suffix() -> u128 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
    }
}
