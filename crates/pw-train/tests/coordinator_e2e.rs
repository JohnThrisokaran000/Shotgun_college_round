//! The coordinator over real HTTP: staleness window, quorum, and the fp16 global download.

use std::time::Duration;

use pw_train::coordinator::{self, Config, OuterOpt, Params};
use safetensors::tensor::Dtype;

fn params(values: &[f32]) -> Params {
    coordinator::params_from(vec![("w", vec![values.len()], values.to_vec())])
}

async fn spawn(workers: usize, quorum: usize, initial: Params) -> (String, tempfile_dir::Dir) {
    let dir = tempfile_dir::Dir::new("coord-e2e");
    let cfg = Config {
        workers,
        quorum,
        timeout: Duration::from_secs(3600),
        outer: OuterOpt { lr: 1.0, momentum: 0.0 },
        token: None,
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
    let bad = c.get(format!("{base}/v1/global?dtype=int8")).send().await.unwrap();
    assert_eq!(bad.status().as_u16(), 400);
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
