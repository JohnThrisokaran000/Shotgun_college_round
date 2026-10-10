//! The Rust decoder against the Python reference (training/tests/make_rust_fixture.py), and the failure paths.

use std::path::{Path, PathBuf};

use pw_core::anchor::Anchor;
use pw_core::relocate::{Choice, Chooser};
use pw_core::snapshot::Snapshot;
use pw_model::{Model, ModelChooser};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixture")
}

fn expected() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("expected.json")).unwrap()).unwrap()
}

#[test]
fn logits_and_greedy_text_match_the_python_reference() {
    let want = expected();
    let mut model = Model::load(&fixture()).expect("the fixture loads");
    let logits = model.last_logits(want["prompt"].as_str().unwrap()).unwrap();
    assert_eq!(logits.len(), 32768);
    for (i, expect) in want["logits_first16"].as_array().unwrap().iter().enumerate() {
        let expect = expect.as_f64().unwrap() as f32;
        assert!((logits[i] - expect).abs() < 1e-3, "logit {i}: rust {} python {expect}", logits[i]);
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let top5: Vec<u64> = order[..5].iter().map(|&i| i as u64).collect();
    let python_top5: Vec<u64> = want["top5_index"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    assert_eq!(top5, python_top5, "the same five tokens must be the best");
    let generated = model.generate(want["prompt"].as_str().unwrap(), 8).unwrap();
    assert_eq!(generated.text, want["greedy_text"].as_str().unwrap());
    assert_eq!(generated.token_logprobs.len(), want["greedy_tokens"].as_array().unwrap().len());
}

#[test]
fn a_prompt_longer_than_the_context_is_refused_not_truncated() {
    let mut model = Model::load(&fixture()).unwrap();
    let long = "word ".repeat(2000);
    let error = model.generate(&long, 4).unwrap_err().to_string();
    assert!(error.contains("context"), "unexpected error: {error}");
}

#[test]
fn a_config_missing_its_architecture_is_refused() {
    let dir = std::env::temp_dir().join(format!("pw-model-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), "{\"model_type\": \"gemma3_text\"}").unwrap();
    let error = Model::load(&dir).err().expect("a config without its architecture must not load").to_string();
    std::fs::remove_dir_all(&dir).ok();
    assert!(error.contains("config.json has no"), "unexpected error: {error}");
}

#[test]
fn the_chooser_never_applies_an_answer_below_its_confidence_bar() {
    // A random model's answer is not a relocation. Whatever it writes, a pick is only returned when the model
    // is at least as confident as `model_accept`.
    let html = r#"<!doctype html><html><body><main><section>
        <p>Subtotal</p><button id="a">Add to cart</button><button id="b">Place order</button>
        </section></main></body></html>"#;
    let snap = Snapshot::parse(html);
    let gold = snap.elements().iter().find(|e| e.text.contains("Place order")).expect("target on the page");
    let anchor = Anchor::from_element(gold, &snap);
    let candidates: Vec<_> = snap.elements().iter().map(|e| (e, 1.0)).collect();
    let bar = 0.99;
    let chooser = ModelChooser::new(Model::load(&fixture()).unwrap(), bar);
    match chooser.choose(&anchor.fingerprint, &candidates).unwrap() {
        Choice::Unsure => {}
        Choice::Null => {}
        Choice::Pick { index, confidence } => {
            assert!(index < candidates.len());
            assert!(confidence >= bar, "a pick below the bar was applied: {confidence}");
        }
    }
}
