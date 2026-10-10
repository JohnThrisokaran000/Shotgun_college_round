//! Print the model's top logits and greedy answer for a prompt. Used to check a real export against Python.
//!
//!     cargo run --release -p pw-model --example logits -- <model-dir> <prompt-file>

use pw_model::Model;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: logits <model-dir> <prompt-file>");
        std::process::exit(2);
    }
    let prompt = std::fs::read_to_string(&args[2]).expect("reading the prompt file");
    let mut model = Model::load(std::path::Path::new(&args[1])).expect("loading the model");
    let logits = model.last_logits(&prompt).expect("the forward pass");
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let top5: Vec<usize> = order[..5].to_vec();
    let first8: Vec<f32> = logits[..8].to_vec();
    let generated = model.generate(&prompt, 24).expect("generation");
    println!("{}", serde_json::json!({
        "top5_index": top5,
        "top5_logit": top5.iter().map(|&i| logits[i]).collect::<Vec<f32>>(),
        "logits_first8": first8,
        "greedy_text": generated.text,
    }));
}
