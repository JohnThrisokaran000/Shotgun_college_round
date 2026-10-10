//! The decoder itself: loads an exported folder and decodes greedily with a KV cache.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::gemma3::{Config, Model as Net};
use pw_core::error::{PwError, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

fn err(message: impl Into<String>) -> PwError {
    PwError::Other(message.into())
}

fn from_candle(e: candle_core::Error) -> PwError {
    err(format!("model: {e}"))
}

/// What the model produced, with how sure it was of each token.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub text: String,
    /// Log-probability of each generated token under the model.
    pub token_logprobs: Vec<f32>,
}

impl Generation {
    /// exp(mean log-probability) of the generated tokens: 1.0 means the model was certain of every token.
    /// An empty generation has confidence 0.
    pub fn confidence(&self) -> f64 {
        if self.token_logprobs.is_empty() {
            return 0.0;
        }
        let mean = self.token_logprobs.iter().map(|&x| x as f64).sum::<f64>() / self.token_logprobs.len() as f64;
        mean.exp()
    }
}

/// A loaded model. Holds a KV cache, so one `Model` decodes one prompt at a time.
pub struct Model {
    net: Net,
    tokenizer: Tokenizer,
    device: Device,
    bos: u32,
    eos: u32,
    context: usize,
}

/// Read the architecture from a config.json written by `training/prepare.py` or `training/eval_checkpoint.py`.
/// Only the keys the architecture needs are read. Anything unknown is refused rather than guessed.
fn read_config(path: &Path) -> Result<(Config, Value)> {
    let text = std::fs::read_to_string(path).map_err(|e| err(format!("reading {}: {e}", path.display())))?;
    let raw: Value = serde_json::from_str(&text).map_err(|e| err(format!("{}: {e}", path.display())))?;
    let uint = |key: &str| -> Result<usize> {
        raw.get(key)
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| err(format!("config.json has no integer `{key}`")))
    };
    let theta = |layer: &str| -> Result<f64> {
        raw.pointer(&format!("/rope_parameters/{layer}/rope_theta"))
            .and_then(Value::as_f64)
            .ok_or_else(|| err(format!("config.json has no rope_parameters.{layer}.rope_theta")))
    };
    // layer_types marks each layer sliding or full. Full attention every `pattern` layers is candle's convention.
    let pattern = raw
        .get("layer_types")
        .and_then(Value::as_array)
        .and_then(|types| types.iter().position(|t| t == "full_attention").map(|i| i + 1))
        .unwrap_or(6);
    // candle's sliding-window mask admits one position more than transformers' does, so the two only agree
    // when no position is ever outside the window. PatchWork's configs set the window to the context length.
    // Anything else is refused rather than answered slightly wrong.
    let context = uint("max_position_embeddings")?;
    let window = uint("sliding_window")?;
    if window < context {
        return Err(err(format!(
            "sliding_window {window} is shorter than the {context}-token context; this decoder is verified only when they are equal"
        )));
    }
    let cfg = Config {
        attention_bias: false,
        head_dim: uint("head_dim")?,
        hidden_activation: match raw.get("hidden_activation").and_then(Value::as_str) {
            Some("gelu_pytorch_tanh") => Activation::GeluPytorchTanh,
            other => {
                return Err(err(format!(
                    "unsupported hidden_activation {other:?}; the PatchWork model uses gelu_pytorch_tanh"
                )))
            }
        },
        hidden_size: uint("hidden_size")?,
        intermediate_size: uint("intermediate_size")?,
        num_attention_heads: uint("num_attention_heads")?,
        num_hidden_layers: uint("num_hidden_layers")?,
        num_key_value_heads: uint("num_key_value_heads")?,
        rms_norm_eps: raw.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6),
        rope_theta: theta("full_attention")?,
        rope_local_base_freq: theta("sliding_attention")?,
        vocab_size: uint("vocab_size")?,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: uint("query_pre_attn_scalar")?,
        sliding_window: uint("sliding_window")?,
        sliding_window_pattern: pattern,
        max_position_embeddings: uint("max_position_embeddings")?,
    };
    Ok((cfg, raw))
}

impl Model {
    /// Load an exported folder. Runs on the CPU in f32, which is exact with the training code's checkpoints.
    pub fn load(dir: &Path) -> Result<Self> {
        let (cfg, raw) = read_config(&dir.join("config.json"))?;
        let token = |key: &str| -> Result<u32> {
            raw.get(key)
                .and_then(Value::as_u64)
                .map(|v| v as u32)
                .ok_or_else(|| err(format!("config.json has no `{key}`")))
        };
        let device = Device::Cpu;
        let weights = dir.join("model.safetensors");
        // Safety: the file is read-only for the life of the model and is not written by this process.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device) }.map_err(from_candle)?;
        let net = Net::new(false, &cfg, vb).map_err(from_candle)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err(format!("tokenizer.json: {e}")))?;
        Ok(Self {
            net,
            tokenizer,
            device,
            bos: token("bos_token_id")?,
            eos: token("eos_token_id")?,
            context: cfg.max_position_embeddings,
        })
    }

    /// Positions this model can take, including the BOS token.
    pub fn context(&self) -> usize {
        self.context
    }

    /// Logits for the last of `ids`, which start at position `offset`.
    fn step(&mut self, ids: &[u32], offset: usize) -> Result<Vec<f32>> {
        let input = Tensor::new(ids, &self.device).map_err(from_candle)?.unsqueeze(0).map_err(from_candle)?;
        let logits = self.net.forward(&input, offset).map_err(from_candle)?;
        // candle returns [1, 1, vocab] for one position; flattening gives the vocabulary either way.
        logits.flatten_all().map_err(from_candle)?.to_vec1::<f32>().map_err(from_candle)
    }

    /// The logits after `<bos>` and the prompt, before any token is generated. Used to check the decoder
    /// against the Python reference, token for token.
    pub fn last_logits(&mut self, prompt: &str) -> Result<Vec<f32>> {
        let encoded = self.tokenizer.encode(prompt, false).map_err(|e| err(format!("tokenizing: {e}")))?;
        let mut context = vec![self.bos];
        context.extend_from_slice(encoded.get_ids());
        if context.len() > self.context {
            return Err(err(format!("prompt is {} tokens; the model's context is {}", context.len(), self.context)));
        }
        self.net.clear_kv_cache();
        self.step(&context, 0)
    }

    /// Greedy decoding after `<bos>` and the prompt, stopping at `<eos>` or after `max_new` tokens.
    /// A prompt that does not fit the context is an error, not a truncation.
    pub fn generate(&mut self, prompt: &str, max_new: usize) -> Result<Generation> {
        let encoded = self.tokenizer.encode(prompt, false).map_err(|e| err(format!("tokenizing: {e}")))?;
        let mut context = vec![self.bos];
        context.extend_from_slice(encoded.get_ids());
        if context.len() > self.context {
            return Err(err(format!(
                "prompt is {} tokens; the model's context is {}",
                context.len(),
                self.context
            )));
        }
        self.net.clear_kv_cache();
        let mut logits = self.step(&context, 0)?;
        let mut produced: Vec<u32> = Vec::new();
        let mut logprobs: Vec<f32> = Vec::new();
        for _ in 0..max_new {
            let (token, logprob) = greedy(&logits)?;
            if token == self.eos {
                break;
            }
            produced.push(token);
            logprobs.push(logprob);
            let position = context.len() + produced.len() - 1;
            if position >= self.context {
                break;
            }
            logits = self.step(&[token], position)?;
        }
        let text = self.tokenizer.decode(&produced, true).map_err(|e| err(format!("detokenizing: {e}")))?;
        Ok(Generation { text: text.trim().to_string(), token_logprobs: logprobs })
    }
}

/// The highest logit and its log-probability under the full softmax.
fn greedy(logits: &[f32]) -> Result<(u32, f32)> {
    let (best, &top) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| err("the model returned no logits"))?;
    if !top.is_finite() {
        return Err(err("the model returned non-finite logits"));
    }
    let max = top as f64;
    let log_sum = logits.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln() + max;
    Ok((best as u32, (top as f64 - log_sum) as f32))
}
