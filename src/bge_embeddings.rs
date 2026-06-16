use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use crate::bge_tokenizer::BgeTokenizer;
use crate::config::ModelConfig;
use crate::inference::{lock_recover, EmbeddingExecutor, InferenceFailed, ModelLoadFailed};
use crate::openvino_c::{CompiledModel, OpenVinoRuntime};

struct LoadedModel {
    id: String,
    compiled: Arc<CompiledModel>,
    tokenizer: Arc<BgeTokenizer>,
}

pub struct BgeEmbeddingExecutor {
    runtime: OpenVinoRuntime,
    loaded: Mutex<HashMap<PathBuf, LoadedModel>>,
}

impl BgeEmbeddingExecutor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            runtime: OpenVinoRuntime::new()?,
            loaded: Mutex::new(HashMap::new()),
        })
    }

    /// Return the compiled model and tokenizer for `model`, loading and caching
    /// both (including the multi-megabyte `tokenizer.json`) on first use so they
    /// are not re-read from disk on every request.
    fn load(&self, model: &ModelConfig) -> Result<(Arc<CompiledModel>, Arc<BgeTokenizer>)> {
        let model_path = model.path.join("model.xml");
        if let Some(entry) = lock_recover(&self.loaded).get(&model_path) {
            return Ok((Arc::clone(&entry.compiled), Arc::clone(&entry.tokenizer)));
        }

        let model_ir = self
            .runtime
            .read_model(&model_path)
            .with_context(|| format!("failed to read model {}", model_path.display()))?;
        let compiled = Arc::new(
            self.runtime
                .compile_model(&model_ir, "NPU")
                .with_context(|| {
                    format!("failed to compile model {} on NPU", model_path.display())
                })?,
        );
        let tokenizer = Arc::new(BgeTokenizer::from_model_dir(&model.path)?);

        let mut loaded = lock_recover(&self.loaded);
        let entry = loaded.entry(model_path).or_insert_with(|| LoadedModel {
            id: model.id.clone(),
            compiled: Arc::clone(&compiled),
            tokenizer: Arc::clone(&tokenizer),
        });
        Ok((Arc::clone(&entry.compiled), Arc::clone(&entry.tokenizer)))
    }
}

impl EmbeddingExecutor for BgeEmbeddingExecutor {
    fn preload(&self, model: &ModelConfig) -> Result<()> {
        self.load(model)
            .map(|_| ())
            .map_err(|e| ModelLoadFailed(e).into())
    }

    fn embed(&self, model: &ModelConfig, input: &[String]) -> Result<Vec<Vec<f32>>> {
        let (compiled, tokenizer) = self.load(model).map_err(ModelLoadFailed)?;
        let mut embeddings = Vec::with_capacity(input.len());

        for text in input {
            let encoded = tokenizer.encode(text).map_err(InferenceFailed)?;
            let mut embedding = compiled
                .infer_i64_inputs_to_f32_output(
                    &[
                        ("input_ids", &encoded.input_ids),
                        ("attention_mask", &encoded.attention_mask),
                    ],
                    &[1, 512],
                    "sentence_embedding",
                )
                .map_err(InferenceFailed)?;

            if model.normalize.unwrap_or(true) {
                normalize_l2(&mut embedding);
            }
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    fn loaded_models(&self) -> Vec<String> {
        lock_recover(&self.loaded)
            .values()
            .map(|entry| entry.id.clone())
            .collect()
    }

    fn unload_all(&self) -> Result<usize> {
        let mut loaded = lock_recover(&self.loaded);
        let unloaded = loaded.len();
        loaded.clear();
        Ok(unloaded)
    }
}

fn normalize_l2(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values {
            *value /= norm;
        }
    }
}
