use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use crate::bge_tokenizer::BgeTokenizer;
use crate::config::ModelConfig;
use crate::inference::EmbeddingExecutor;
use crate::openvino_c::{CompiledModel, OpenVinoRuntime};

pub struct BgeEmbeddingExecutor {
    runtime: OpenVinoRuntime,
    compiled_models: Mutex<HashMap<PathBuf, Arc<CompiledModel>>>,
}

impl BgeEmbeddingExecutor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            runtime: OpenVinoRuntime::new()?,
            compiled_models: Mutex::new(HashMap::new()),
        })
    }

    fn compiled_model(&self, model: &ModelConfig) -> Result<Arc<CompiledModel>> {
        let model_path = model.path.join("model.xml");
        if let Some(compiled) = self
            .compiled_models
            .lock()
            .expect("compiled model cache poisoned")
            .get(&model_path)
            .cloned()
        {
            return Ok(compiled);
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

        self.compiled_models
            .lock()
            .expect("compiled model cache poisoned")
            .insert(model_path, compiled.clone());
        Ok(compiled)
    }
}

impl EmbeddingExecutor for BgeEmbeddingExecutor {
    fn preload(&self, model: &ModelConfig) -> Result<()> {
        self.compiled_model(model).map(|_| ())
    }

    fn embed(&self, model: &ModelConfig, input: &[String]) -> Result<Vec<Vec<f32>>> {
        let tokenizer = BgeTokenizer::from_model_dir(&model.path)?;
        let compiled = self.compiled_model(model)?;
        let mut embeddings = Vec::with_capacity(input.len());

        for text in input {
            let encoded = tokenizer.encode(text)?;
            let mut embedding = compiled.infer_i64_inputs_to_f32_output(
                &[
                    ("input_ids", &encoded.input_ids),
                    ("attention_mask", &encoded.attention_mask),
                ],
                &[1, 512],
                "sentence_embedding",
            )?;

            if model.normalize.unwrap_or(true) {
                normalize_l2(&mut embedding);
            }
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    fn loaded_models(&self) -> Vec<String> {
        self.compiled_models
            .lock()
            .expect("compiled model cache poisoned")
            .keys()
            .map(|path| path.display().to_string())
            .collect()
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
