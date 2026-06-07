use std::path::PathBuf;

use serde::Serialize;

use crate::config::{AppConfig, ModelConfig};

#[derive(Debug, Clone)]
pub struct ModelRegistry {
    config: AppConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelBundleStatus {
    pub id: String,
    pub model_type: String,
    pub path: PathBuf,
    pub valid: bool,
    pub missing_files: Vec<String>,
}

impl ModelRegistry {
    pub fn new(config: AppConfig) -> Self {
        Self { config }
    }

    pub fn models(&self) -> &[ModelConfig] {
        &self.config.models
    }

    pub fn enabled_models(&self) -> impl Iterator<Item = &ModelConfig> {
        self.config.models.iter().filter(|model| model.enabled)
    }

    pub fn find_enabled_model(&self, id: &str) -> Option<&ModelConfig> {
        self.enabled_models().find(|model| model.id == id)
    }

    pub fn validate_bundle(&self, model: &ModelConfig) -> ModelBundleStatus {
        let missing_files = required_bundle_files(model.model_type.as_str())
            .into_iter()
            .filter(|file| !model.path.join(file).is_file())
            .collect::<Vec<_>>();

        ModelBundleStatus {
            id: model.id.clone(),
            model_type: model.model_type.to_string(),
            path: model.path.clone(),
            valid: missing_files.is_empty(),
            missing_files,
        }
    }

    pub fn validate_bundles(&self) -> Vec<ModelBundleStatus> {
        self.enabled_models()
            .map(|model| self.validate_bundle(model))
            .collect()
    }
}

pub fn required_bundle_files(model_type: &str) -> Vec<String> {
    match model_type {
        "embedding" => [
            "model.xml",
            "model.bin",
            "tokenizer.json",
            "config.json",
            "sentencepiece.bpe.model",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        "whisper" => [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "openvino_encoder_model.xml",
            "openvino_encoder_model.bin",
            "openvino_decoder_model.xml",
            "openvino_decoder_model.bin",
            "openvino_tokenizer.xml",
            "openvino_tokenizer.bin",
            "openvino_detokenizer.xml",
            "openvino_detokenizer.bin",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        _ => Vec::new(),
    }
}
