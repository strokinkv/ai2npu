use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

pub fn normalize_input(input: EmbeddingInput) -> Result<Vec<String>> {
    let values = match input {
        EmbeddingInput::Single(value) => vec![value],
        EmbeddingInput::Batch(values) => values,
    };

    if values.is_empty() {
        bail!("input must not be empty");
    }
    if values.iter().any(|value| value.is_empty()) {
        bail!("input strings must not be empty");
    }

    Ok(values)
}
