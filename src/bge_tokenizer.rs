use anyhow::{anyhow, Result};
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub struct BgeTokenizer {
    tokenizer: Tokenizer,
}

pub struct EncodedInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
}

impl BgeTokenizer {
    pub fn from_model_dir(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer_path = path.as_ref().join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("failed to load {}: {err}", tokenizer_path.display()))?;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(|err| anyhow!("failed to configure tokenizer truncation: {err}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(512),
            ..Default::default()
        }));

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<EncodedInput> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|err| anyhow!("failed to tokenize input: {err}"))?;

        Ok(EncodedInput {
            input_ids: encoding
                .get_ids()
                .iter()
                .map(|value| i64::from(*value))
                .collect(),
            attention_mask: encoding
                .get_attention_mask()
                .iter()
                .map(|value| i64::from(*value))
                .collect(),
        })
    }
}
