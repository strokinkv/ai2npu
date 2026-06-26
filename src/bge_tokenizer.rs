use anyhow::{anyhow, Result};
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub struct BgeTokenizer {
    tokenizer: Tokenizer,
}

/// Maximum number of tokens BGE input is truncated to. The model on the NPU has
/// a static input shape of 512 tokens; longer inputs are silently truncated by
/// the tokenizer, so callers should surface [`EncodedInput::truncated`].
pub const MAX_TOKENS: usize = 512;

pub struct EncodedInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    /// `true` when the input exceeded [`MAX_TOKENS`] and was truncated.
    pub truncated: bool,
}

impl BgeTokenizer {
    pub fn from_model_dir(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer_path = path.as_ref().join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("failed to load {}: {err}", tokenizer_path.display()))?;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|err| anyhow!("failed to configure tokenizer truncation: {err}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(MAX_TOKENS),
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
            // The tokenizer stores tokens that did not fit `MAX_TOKENS` as
            // overflowing encodings; a non-empty list means we truncated.
            truncated: !encoding.get_overflowing().is_empty(),
        })
    }
}
