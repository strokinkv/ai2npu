use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// Top-level service configuration loaded from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub queue: QueueConfig,
    pub logging: LoggingConfig,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

/// HTTP listener and request handling limits.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub request_body_limit_mb: u64,
    pub thread_count: usize,
}

/// Shared inference queue limits.
#[derive(Debug, Clone, Deserialize)]
pub struct QueueConfig {
    pub max_pending_requests: usize,
    pub default_timeout_sec: u64,
}

/// File logging and rotation settings.
#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub directory: PathBuf,
    pub max_file_size_mb: u64,
    pub max_files: usize,
}

/// One configured OpenAI-compatible model endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    #[serde(rename = "type")]
    pub model_type: ModelType,
    pub path: PathBuf,
    pub enabled: bool,
    pub preload: bool,
    pub queue_timeout_sec: u64,
    #[serde(default)]
    pub normalize: Option<bool>,
    #[serde(default)]
    pub max_audio_duration_sec: Option<u64>,
}

/// Supported model families in the service configuration.
#[derive(Debug, Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Embedding,
    Whisper,
}

impl ModelType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Whisper => "whisper",
        }
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let cfg: Self = toml::from_str(&text)
            .with_context(|| format!("failed to parse config {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.server.host != "127.0.0.1" {
            bail!("server.host must be 127.0.0.1");
        }
        if self.server.port == 0 {
            bail!("server.port must be greater than 0");
        }
        if self.server.request_body_limit_mb == 0 {
            bail!("server.request_body_limit_mb must be greater than 0");
        }
        if self.server.thread_count <= self.queue.max_pending_requests + 1 {
            bail!("server.thread_count must be greater than queue.max_pending_requests + 1");
        }
        if self.queue.default_timeout_sec == 0 {
            bail!("queue.default_timeout_sec must be greater than 0");
        }
        if self.logging.max_file_size_mb == 0 {
            bail!("logging.max_file_size_mb must be greater than 0");
        }
        if self.logging.max_files == 0 {
            bail!("logging.max_files must be greater than 0");
        }
        if !matches!(
            self.logging.level.as_str(),
            "trace" | "debug" | "info" | "warn" | "error"
        ) {
            bail!("logging.level must be one of trace, debug, info, warn, error");
        }
        let mut model_ids = HashSet::new();
        for model in &self.models {
            validate_model(model)?;
            if !model_ids.insert(&model.id) {
                bail!("models.id must be unique: {}", model.id);
            }
        }

        Ok(())
    }
}

fn validate_model(model: &ModelConfig) -> Result<()> {
    if model.id.trim().is_empty() {
        bail!("models.id must not be empty");
    }
    if model.path.as_os_str().is_empty() {
        bail!("models.path must not be empty");
    }
    if model.queue_timeout_sec == 0 {
        bail!("models.queue_timeout_sec must be greater than 0");
    }

    match model.model_type {
        ModelType::Embedding => {
            if model.max_audio_duration_sec.is_some() {
                bail!("embedding models must not set max_audio_duration_sec");
            }
        }
        ModelType::Whisper => {
            if model.normalize.is_some() {
                bail!("whisper models must not set normalize");
            }
            if model.max_audio_duration_sec == Some(0) {
                bail!("whisper max_audio_duration_sec must be greater than 0");
            }
        }
    }

    Ok(())
}
