use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, Result};
use thiserror::Error;

use crate::audio::{AudioEndpoint, AudioOutput};
use crate::bge_embeddings::BgeEmbeddingExecutor;
use crate::config::ModelConfig;
use crate::genai_bridge::{GenAiWhisperBridge, GenAiWhisperSession};

/// Wraps an error that occurred while loading or compiling a model so the HTTP
/// layer can map it to the `model_load_failed` API error code.
#[derive(Debug, Error)]
#[error("{0:#}")]
pub struct ModelLoadFailed(pub anyhow::Error);

/// Wraps an error that occurred while running inference so the HTTP layer can
/// map it to the `inference_failed` API error code.
#[derive(Debug, Error)]
#[error("{0:#}")]
pub struct InferenceFailed(pub anyhow::Error);

/// Recover the inner guard from a poisoned mutex instead of panicking, so a
/// single panicked inference job cannot permanently break the service.
pub(crate) fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub trait EmbeddingExecutor: Send + Sync {
    fn embed(&self, model: &ModelConfig, input: &[String]) -> Result<Vec<Vec<f32>>>;
    fn preload(&self, model: &ModelConfig) -> Result<()> {
        let _ = model;
        Ok(())
    }
    fn loaded_models(&self) -> Vec<String> {
        Vec::new()
    }
    fn unload_all(&self) -> Result<usize> {
        Ok(0)
    }
}

pub trait AudioExecutor: Send + Sync {
    fn transcribe(
        &self,
        model: &ModelConfig,
        samples: &[f32],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput>;
    fn preload(&self, model: &ModelConfig) -> Result<()> {
        let samples = vec![0.0_f32; 1_600];
        self.transcribe(
            model,
            &samples,
            &AudioInferenceOptions {
                endpoint: AudioEndpoint::Transcriptions,
                language: None,
                prompt: None,
                temperature: None,
                return_timestamps: false,
            },
        )
        .map(|_| ())
    }
    fn loaded_models(&self) -> Vec<String> {
        Vec::new()
    }
    fn unload_all(&self) -> Result<usize> {
        Ok(0)
    }
}

#[derive(Debug, Clone)]
pub struct AudioInferenceOptions {
    pub endpoint: AudioEndpoint,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub temperature: Option<f32>,
    pub return_timestamps: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioExecutorKind {
    NativeGenAi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingExecutorKind {
    RustOpenVino,
}

pub struct EmbeddingExecutorSelection {
    kind: EmbeddingExecutorKind,
    executor: Arc<dyn EmbeddingExecutor>,
}

impl EmbeddingExecutorSelection {
    pub fn kind(&self) -> EmbeddingExecutorKind {
        self.kind
    }

    pub fn into_executor(self) -> Arc<dyn EmbeddingExecutor> {
        self.executor
    }
}

pub fn embedding_executor_kind_from_env() -> Result<EmbeddingExecutorKind> {
    match std::env::var("AI2NPU_EMBEDDINGS_EXECUTOR") {
        Ok(value)
            if value.eq_ignore_ascii_case("rust") || value.eq_ignore_ascii_case("openvino") =>
        {
            Ok(EmbeddingExecutorKind::RustOpenVino)
        }
        Ok(value) => anyhow::bail!(
            "AI2NPU_EMBEDDINGS_EXECUTOR must be unset, rust, or openvino; got {value}"
        ),
        Err(std::env::VarError::NotPresent) => Ok(EmbeddingExecutorKind::RustOpenVino),
        Err(error) => Err(error).context("failed to read AI2NPU_EMBEDDINGS_EXECUTOR"),
    }
}

pub fn embedding_executor_from_env() -> Result<EmbeddingExecutorSelection> {
    let kind = embedding_executor_kind_from_env()?;
    let executor: Arc<dyn EmbeddingExecutor> = match kind {
        EmbeddingExecutorKind::RustOpenVino => Arc::new(BgeEmbeddingExecutor::new()?),
    };

    Ok(EmbeddingExecutorSelection { kind, executor })
}

pub fn audio_executor_kind_from_env() -> Result<AudioExecutorKind> {
    match std::env::var("AI2NPU_AUDIO_EXECUTOR") {
        Ok(value)
            if value.eq_ignore_ascii_case("native") || value.eq_ignore_ascii_case("genai") =>
        {
            Ok(AudioExecutorKind::NativeGenAi)
        }
        Ok(value) => {
            anyhow::bail!("AI2NPU_AUDIO_EXECUTOR must be unset, native, or genai; got {value}")
        }
        Err(std::env::VarError::NotPresent) => Ok(AudioExecutorKind::NativeGenAi),
        Err(error) => Err(error).context("failed to read AI2NPU_AUDIO_EXECUTOR"),
    }
}

pub fn audio_executor_from_env() -> Result<Arc<dyn AudioExecutor>> {
    match audio_executor_kind_from_env()? {
        AudioExecutorKind::NativeGenAi => Ok(Arc::new(NativeWhisperExecutor::new()?)),
    }
}

pub struct NativeWhisperExecutor {
    bridge: Arc<GenAiWhisperBridge>,
    device: String,
    sessions: Mutex<HashMap<PathBuf, GenAiWhisperSession>>,
    loaded_models: Mutex<Vec<String>>,
}

impl NativeWhisperExecutor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            bridge: GenAiWhisperBridge::load_default()?,
            device: "NPU".to_string(),
            sessions: Mutex::new(HashMap::new()),
            loaded_models: Mutex::new(Vec::new()),
        })
    }

    fn transcribe_native(
        &self,
        model: &ModelConfig,
        samples: &[f32],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput> {
        let mut sessions = lock_recover(&self.sessions);
        if !sessions.contains_key(&model.path) {
            let session = self
                .bridge
                .create_session(&model.path, &self.device)
                .map_err(ModelLoadFailed)?;
            sessions.insert(model.path.clone(), session);
        }
        let session = sessions
            .get(&model.path)
            .expect("native whisper session exists");
        let output = session
            .transcribe(samples, options)
            .map_err(InferenceFailed)?;
        let mut loaded = lock_recover(&self.loaded_models);
        if !loaded.iter().any(|id| id == &model.id) {
            loaded.push(model.id.clone());
        }
        Ok(output)
    }
}

impl AudioExecutor for NativeWhisperExecutor {
    fn transcribe(
        &self,
        model: &ModelConfig,
        samples: &[f32],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput> {
        self.transcribe_native(model, samples, options)
    }

    fn loaded_models(&self) -> Vec<String> {
        lock_recover(&self.loaded_models).clone()
    }

    fn unload_all(&self) -> Result<usize> {
        let mut sessions = lock_recover(&self.sessions);
        let unloaded = sessions.len();
        sessions.clear();
        lock_recover(&self.loaded_models).clear();
        Ok(unloaded)
    }
}
