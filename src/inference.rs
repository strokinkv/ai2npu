use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use crate::audio::{wav_pcm_s16le_as_f32, AudioEndpoint, AudioOutput};
use crate::bge_embeddings::BgeEmbeddingExecutor;
use crate::config::ModelConfig;
use crate::genai_bridge::{GenAiWhisperBridge, GenAiWhisperSession};

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
        wav_bytes: &[u8],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput>;
    fn preload(&self, model: &ModelConfig) -> Result<()> {
        let wav = silent_wav_100ms();
        self.transcribe(
            model,
            &wav,
            &AudioInferenceOptions {
                endpoint: AudioEndpoint::Transcriptions,
                language: None,
                prompt: None,
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
        wav_bytes: &[u8],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput> {
        let samples = wav_pcm_s16le_as_f32(wav_bytes)?;
        let mut sessions = self
            .sessions
            .lock()
            .expect("native whisper session mutex poisoned");
        if !sessions.contains_key(&model.path) {
            sessions.insert(
                model.path.clone(),
                self.bridge.create_session(&model.path, &self.device)?,
            );
        }
        let session = sessions
            .get(&model.path)
            .expect("native whisper session exists");
        let output = session.transcribe(&samples, options)?;
        let mut loaded = self
            .loaded_models
            .lock()
            .expect("loaded model mutex poisoned");
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
        wav_bytes: &[u8],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput> {
        self.transcribe_native(model, wav_bytes, options)
    }

    fn loaded_models(&self) -> Vec<String> {
        self.loaded_models
            .lock()
            .expect("loaded model mutex poisoned")
            .clone()
    }

    fn unload_all(&self) -> Result<usize> {
        let mut sessions = self
            .sessions
            .lock()
            .expect("native whisper session mutex poisoned");
        let unloaded = sessions.len();
        sessions.clear();
        self.loaded_models
            .lock()
            .expect("loaded model mutex poisoned")
            .clear();
        Ok(unloaded)
    }
}

fn silent_wav_100ms() -> Vec<u8> {
    let data_len = 16_000u32 / 10 * 2;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&16_000u32.to_le_bytes());
    bytes.extend_from_slice(&32_000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    bytes.resize(bytes.len() + data_len as usize, 0);
    bytes
}
