use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ClientEvent {
    #[serde(rename = "transcription_session.update")]
    TranscriptionSessionUpdate { session: TranscriptionSessionConfig },
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend { audio: String },
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TranscriptionSessionConfig {
    #[serde(default)]
    pub input_audio_format: Option<String>,
    #[serde(default)]
    pub input_audio_transcription: Option<InputAudioTranscriptionConfig>,
    #[serde(default)]
    pub turn_detection: Option<TurnDetectionConfig>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
    #[serde(default)]
    pub max_segment_ms: Option<u64>,
    #[serde(default)]
    pub word_timestamps: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InputAudioTranscriptionConfig {
    pub model: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TurnDetectionConfig {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub threshold: Option<f32>,
    #[serde(default)]
    pub silence_duration_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ServerEvent {
    #[serde(rename = "transcription_session.created")]
    TranscriptionSessionCreated { session_id: u64 },
    #[serde(rename = "transcription_session.updated")]
    TranscriptionSessionUpdated,
    #[serde(rename = "input_audio_buffer.speech_started")]
    InputAudioBufferSpeechStarted {
        audio_start_ms: u64,
        item_id: String,
    },
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    InputAudioBufferSpeechStopped { audio_end_ms: u64, item_id: String },
    #[serde(rename = "input_audio_buffer.committed")]
    InputAudioBufferCommitted { item_id: String },
    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    InputAudioTranscriptionDelta { item_id: String, delta: String },
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    InputAudioTranscriptionCompleted {
        item_id: String,
        transcript: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        words: Option<Vec<TranscriptionWord>>,
    },
    #[serde(rename = "error")]
    Error { error: StreamingErrorBody },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TranscriptionWord {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StreamingErrorBody {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingError {
    code: &'static str,
    message: String,
}

impl StreamingError {
    fn streaming_busy() -> Self {
        Self {
            code: "streaming_busy",
            message: "another streaming session is already active".to_string(),
        }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn into_server_event(self) -> ServerEvent {
        ServerEvent::Error {
            error: StreamingErrorBody {
                code: self.code.to_string(),
                message: self.message,
            },
        }
    }
}

#[derive(Debug)]
pub struct SessionGuard {
    active_session: Arc<Mutex<Option<u64>>>,
    session_id: u64,
}

impl SessionGuard {
    pub fn try_acquire(
        active_session: Arc<Mutex<Option<u64>>>,
        session_id: u64,
    ) -> Result<Self, StreamingError> {
        {
            let mut active = active_session
                .lock()
                .map_err(|_| StreamingError::streaming_busy())?;
            if active.is_some() {
                return Err(StreamingError::streaming_busy());
            }
            *active = Some(session_id);
        }

        Ok(Self {
            active_session,
            session_id,
        })
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        if let Ok(mut active) = self.active_session.lock() {
            if active.as_ref() == Some(&self.session_id) {
                *active = None;
            }
        }
    }
}
