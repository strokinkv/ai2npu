use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ai2npu::audio::AudioOutput;
use ai2npu::config::{ModelConfig, ModelType};
use ai2npu::inference::{AudioExecutor, AudioInferenceOptions};
use ai2npu::queue::InferenceQueue;
use ai2npu::streaming::{
    run_session, ClientEvent, ServerEvent, SessionConfig, SessionGuard, SessionInput,
};
use ai2npu::vad::{SpeechProb, VadSegmenter, WINDOW_SAMPLES};
use anyhow::{bail, Result};
use base64::{engine::general_purpose, Engine as _};
use serde_json::json;

#[test]
fn deserializes_realtime_transcription_session_update() {
    let event: ClientEvent = serde_json::from_value(json!({
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "openai/whisper-large-v3-turbo",
                "language": "ru",
                "prompt": "Запиши Открой Telegram"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 400
            },
            "sample_rate": 48000,
            "max_segment_ms": 30000,
            "word_timestamps": true
        }
    }))
    .unwrap();

    let ClientEvent::TranscriptionSessionUpdate { session } = event else {
        panic!("expected transcription_session.update");
    };
    let transcription = session.input_audio_transcription.unwrap();
    let turn_detection = session.turn_detection.unwrap();

    assert_eq!(session.input_audio_format.as_deref(), Some("pcm16"));
    assert_eq!(transcription.model, "openai/whisper-large-v3-turbo");
    assert_eq!(transcription.language.as_deref(), Some("ru"));
    assert_eq!(
        transcription.prompt.as_deref(),
        Some("Запиши Открой Telegram")
    );
    assert_eq!(turn_detection.kind, "server_vad");
    assert_eq!(turn_detection.threshold, Some(0.5));
    assert_eq!(turn_detection.silence_duration_ms, Some(400));
    assert_eq!(session.sample_rate, Some(48000));
    assert_eq!(session.max_segment_ms, Some(30000));
    assert_eq!(session.word_timestamps, Some(true));
}

#[test]
fn deserializes_audio_append_event() {
    let event: ClientEvent = serde_json::from_value(json!({
        "type": "input_audio_buffer.append",
        "audio": "AQIDBA=="
    }))
    .unwrap();

    let ClientEvent::InputAudioBufferAppend { audio } = event else {
        panic!("expected input_audio_buffer.append");
    };

    assert_eq!(audio, "AQIDBA==");
}

#[test]
fn session_guard_allows_only_one_active_session() {
    let active_session = Arc::new(Mutex::new(None));
    let first = SessionGuard::try_acquire(Arc::clone(&active_session), 1).unwrap();

    assert_eq!(*active_session.lock().unwrap(), Some(1));

    let err = SessionGuard::try_acquire(Arc::clone(&active_session), 2)
        .expect_err("second session should be rejected");
    assert_eq!(err.code(), "streaming_busy");

    let error_event = err.into_server_event();
    let json = serde_json::to_value(error_event).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["error"]["code"], "streaming_busy");

    drop(first);
    assert_eq!(*active_session.lock().unwrap(), None);

    let second = SessionGuard::try_acquire(Arc::clone(&active_session), 2).unwrap();
    assert_eq!(*active_session.lock().unwrap(), Some(2));
    drop(second);
    assert_eq!(*active_session.lock().unwrap(), None);
}

#[derive(Debug)]
struct ScriptedSpeechProb {
    probs: VecDeque<f32>,
}

impl ScriptedSpeechProb {
    fn new(probs: impl IntoIterator<Item = f32>) -> Self {
        Self {
            probs: probs.into_iter().collect(),
        }
    }
}

impl SpeechProb for ScriptedSpeechProb {
    fn prob(&mut self, _window: &[f32]) -> f32 {
        self.probs.pop_front().unwrap_or(0.0)
    }
}

#[derive(Debug)]
struct ScriptedAudioExecutor {
    outputs: Mutex<VecDeque<String>>,
    prompts: Mutex<Vec<Option<String>>>,
}

impl ScriptedAudioExecutor {
    fn new(outputs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            outputs: Mutex::new(outputs.into_iter().map(Into::into).collect()),
            prompts: Mutex::new(Vec::new()),
        }
    }

    fn prompts(&self) -> Vec<Option<String>> {
        self.prompts.lock().unwrap().clone()
    }
}

impl AudioExecutor for ScriptedAudioExecutor {
    fn transcribe(
        &self,
        _model: &ModelConfig,
        _samples: &[f32],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput> {
        self.prompts.lock().unwrap().push(options.prompt.clone());
        let Some(text) = self.outputs.lock().unwrap().pop_front() else {
            bail!("no scripted output left");
        };
        Ok(AudioOutput {
            text,
            language: options.language.clone(),
            duration: 0.0,
            segments: Vec::new(),
        })
    }
}

fn pcm16_base64_for_windows(windows: usize) -> String {
    let samples = windows * WINDOW_SAMPLES;
    let mut pcm = Vec::with_capacity(samples * 2);
    for _ in 0..samples {
        pcm.extend_from_slice(&1_i16.to_le_bytes());
    }
    general_purpose::STANDARD.encode(pcm)
}

fn whisper_model() -> ModelConfig {
    ModelConfig {
        id: "openai/whisper-large-v3-turbo".to_string(),
        model_type: ModelType::Whisper,
        path: PathBuf::from("models/whisper"),
        enabled: true,
        preload: false,
        queue_timeout_sec: 30,
        normalize: None,
        max_audio_duration_sec: Some(30),
    }
}

#[tokio::test]
async fn run_session_emits_ordered_finals() {
    let vad = VadSegmenter::with_probability_source(
        ScriptedSpeechProb::new([0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0]),
        64,
        0.5,
        30_000,
    )
    .unwrap();
    let cfg = SessionConfig {
        session_id: 42,
        input_sample_rate: 16_000,
        input_channels: 1,
        max_input_buffer_sec: 30,
        language: None,
        prompt: None,
        word_timestamps: false,
        vad,
    };
    let executor = Arc::new(ScriptedAudioExecutor::new(["Запиши", "сообщение"]));
    let (input_tx, input_rx) = tokio::sync::mpsc::channel(8);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(16);

    let session = tokio::spawn(run_session(
        cfg,
        input_rx,
        event_tx,
        executor.clone(),
        InferenceQueue::new(4),
        whisper_model(),
    ));

    input_tx
        .send(SessionInput::ClientEvent(
            ClientEvent::TranscriptionSessionUpdate {
                session: serde_json::from_value(json!({
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "openai/whisper-large-v3-turbo",
                        "language": "ru",
                        "prompt": "Команды: Запиши"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 64
                    },
                    "sample_rate": 16000,
                    "word_timestamps": false
                }))
                .unwrap(),
            },
        ))
        .await
        .unwrap();
    input_tx
        .send(SessionInput::ClientEvent(
            ClientEvent::InputAudioBufferAppend {
                audio: pcm16_base64_for_windows(8),
            },
        ))
        .await
        .unwrap();
    drop(input_tx);

    session.await.unwrap().unwrap();

    let mut events = Vec::new();
    while let Some(event) = event_rx.recv().await {
        events.push(event);
    }

    assert_eq!(
        events,
        vec![
            ServerEvent::TranscriptionSessionCreated { session_id: 42 },
            ServerEvent::TranscriptionSessionUpdated,
            ServerEvent::InputAudioBufferSpeechStarted {
                audio_start_ms: 0,
                item_id: "item_0".to_string(),
            },
            ServerEvent::InputAudioBufferSpeechStopped {
                audio_end_ms: 64,
                item_id: "item_0".to_string(),
            },
            ServerEvent::InputAudioBufferCommitted {
                item_id: "item_0".to_string(),
            },
            ServerEvent::InputAudioTranscriptionCompleted {
                item_id: "item_0".to_string(),
                transcript: "Запиши".to_string(),
                words: None,
            },
            ServerEvent::InputAudioBufferSpeechStarted {
                audio_start_ms: 128,
                item_id: "item_1".to_string(),
            },
            ServerEvent::InputAudioBufferSpeechStopped {
                audio_end_ms: 192,
                item_id: "item_1".to_string(),
            },
            ServerEvent::InputAudioBufferCommitted {
                item_id: "item_1".to_string(),
            },
            ServerEvent::InputAudioTranscriptionCompleted {
                item_id: "item_1".to_string(),
                transcript: "сообщение".to_string(),
                words: None,
            },
        ]
    );

    let prompts = executor.prompts();
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0].as_deref(), Some("Команды: Запиши"));
    assert!(
        prompts[1]
            .as_deref()
            .is_some_and(|prompt| prompt.contains("Команды: Запиши") && prompt.contains("Запиши")),
        "second prompt should include session prompt and previous transcript, got {prompts:?}"
    );
}
