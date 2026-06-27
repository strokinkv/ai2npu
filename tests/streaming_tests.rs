use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ai2npu::audio::{AudioOutput, AudioWord};
use ai2npu::config::{AppConfig, ModelConfig, ModelType};
use ai2npu::http::{build_router_with_executors, build_router_with_streaming_vad_factory};
use ai2npu::inference::{
    audio_executor_from_env, AudioExecutor, AudioInferenceOptions, EmbeddingExecutor,
};
use ai2npu::openvino_backend::OpenVinoStatus;
use ai2npu::queue::InferenceQueue;
use ai2npu::streaming::{
    run_session, ClientEvent, ServerEvent, SessionConfig, SessionGuard, SessionInput, StreamingVad,
};
use ai2npu::vad::{SpeechProb, VadSegmenter, WINDOW_SAMPLES};
use anyhow::{bail, Result};
use axum::Router;
use base64::{engine::general_purpose, Engine as _};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, WebSocketStream};

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
            words: Vec::new(),
        })
    }
}

#[derive(Debug)]
struct NoopEmbeddingExecutor;

impl EmbeddingExecutor for NoopEmbeddingExecutor {
    fn embed(&self, _model: &ModelConfig, input: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(vec![vec![0.0]; input.len()])
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

fn example_config() -> AppConfig {
    let text = std::fs::read_to_string("config.example.toml").unwrap();
    toml::from_str(&text).unwrap()
}

fn status_with_npu() -> OpenVinoStatus {
    OpenVinoStatus {
        runtime_available: true,
        devices: vec!["CPU".to_string(), "GPU".to_string(), "NPU".to_string()],
        npu_available: true,
        error: None,
    }
}

async fn spawn_test_server(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("ws://{addr}/v1/realtime"), server)
}

async fn next_ws_json<S>(socket: &mut WebSocketStream<S>) -> serde_json::Value
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let message = tokio::time::timeout(Duration::from_secs(2), socket.next())
        .await
        .expect("timed out waiting for websocket event")
        .expect("websocket closed")
        .expect("websocket error");
    let Message::Text(text) = message else {
        panic!("expected text websocket frame, got {message:?}");
    };
    serde_json::from_str(&text).unwrap()
}

fn streaming_test_app(
    executor: Arc<ScriptedAudioExecutor>,
    vad_scripts: impl IntoIterator<Item = Vec<f32>>,
) -> Router {
    let scripts = Arc::new(Mutex::new(vad_scripts.into_iter().collect::<VecDeque<_>>()));
    let vad_factory = {
        let scripts = Arc::clone(&scripts);
        Arc::new(move || -> Result<Box<dyn StreamingVad>> {
            let probs = scripts.lock().unwrap().pop_front().unwrap_or_default();
            let vad = VadSegmenter::with_probability_source(
                ScriptedSpeechProb::new(probs),
                64,
                0.5,
                30_000,
            )?;
            Ok(Box::new(vad))
        })
    };

    build_router_with_streaming_vad_factory(
        example_config(),
        status_with_npu(),
        Arc::new(NoopEmbeddingExecutor),
        executor,
        vad_factory,
    )
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
        vad: Box::new(vad),
        cancel: Arc::new(AtomicBool::new(false)),
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
                content_index: 0,
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
                content_index: 0,
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

#[tokio::test]
async fn run_session_emits_deltas_then_final() {
    // speech(2) -> micro-pause(1) -> speech(2) -> endpoint(2).
    let mut vad = VadSegmenter::with_probability_source(
        ScriptedSpeechProb::new([0.9, 0.9, 0.2, 0.9, 0.9, 0.0, 0.0]),
        64,
        0.5,
        30_000,
    )
    .unwrap();
    vad.set_partial_silence_ms(32);
    let cfg = SessionConfig {
        session_id: 7,
        input_sample_rate: 16_000,
        input_channels: 1,
        max_input_buffer_sec: 30,
        language: None,
        prompt: None,
        word_timestamps: false,
        vad: Box::new(vad),
        cancel: Arc::new(AtomicBool::new(false)),
    };
    let executor = Arc::new(ScriptedAudioExecutor::new([
        "Запиши",
        "Запиши",
        "Запиши сообщение",
    ]));
    let (input_tx, input_rx) = tokio::sync::mpsc::channel(8);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(32);

    let session = tokio::spawn(run_session(
        cfg,
        input_rx,
        event_tx,
        executor.clone(),
        InferenceQueue::new(4),
        whisper_model(),
    ));

    input_tx
        .send(SessionInput::Audio(
            general_purpose::STANDARD
                .decode(pcm16_base64_for_windows(7))
                .unwrap(),
        ))
        .await
        .unwrap();
    drop(input_tx);
    session.await.unwrap().unwrap();

    let mut events = Vec::new();
    while let Some(event) = event_rx.recv().await {
        events.push(event);
    }

    let deltas = events
        .iter()
        .filter_map(|e| match e {
            ServerEvent::InputAudioTranscriptionDelta {
                item_id,
                content_index,
                delta,
            } => Some((item_id.clone(), *content_index, delta.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        deltas,
        vec![("item_0".to_string(), 0, "Запиши".to_string())]
    );

    let final_pos = events
        .iter()
        .position(|e| matches!(e, ServerEvent::InputAudioTranscriptionCompleted { .. }))
        .expect("expected a completed event");
    let delta_pos = events
        .iter()
        .position(|e| matches!(e, ServerEvent::InputAudioTranscriptionDelta { .. }))
        .unwrap();
    assert!(
        delta_pos < final_pos,
        "delta must precede completed: {events:?}"
    );

    match &events[final_pos] {
        ServerEvent::InputAudioTranscriptionCompleted { transcript, .. } => {
            assert_eq!(transcript, "Запиши сообщение");
        }
        other => panic!("expected completed, got {other:?}"),
    }
}

#[tokio::test]
async fn ws_roundtrip_streams_realtime_events_and_rejects_busy_session() {
    let executor = Arc::new(ScriptedAudioExecutor::new(["Запиши", "сообщение"]));
    let app = streaming_test_app(
        Arc::clone(&executor),
        [vec![0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0]],
    );
    let (url, server) = spawn_test_server(app).await;

    let (mut socket, _) = connect_async(&url).await.unwrap();
    assert_eq!(
        next_ws_json(&mut socket).await,
        json!({
            "type": "transcription_session.created",
            "session_id": 1
        })
    );

    let (mut busy_socket, _) = connect_async(&url).await.unwrap();
    let busy = next_ws_json(&mut busy_socket).await;
    assert_eq!(busy["type"], "error");
    assert_eq!(busy["error"]["code"], "streaming_busy");
    let _ = busy_socket.close(None).await;

    socket
        .send(Message::Text(
            json!({
                "type": "transcription_session.update",
                "session": {
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
                }
            })
            .to_string(),
        ))
        .await
        .unwrap();
    socket
        .send(Message::Text(
            json!({
                "type": "input_audio_buffer.append",
                "audio": pcm16_base64_for_windows(8)
            })
            .to_string(),
        ))
        .await
        .unwrap();

    let mut events = Vec::new();
    for _ in 0..9 {
        events.push(next_ws_json(&mut socket).await);
    }

    assert_eq!(
        events,
        vec![
            json!({"type": "transcription_session.updated"}),
            json!({
                "type": "input_audio_buffer.speech_started",
                "audio_start_ms": 0,
                "item_id": "item_0"
            }),
            json!({
                "type": "input_audio_buffer.speech_stopped",
                "audio_end_ms": 64,
                "item_id": "item_0"
            }),
            json!({
                "type": "input_audio_buffer.committed",
                "item_id": "item_0"
            }),
            json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_0",
                "content_index": 0,
                "transcript": "Запиши"
            }),
            json!({
                "type": "input_audio_buffer.speech_started",
                "audio_start_ms": 128,
                "item_id": "item_1"
            }),
            json!({
                "type": "input_audio_buffer.speech_stopped",
                "audio_end_ms": 192,
                "item_id": "item_1"
            }),
            json!({
                "type": "input_audio_buffer.committed",
                "item_id": "item_1"
            }),
            json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "content_index": 0,
                "transcript": "сообщение"
            }),
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

    let _ = socket.close(None).await;
    server.abort();
}

#[derive(Debug)]
struct BlockingAudioExecutor {
    calls: Arc<AtomicUsize>,
    entered: Mutex<Option<std::sync::mpsc::Sender<()>>>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl AudioExecutor for BlockingAudioExecutor {
    fn transcribe(
        &self,
        _model: &ModelConfig,
        _samples: &[f32],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            if let Some(tx) = self.entered.lock().unwrap().take() {
                let _ = tx.send(());
            }
            // Block until the test releases the first (in-flight) decode.
            let _ = self.release.lock().unwrap().recv();
        }
        Ok(AudioOutput {
            text: format!("segment {n}"),
            language: options.language.clone(),
            duration: 0.0,
            segments: Vec::new(),
            words: Vec::new(),
        })
    }
}

/// Task 1.7: when the cancel flag is set while a decode is in-flight, the
/// orchestrator must drain the current decode but NOT start the remaining
/// buffered segments (cooperative cancellation, not kill).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_session_stops_decoding_after_cancel() {
    // 8 windows -> two speech segments.
    let vad = VadSegmenter::with_probability_source(
        ScriptedSpeechProb::new([0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0]),
        64,
        0.5,
        30_000,
    )
    .unwrap();

    let cancel = Arc::new(AtomicBool::new(false));
    let cfg = SessionConfig {
        session_id: 7,
        input_sample_rate: 16_000,
        input_channels: 1,
        max_input_buffer_sec: 30,
        language: None,
        prompt: None,
        word_timestamps: false,
        vad: Box::new(vad),
        cancel: Arc::clone(&cancel),
    };

    let calls = Arc::new(AtomicUsize::new(0));
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let executor = Arc::new(BlockingAudioExecutor {
        calls: Arc::clone(&calls),
        entered: Mutex::new(Some(entered_tx)),
        release: Mutex::new(release_rx),
    });

    let (input_tx, input_rx) = tokio::sync::mpsc::channel(8);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(32);

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
            ClientEvent::InputAudioBufferAppend {
                audio: pcm16_base64_for_windows(8),
            },
        ))
        .await
        .unwrap();

    // Wait until the first segment decode is in-flight (blocked).
    tokio::task::spawn_blocking(move || entered_rx.recv())
        .await
        .unwrap()
        .unwrap();

    // Disconnect: signal cooperative cancel, then release the in-flight decode.
    cancel.store(true, Ordering::SeqCst);
    release_tx.send(()).unwrap();
    drop(input_tx);

    session.await.unwrap().unwrap();

    // Only the first (already in-flight) segment was decoded; the second was skipped.
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let mut completed = 0;
    while let Some(event) = event_rx.recv().await {
        if matches!(event, ServerEvent::InputAudioTranscriptionCompleted { .. }) {
            completed += 1;
        }
    }
    assert_eq!(completed, 1, "second buffered segment must not be decoded");
}

#[derive(Debug)]
struct WordAudioExecutor {
    text: String,
    words: Vec<AudioWord>,
}

impl AudioExecutor for WordAudioExecutor {
    fn transcribe(
        &self,
        _model: &ModelConfig,
        _samples: &[f32],
        options: &AudioInferenceOptions,
    ) -> Result<AudioOutput> {
        Ok(AudioOutput {
            text: self.text.clone(),
            language: options.language.clone(),
            duration: 0.0,
            segments: Vec::new(),
            words: self.words.clone(),
        })
    }
}

fn single_segment_session(word_timestamps: bool, cancel: Arc<AtomicBool>) -> SessionConfig {
    let vad = VadSegmenter::with_probability_source(
        ScriptedSpeechProb::new([0.9, 0.9, 0.0, 0.0]),
        64,
        0.5,
        30_000,
    )
    .unwrap();
    SessionConfig {
        session_id: 9,
        input_sample_rate: 16_000,
        input_channels: 1,
        max_input_buffer_sec: 30,
        language: None,
        prompt: None,
        word_timestamps,
        vad: Box::new(vad),
        cancel,
    }
}

async fn run_single_segment(
    cfg: SessionConfig,
    executor: Arc<WordAudioExecutor>,
) -> Vec<ServerEvent> {
    let (input_tx, input_rx) = tokio::sync::mpsc::channel(8);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(16);
    let session = tokio::spawn(run_session(
        cfg,
        input_rx,
        event_tx,
        executor,
        InferenceQueue::new(4),
        whisper_model(),
    ));
    input_tx
        .send(SessionInput::ClientEvent(
            ClientEvent::InputAudioBufferAppend {
                audio: pcm16_base64_for_windows(4),
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
    events
}

/// Task 1.8: with `word_timestamps` enabled and an executor that returns word
/// spans, the `completed` event carries per-word timestamps (ms).
#[tokio::test]
async fn run_session_emits_word_timestamps_when_enabled() {
    let executor = Arc::new(WordAudioExecutor {
        text: "Привет мир".to_string(),
        words: vec![
            AudioWord {
                word: "Привет".to_string(),
                start: 0.0,
                end: 0.5,
            },
            AudioWord {
                word: "мир".to_string(),
                start: 0.5,
                end: 1.0,
            },
        ],
    });
    let cfg = single_segment_session(true, Arc::new(AtomicBool::new(false)));
    let events = run_single_segment(cfg, executor).await;

    let completed = events
        .into_iter()
        .find_map(|event| match event {
            ServerEvent::InputAudioTranscriptionCompleted { words, .. } => Some(words),
            _ => None,
        })
        .expect("missing completed event");
    let words = completed.expect("words should be present when word_timestamps is enabled");
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text, "Привет");
    assert_eq!(words[0].start_ms, 0);
    assert_eq!(words[0].end_ms, 500);
    assert_eq!(words[1].text, "мир");
    assert_eq!(words[1].start_ms, 500);
    assert_eq!(words[1].end_ms, 1000);
}

/// Task 1.8: word timestamps are omitted by default (drop-in OpenAI Realtime).
#[tokio::test]
async fn run_session_omits_word_timestamps_by_default() {
    let executor = Arc::new(WordAudioExecutor {
        text: "Привет".to_string(),
        words: vec![AudioWord {
            word: "Привет".to_string(),
            start: 0.0,
            end: 0.5,
        }],
    });
    let cfg = single_segment_session(false, Arc::new(AtomicBool::new(false)));
    let events = run_single_segment(cfg, executor).await;

    let completed = events
        .into_iter()
        .find_map(|event| match event {
            ServerEvent::InputAudioTranscriptionCompleted { words, .. } => Some(words),
            _ => None,
        })
        .expect("missing completed event");
    assert!(
        completed.is_none(),
        "words must be absent when word_timestamps is disabled"
    );
}

/// Minimal PCM16 mono WAV reader for the live smoke test: returns the sample
/// rate and interleaved i16 samples (first channel only if multi-channel).
fn read_wav_pcm16(path: &std::path::Path) -> (u32, Vec<i16>) {
    let bytes = std::fs::read(path).expect("read smoke WAV");
    assert_eq!(&bytes[0..4], b"RIFF", "not a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "not a WAVE file");

    let mut sample_rate = 0u32;
    let mut channels = 1u16;
    let mut data: Vec<i16> = Vec::new();
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap());
            let bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap());
            assert_eq!(bits, 16, "smoke WAV must be 16-bit PCM");
        } else if id == b"data" {
            let end = (body + size).min(bytes.len());
            let step = channels as usize;
            let mut i = body;
            while i + 2 <= end {
                data.push(i16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()));
                i += 2 * step; // keep first channel only
            }
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    assert!(sample_rate > 0 && !data.is_empty(), "empty/invalid WAV");
    (sample_rate, data)
}

/// Task 1.10 (step 2): live NPU end-to-end streaming smoke. Streams a real WAV
/// over the production WebSocket route (real Whisper NPU executor + real Silero
/// VAD) and verifies ordered, non-empty `...completed` events. Gated behind
/// `AI2NPU_RUN_NPU_TESTS=1` AND a `AI2NPU_SMOKE_WAV` path to a 16-bit PCM WAV.
/// Requires the `ai2npuService` stopped (NPU is single-context) and the
/// ort-compatible onnxruntime.dll on the search path (`ORT_DYLIB_PATH`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_npu_streaming_transcription_smoke() {
    if std::env::var("AI2NPU_RUN_NPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping live NPU streaming smoke; set AI2NPU_RUN_NPU_TESTS=1");
        return;
    }
    let Ok(wav_path) = std::env::var("AI2NPU_SMOKE_WAV") else {
        eprintln!(
            "skipping live NPU streaming smoke; set AI2NPU_SMOKE_WAV=<path to 16-bit PCM WAV>"
        );
        return;
    };
    let language = std::env::var("AI2NPU_SMOKE_LANG").unwrap_or_else(|_| "en".to_string());

    let (sample_rate, samples) = read_wav_pcm16(std::path::Path::new(&wav_path));
    eprintln!(
        "smoke WAV: {} samples @ {} Hz ({:.2}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate as f64
    );

    let app = build_router_with_executors(
        example_config(),
        OpenVinoStatus::detect(),
        Arc::new(NoopEmbeddingExecutor),
        audio_executor_from_env().expect("native whisper executor"),
    );
    let (url, server) = spawn_test_server(app).await;
    let (mut socket, _) = connect_async(&url).await.unwrap();

    // created
    let created = next_ws_json(&mut socket).await;
    assert_eq!(created["type"], "transcription_session.created");

    socket
        .send(Message::Text(
            json!({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "openai/whisper-large-v3-turbo",
                        "language": language,
                    },
                    "turn_detection": { "type": "server_vad", "threshold": 0.5 },
                    "sample_rate": sample_rate,
                    "word_timestamps": false,
                }
            })
            .to_string(),
        ))
        .await
        .unwrap();

    // Stream ~40ms PCM16 chunks to mimic a real microphone cadence.
    let chunk_samples = (sample_rate as usize / 25).max(1);
    for chunk in samples.chunks(chunk_samples) {
        let mut pcm = Vec::with_capacity(chunk.len() * 2);
        for s in chunk {
            pcm.extend_from_slice(&s.to_le_bytes());
        }
        socket
            .send(Message::Text(
                json!({
                    "type": "input_audio_buffer.append",
                    "audio": general_purpose::STANDARD.encode(&pcm),
                })
                .to_string(),
            ))
            .await
            .unwrap();
    }
    // Force-close the trailing phrase so we don't depend on socket-close flush.
    socket
        .send(Message::Text(
            json!({ "type": "input_audio_buffer.commit" }).to_string(),
        ))
        .await
        .unwrap();

    // Collect events until the first completed lands (NPU cold start can be slow).
    let mut transcripts: Vec<(u64, String)> = Vec::new();
    let mut delta_count = 0usize;
    let mut last_speech_stopped = std::time::Instant::now();
    let mut latency: Option<Duration> = None;
    let deadline = std::time::Instant::now() + Duration::from_secs(300);
    while std::time::Instant::now() < deadline {
        // Wait long for the first final (NPU cold start compiles the Whisper
        // model on first decode); once we have one, a short read timeout means
        // the stream has drained.
        let read_timeout = if transcripts.is_empty() {
            Duration::from_secs(180)
        } else {
            // Allow a trailing phrase still decoding on the NPU to drain so the
            // ordering assertion sees more than one item. Overridable via
            // AI2NPU_SMOKE_DRAIN_SECS for slow tail segments.
            let secs = std::env::var("AI2NPU_SMOKE_DRAIN_SECS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(15);
            Duration::from_secs(secs)
        };
        let next = tokio::time::timeout(read_timeout, socket.next()).await;
        let frame = match next {
            Ok(Some(Ok(Message::Text(text)))) => text,
            Ok(Some(Ok(other))) => {
                eprintln!("non-text frame: {other:?}");
                continue;
            }
            Ok(Some(Err(e))) => {
                eprintln!("websocket error: {e}");
                break;
            }
            Ok(None) => {
                eprintln!("websocket closed by server");
                break;
            }
            Err(_) => {
                eprintln!("read timed out after {read_timeout:?}");
                break;
            }
        };
        let event: serde_json::Value = serde_json::from_str(&frame).unwrap();
        eprintln!("event: {}", event["type"].as_str().unwrap_or("?"));
        match event["type"].as_str() {
            Some("input_audio_buffer.speech_stopped") => {
                last_speech_stopped = std::time::Instant::now();
            }
            Some("conversation.item.input_audio_transcription.completed") => {
                if latency.is_none() {
                    latency = Some(last_speech_stopped.elapsed());
                }
                let item_id = event["item_id"].as_str().unwrap_or("");
                let n = item_id
                    .rsplit('_')
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(transcripts.len() as u64);
                let transcript = event["transcript"].as_str().unwrap_or("").to_string();
                eprintln!("completed {item_id}: {transcript:?}");
                transcripts.push((n, transcript));
            }
            Some("conversation.item.input_audio_transcription.delta") => {
                delta_count += 1;
                eprintln!(
                    "delta {}: {:?}",
                    event["item_id"].as_str().unwrap_or(""),
                    event["delta"].as_str().unwrap_or("")
                );
            }
            Some("error") => {
                eprintln!("server error event: {event}");
                break;
            }
            _ => {}
        }
    }

    server.abort();

    let ids: Vec<u64> = transcripts.iter().map(|(n, _)| *n).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    let mut failures: Vec<String> = Vec::new();
    if transcripts.is_empty() {
        failures.push("expected at least one completed transcription event".into());
    }
    if !transcripts.iter().any(|(_, t)| !t.trim().is_empty()) {
        failures.push(format!("all transcripts were empty: {transcripts:?}"));
    }
    if ids != sorted {
        failures.push(format!(
            "completed events must be ordered by item id: {ids:?}"
        ));
    }

    if failures.is_empty() {
        eprintln!(
            "live NPU streaming smoke OK: {} phrase(s), {} delta(s), first-phrase latency {:?}",
            transcripts.len(),
            delta_count,
            latency
        );
    } else {
        for failure in &failures {
            eprintln!("live NPU streaming smoke FAILED: {failure}");
        }
    }

    // Unloading the OpenVINO NPU runtime + ort deadlocks at process teardown
    // (CPUStreamsExecutor destructors wait forever). Since the smoke has already
    // produced its verdict, exit the process explicitly instead of returning
    // through the hanging native destructors. This path only runs in the gated
    // NPU mode; a normal `cargo test` returns early above and never reaches here.
    use std::io::Write as _;
    let _ = std::io::stderr().flush();
    std::process::exit(if failures.is_empty() { 0 } else { 1 });
}
