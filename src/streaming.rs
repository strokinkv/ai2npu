use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::audio::{AudioEndpoint, AudioOutput};
use crate::config::ModelConfig;
use crate::inference::{AudioExecutor, AudioInferenceOptions};
use crate::queue::{InferenceJob, InferenceOutput, InferenceQueue};
use crate::resample::resample_to_16k_mono;
use crate::vad::{SpeechProb, VadEvent, VadSegmenter};

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

#[derive(Debug, Clone, PartialEq)]
pub enum ClientControl {
    Update(TranscriptionSessionConfig),
    Commit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionInput {
    ClientEvent(ClientEvent),
    Audio(Vec<u8>),
    Control(ClientControl),
}

pub trait StreamingVad: Send {
    fn push(&mut self, samples_16k: &[f32]) -> Vec<VadEvent>;
    fn flush(&mut self) -> Option<VadEvent>;
    fn set_min_silence_ms(&mut self, ms: u64);
}

impl<P> StreamingVad for VadSegmenter<P>
where
    P: SpeechProb + Send,
{
    fn push(&mut self, samples_16k: &[f32]) -> Vec<VadEvent> {
        VadSegmenter::push(self, samples_16k)
    }

    fn flush(&mut self) -> Option<VadEvent> {
        VadSegmenter::flush(self)
    }

    fn set_min_silence_ms(&mut self, ms: u64) {
        VadSegmenter::set_min_silence_ms(self, ms);
    }
}

pub struct SessionConfig {
    pub session_id: u64,
    pub input_sample_rate: u32,
    pub input_channels: u16,
    pub max_input_buffer_sec: u64,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub word_timestamps: bool,
    pub vad: Box<dyn StreamingVad>,
    /// Cooperative cancellation flag. Set by the transport (e.g. on socket
    /// disconnect) to stop submitting further segments without killing the
    /// in-flight decode. See Task 1.7 / spec §9.
    pub cancel: Arc<AtomicBool>,
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
    InputAudioTranscriptionDelta {
        item_id: String,
        content_index: u64,
        delta: String,
    },
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    InputAudioTranscriptionCompleted {
        item_id: String,
        content_index: u64,
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

struct SessionRuntime {
    session_id: u64,
    input_sample_rate: u32,
    input_channels: u16,
    max_input_buffer_sec: u64,
    language: Option<String>,
    prompt: Option<String>,
    word_timestamps: bool,
    vad: Box<dyn StreamingVad>,
    cancel: Arc<AtomicBool>,
    current_item_id: Option<String>,
    next_item_index: u64,
    confirmed_transcripts: Vec<String>,
    emitted_delta_text: String,
}

impl SessionRuntime {
    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

pub async fn run_session(
    cfg: SessionConfig,
    mut audio_rx: mpsc::Receiver<SessionInput>,
    event_tx: mpsc::Sender<ServerEvent>,
    executor: Arc<dyn AudioExecutor>,
    queue: InferenceQueue,
    model: ModelConfig,
) -> Result<()> {
    let cancel = Arc::clone(&cfg.cancel);
    // Cancel the session if this future is dropped (e.g. task aborted) so a
    // re-entrant decode loop cannot keep running.
    let _cancel_on_drop = SessionCancelOnDrop(Arc::clone(&cancel));
    let mut runtime = SessionRuntime {
        session_id: cfg.session_id,
        input_sample_rate: cfg.input_sample_rate,
        input_channels: cfg.input_channels,
        max_input_buffer_sec: cfg.max_input_buffer_sec,
        language: cfg.language,
        prompt: cfg.prompt,
        word_timestamps: cfg.word_timestamps,
        vad: cfg.vad,
        cancel: Arc::clone(&cancel),
        current_item_id: None,
        next_item_index: 0,
        confirmed_transcripts: Vec::new(),
        emitted_delta_text: String::new(),
    };

    send_event(
        &event_tx,
        ServerEvent::TranscriptionSessionCreated {
            session_id: runtime.session_id,
        },
    )
    .await?;

    while let Some(input) = audio_rx.recv().await {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        match input {
            SessionInput::ClientEvent(event) => {
                handle_client_event(event, &mut runtime, &event_tx, &executor, &queue, &model)
                    .await?;
            }
            SessionInput::Audio(bytes) => {
                process_audio_bytes(bytes, &mut runtime, &event_tx, &executor, &queue, &model)
                    .await?;
            }
            SessionInput::Control(control) => {
                handle_control(control, &mut runtime, &event_tx, &executor, &queue, &model).await?;
            }
        }
    }

    if runtime.is_cancelled() {
        // Cooperative cancel: do not decode trailing speech after disconnect.
        return Ok(());
    }
    flush_segment(&mut runtime, &event_tx, &executor, &queue, &model).await
}

struct SessionCancelOnDrop(Arc<AtomicBool>);

impl Drop for SessionCancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

async fn handle_client_event(
    event: ClientEvent,
    runtime: &mut SessionRuntime,
    event_tx: &mpsc::Sender<ServerEvent>,
    executor: &Arc<dyn AudioExecutor>,
    queue: &InferenceQueue,
    model: &ModelConfig,
) -> Result<()> {
    match event {
        ClientEvent::TranscriptionSessionUpdate { session } => {
            handle_control(
                ClientControl::Update(session),
                runtime,
                event_tx,
                executor,
                queue,
                model,
            )
            .await
        }
        ClientEvent::InputAudioBufferAppend { audio } => {
            let bytes = general_purpose::STANDARD
                .decode(audio)
                .context("invalid base64 audio")?;
            process_audio_bytes(bytes, runtime, event_tx, executor, queue, model).await
        }
        ClientEvent::InputAudioBufferCommit => {
            handle_control(
                ClientControl::Commit,
                runtime,
                event_tx,
                executor,
                queue,
                model,
            )
            .await
        }
    }
}

async fn handle_control(
    control: ClientControl,
    runtime: &mut SessionRuntime,
    event_tx: &mpsc::Sender<ServerEvent>,
    executor: &Arc<dyn AudioExecutor>,
    queue: &InferenceQueue,
    model: &ModelConfig,
) -> Result<()> {
    match control {
        ClientControl::Update(session) => {
            apply_session_update(session, runtime)?;
            send_event(event_tx, ServerEvent::TranscriptionSessionUpdated).await
        }
        ClientControl::Commit => flush_segment(runtime, event_tx, executor, queue, model).await,
    }
}

fn apply_session_update(
    session: TranscriptionSessionConfig,
    runtime: &mut SessionRuntime,
) -> Result<()> {
    if let Some(format) = session.input_audio_format {
        if format != "pcm16" {
            bail!("unsupported input_audio_format: {format}");
        }
    }
    if let Some(sample_rate) = session.sample_rate {
        runtime.input_sample_rate = sample_rate;
    }
    if let Some(transcription) = session.input_audio_transcription {
        runtime.language = transcription.language;
        runtime.prompt = transcription.prompt;
    }
    if let Some(turn_detection) = session.turn_detection {
        if turn_detection.kind != "server_vad" {
            bail!("unsupported turn_detection type: {}", turn_detection.kind);
        }
        if let Some(silence_duration_ms) = turn_detection.silence_duration_ms {
            runtime.vad.set_min_silence_ms(silence_duration_ms);
        }
    }
    if let Some(word_timestamps) = session.word_timestamps {
        runtime.word_timestamps = word_timestamps;
    }

    Ok(())
}

async fn process_audio_bytes(
    bytes: Vec<u8>,
    runtime: &mut SessionRuntime,
    event_tx: &mpsc::Sender<ServerEvent>,
    executor: &Arc<dyn AudioExecutor>,
    queue: &InferenceQueue,
    model: &ModelConfig,
) -> Result<()> {
    validate_input_buffer_len(&bytes, runtime)?;
    let samples = pcm_s16le_to_f32(&bytes)?;
    let samples_16k =
        resample_to_16k_mono(&samples, runtime.input_sample_rate, runtime.input_channels)?;
    let events = runtime.vad.push(&samples_16k);
    handle_vad_events(events, runtime, event_tx, executor, queue, model).await
}

fn validate_input_buffer_len(bytes: &[u8], runtime: &SessionRuntime) -> Result<()> {
    let max_bytes = u64::from(runtime.input_sample_rate)
        .checked_mul(u64::from(runtime.input_channels))
        .and_then(|samples| samples.checked_mul(2))
        .and_then(|bytes_per_sec| bytes_per_sec.checked_mul(runtime.max_input_buffer_sec))
        .and_then(|max_bytes| usize::try_from(max_bytes).ok())
        .context("streaming input buffer limit overflow")?;

    if bytes.len() > max_bytes {
        bail!("streaming input buffer exceeded");
    }
    Ok(())
}

fn pcm_s16le_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        bail!("invalid audio: pcm16 byte length must be even");
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
        .collect())
}

async fn handle_vad_events(
    events: Vec<VadEvent>,
    runtime: &mut SessionRuntime,
    event_tx: &mpsc::Sender<ServerEvent>,
    executor: &Arc<dyn AudioExecutor>,
    queue: &InferenceQueue,
    model: &ModelConfig,
) -> Result<()> {
    for event in events {
        if runtime.is_cancelled() {
            // Cooperative cancel: stop starting/decoding further segments.
            break;
        }
        match event {
            VadEvent::SpeechStart { at_ms } => {
                let item_id = next_item_id(runtime);
                runtime.current_item_id = Some(item_id.clone());
                runtime.emitted_delta_text.clear();
                send_event(
                    event_tx,
                    ServerEvent::InputAudioBufferSpeechStarted {
                        audio_start_ms: at_ms,
                        item_id,
                    },
                )
                .await?;
            }
            VadEvent::SpeechEnd {
                end_ms, samples, ..
            } => {
                decode_segment(samples, end_ms, runtime, event_tx, executor, queue, model).await?;
            }
            VadEvent::SpeechPartial { samples, .. } => {
                emit_partial_delta(samples, runtime, event_tx, executor, queue, model).await?;
            }
        }
    }
    Ok(())
}

async fn flush_segment(
    runtime: &mut SessionRuntime,
    event_tx: &mpsc::Sender<ServerEvent>,
    executor: &Arc<dyn AudioExecutor>,
    queue: &InferenceQueue,
    model: &ModelConfig,
) -> Result<()> {
    if let Some(event) = runtime.vad.flush() {
        handle_vad_events(vec![event], runtime, event_tx, executor, queue, model).await?;
    }
    Ok(())
}

async fn decode_segment(
    samples: Vec<f32>,
    audio_end_ms: u64,
    runtime: &mut SessionRuntime,
    event_tx: &mpsc::Sender<ServerEvent>,
    executor: &Arc<dyn AudioExecutor>,
    queue: &InferenceQueue,
    model: &ModelConfig,
) -> Result<()> {
    let item_id = runtime
        .current_item_id
        .take()
        .unwrap_or_else(|| next_item_id(runtime));
    send_event(
        event_tx,
        ServerEvent::InputAudioBufferSpeechStopped {
            audio_end_ms,
            item_id: item_id.clone(),
        },
    )
    .await?;
    send_event(
        event_tx,
        ServerEvent::InputAudioBufferCommitted {
            item_id: item_id.clone(),
        },
    )
    .await?;

    let options = AudioInferenceOptions {
        endpoint: AudioEndpoint::Transcriptions,
        language: runtime.language.clone(),
        prompt: conditioned_prompt(&runtime.prompt, &runtime.confirmed_transcripts),
        temperature: None,
        return_timestamps: runtime.word_timestamps,
    };
    let output = transcribe_segment(samples, options, executor, queue, model).await?;
    let words = if runtime.word_timestamps && !output.words.is_empty() {
        Some(
            output
                .words
                .iter()
                .map(|word| TranscriptionWord {
                    text: word.word.clone(),
                    start_ms: (word.start * 1000.0).round() as u64,
                    end_ms: (word.end * 1000.0).round() as u64,
                })
                .collect(),
        )
    } else {
        None
    };
    let transcript = output.text;
    if !transcript.trim().is_empty() {
        runtime.confirmed_transcripts.push(transcript.clone());
    }

    send_event(
        event_tx,
        ServerEvent::InputAudioTranscriptionCompleted {
            item_id,
            content_index: 0,
            transcript,
            words,
        },
    )
    .await
}

async fn emit_partial_delta(
    samples: Vec<f32>,
    runtime: &mut SessionRuntime,
    event_tx: &mpsc::Sender<ServerEvent>,
    executor: &Arc<dyn AudioExecutor>,
    queue: &InferenceQueue,
    model: &ModelConfig,
) -> Result<()> {
    let options = AudioInferenceOptions {
        endpoint: AudioEndpoint::Transcriptions,
        language: runtime.language.clone(),
        prompt: conditioned_prompt(&runtime.prompt, &runtime.confirmed_transcripts),
        temperature: None,
        return_timestamps: false,
    };
    let output = match transcribe_segment(samples, options, executor, queue, model).await {
        Ok(output) => output,
        Err(error) => {
            tracing::warn!(%error, "partial transcription decode failed; skipping delta");
            return Ok(());
        }
    };

    let new_full = output.text;
    let Some(delta) = new_full.strip_prefix(&runtime.emitted_delta_text) else {
        return Ok(());
    };
    if delta.is_empty() {
        return Ok(());
    }

    let Some(item_id) = runtime.current_item_id.clone() else {
        return Ok(());
    };

    send_event(
        event_tx,
        ServerEvent::InputAudioTranscriptionDelta {
            item_id,
            content_index: 0,
            delta: delta.to_string(),
        },
    )
    .await?;
    runtime.emitted_delta_text = new_full;
    Ok(())
}

async fn transcribe_segment(
    samples: Vec<f32>,
    options: AudioInferenceOptions,
    executor: &Arc<dyn AudioExecutor>,
    queue: &InferenceQueue,
    model: &ModelConfig,
) -> Result<AudioOutput> {
    let executor = Arc::clone(executor);
    let model = model.clone();

    let output = queue
        .submit(InferenceJob::new(move || {
            executor
                .transcribe(&model, &samples, &options)
                .map(InferenceOutput::Audio)
        }))
        .await?;

    match output {
        InferenceOutput::Audio(output) => Ok(output),
        other => bail!("unexpected queue output for audio transcription: {other:?}"),
    }
}

fn conditioned_prompt(
    session_prompt: &Option<String>,
    confirmed_transcripts: &[String],
) -> Option<String> {
    let history = confirmed_transcripts
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    match (session_prompt.as_deref(), history.is_empty()) {
        (Some(prompt), true) => Some(prompt.to_string()),
        (Some(prompt), false) => Some(format!("{prompt}\n{history}")),
        (None, false) => Some(history),
        (None, true) => None,
    }
}

fn next_item_id(runtime: &mut SessionRuntime) -> String {
    let item_id = format!("item_{}", runtime.next_item_index);
    runtime.next_item_index += 1;
    item_id
}

async fn send_event(event_tx: &mpsc::Sender<ServerEvent>, event: ServerEvent) -> Result<()> {
    event_tx
        .send(event)
        .await
        .map_err(|_| anyhow::anyhow!("streaming event receiver closed"))
}
