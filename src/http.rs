use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Json as JsonExtractor, Multipart, Query, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::audio::{
    is_effectively_empty_audio, validate_wav, AudioEndpoint, AudioJsonResponse,
    AudioMultipartRequest, AudioOutput, AudioResponseFormat,
};
use crate::config::{AppConfig, ModelType};
use crate::embeddings::{
    normalize_input, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
};
use crate::error::ApiError;
use crate::inference::{
    audio_executor_from_env, embedding_executor_from_env, AudioExecutor, AudioInferenceOptions,
    EmbeddingExecutor,
};
use crate::logs::tail_log_file;
use crate::openvino_backend::OpenVinoStatus;
use crate::queue::{InferenceJob, InferenceOutput, InferenceQueue, QueueError};

#[derive(Clone)]
pub struct AppState {
    config: Arc<AppConfig>,
    openvino: OpenVinoStatus,
    embedding_executor: Arc<dyn EmbeddingExecutor>,
    audio_executor: Arc<dyn AudioExecutor>,
    queue: InferenceQueue,
    current_request: Arc<Mutex<Option<String>>>,
    started_at: Instant,
}

pub fn build_router_with_executors(
    config: AppConfig,
    openvino: OpenVinoStatus,
    embedding_executor: Arc<dyn EmbeddingExecutor>,
    audio_executor: Arc<dyn AudioExecutor>,
) -> Router {
    let max_pending_requests = config.queue.max_pending_requests;
    let request_body_limit_bytes = config
        .server
        .request_body_limit_mb
        .saturating_mul(1024 * 1024);
    preload_models(&config, &embedding_executor, &audio_executor)
        .expect("failed to preload configured models");
    let state = AppState {
        config: Arc::new(config),
        openvino,
        embedding_executor,
        audio_executor,
        queue: InferenceQueue::new(max_pending_requests),
        current_request: Arc::new(Mutex::new(None)),
        started_at: Instant::now(),
    };

    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/embeddings", post(create_embeddings))
        .route("/v1/audio/transcriptions", post(create_transcription))
        .route("/v1/audio/translations", post(create_translation))
        .route("/admin/models/unload", post(unload_models))
        .route("/health", get(health))
        .route("/logs", get(logs))
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(
            request_body_limit_bytes,
            enforce_body_limit,
        ))
        .with_state(state)
}

async fn enforce_body_limit(
    State(limit_bytes): State<u64>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(content_length) = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    {
        if content_length > limit_bytes {
            return ApiError::payload_too_large(format!(
                "request body exceeds configured limit of {limit_bytes} bytes"
            ))
            .into_response();
        }
    }

    next.run(request).await
}

fn preload_models(
    config: &AppConfig,
    embedding_executor: &Arc<dyn EmbeddingExecutor>,
    audio_executor: &Arc<dyn AudioExecutor>,
) -> anyhow::Result<()> {
    for model in config
        .models
        .iter()
        .filter(|model| model.enabled && model.preload)
    {
        match model.model_type {
            ModelType::Embedding => embedding_executor.preload(model)?,
            ModelType::Whisper => audio_executor.preload(model)?,
        }
    }
    Ok(())
}

async fn create_transcription(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    create_audio(AudioEndpoint::Transcriptions, state, multipart).await
}

async fn create_translation(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    create_audio(AudioEndpoint::Translations, state, multipart).await
}

async fn create_audio(
    endpoint: AudioEndpoint,
    state: AppState,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    let request = parse_audio_multipart(multipart).await?;
    let model_id = request
        .model
        .as_deref()
        .ok_or_else(|| ApiError::invalid_request("audio model is required"))?;
    let model = state
        .config
        .models
        .iter()
        .find(|model| model.enabled && model.id == model_id)
        .ok_or_else(|| ApiError::model_not_found(format!("model not found: {model_id}")))?;
    if model.model_type != ModelType::Whisper {
        return Err(ApiError::model_not_found(format!(
            "model is not a whisper model: {}",
            model.id
        )));
    }

    let response_format = AudioResponseFormat::parse(request.response_format.as_deref())
        .map_err(|error| ApiError::invalid_request(error.to_string()))?;
    if endpoint == AudioEndpoint::Translations && !request.timestamp_granularities.is_empty() {
        return Err(ApiError::invalid_request(
            "timestamp_granularities is not supported for translations",
        ));
    }

    let file = request
        .file
        .clone()
        .ok_or_else(|| ApiError::invalid_request("audio file is required"))?;
    let wav_info = validate_wav(&file, model.max_audio_duration_sec.unwrap_or(1800))
        .map_err(|error| ApiError::invalid_audio_format(error.to_string()))?;
    if is_effectively_empty_audio(&file, &wav_info)
        .map_err(|error| ApiError::invalid_audio_format(error.to_string()))?
    {
        return Ok(audio_response(
            response_format,
            AudioOutput {
                text: String::new(),
                language: request.language.clone(),
                duration: wav_info.duration_sec,
                segments: Vec::new(),
            },
        ));
    }
    let return_timestamps = response_format == AudioResponseFormat::VerboseJson
        && request
            .timestamp_granularities
            .iter()
            .any(|value| value == "segment" || value == "word");
    let options = AudioInferenceOptions {
        endpoint,
        language: request.language.clone(),
        prompt: request.prompt.clone(),
        return_timestamps,
    };
    let executor = Arc::clone(&state.audio_executor);
    let current_request = Arc::clone(&state.current_request);
    let queue_timeout = Duration::from_secs(model.queue_timeout_sec);
    let model = model.clone();
    let job = InferenceJob::new(move || {
        let _guard =
            CurrentRequestGuard::new(&current_request, format!("{endpoint:?} {}", model.id));
        let mut output = executor.transcribe(&model, &file, &options)?;
        if output.duration == 0.0 {
            output.duration = wav_info.duration_sec;
        }
        Ok(InferenceOutput::Audio(output))
    });

    let output = state
        .queue
        .submit_with_timeout(job, queue_timeout)
        .await
        .map_err(queue_error_to_api)?;
    let InferenceOutput::Audio(output) = output else {
        return Err(ApiError::internal("unexpected non-audio inference output"));
    };

    Ok(audio_response(response_format, output))
}

fn audio_response(response_format: AudioResponseFormat, output: AudioOutput) -> Response {
    match response_format {
        AudioResponseFormat::Json => (
            StatusCode::OK,
            axum::Json(AudioJsonResponse { text: output.text }),
        )
            .into_response(),
        AudioResponseFormat::VerboseJson => (
            StatusCode::OK,
            axum::Json(json!({
                "text": output.text,
                "language": output.language.unwrap_or_default(),
                "duration": output.duration,
                "segments": output.segments
            })),
        )
            .into_response(),
        AudioResponseFormat::Text | AudioResponseFormat::Srt | AudioResponseFormat::Vtt => {
            (StatusCode::OK, output.text).into_response()
        }
    }
}

async fn parse_audio_multipart(
    mut multipart: Multipart,
) -> Result<AudioMultipartRequest, ApiError> {
    let mut request = AudioMultipartRequest::default();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::invalid_request(format!("invalid multipart body: {error}")))?
    {
        let Some(name) = field.name().map(str::to_owned) else {
            continue;
        };
        match name.as_str() {
            "file" => {
                request.file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|error| {
                            ApiError::invalid_request(format!("failed to read audio file: {error}"))
                        })?
                        .to_vec(),
                );
            }
            "model" => request.model = Some(read_text_field(field).await?),
            "language" => request.language = Some(read_text_field(field).await?),
            "prompt" => request.prompt = Some(read_text_field(field).await?),
            "response_format" => request.response_format = Some(read_text_field(field).await?),
            "temperature" => request.temperature = Some(read_text_field(field).await?),
            "timestamp_granularities" => {
                request
                    .timestamp_granularities
                    .push(read_text_field(field).await?);
            }
            _ => {
                tracing::debug!("ignoring unsupported audio multipart field: {name}");
            }
        }
    }

    Ok(request)
}

async fn read_text_field(field: axum::extract::multipart::Field<'_>) -> Result<String, ApiError> {
    field.text().await.map_err(|error| {
        ApiError::invalid_request(format!("failed to read multipart field: {error}"))
    })
}

pub async fn serve(config: AppConfig, openvino: OpenVinoStatus) -> anyhow::Result<()> {
    serve_until(config, openvino, async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await
}

pub async fn serve_until(
    config: AppConfig,
    openvino: OpenVinoStatus,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let selection = embedding_executor_from_env()?;
    tracing::info!("embedding executor: {:?}", selection.kind());
    let executor = selection.into_executor();
    let audio_executor = audio_executor_from_env()?;
    let router = build_router_with_executors(config, openvino, executor, audio_executor);

    tracing::info!("listening on http://{}", addr);
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await?;

    Ok(())
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let data: Vec<_> = state
        .config
        .models
        .iter()
        .filter(|model| model.enabled)
        .map(|model| {
            json!({
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "local"
            })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": data
    }))
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let configured_models = state
        .config
        .models
        .iter()
        .map(|model| model.id.clone())
        .collect();
    let mut loaded_models = state.embedding_executor.loaded_models();
    loaded_models.extend(state.audio_executor.loaded_models());
    loaded_models.sort();
    loaded_models.dedup();

    Json(HealthResponse {
        status: if state.openvino.npu_available {
            "ok"
        } else {
            "degraded"
        },
        version: env!("CARGO_PKG_VERSION"),
        openvino: state.openvino,
        configured_models,
        loaded_models,
        queue_size: state.queue.pending_len(),
        current_request: state
            .current_request
            .lock()
            .ok()
            .and_then(|value| value.clone()),
        uptime_sec: state.started_at.elapsed().as_secs(),
    })
}

async fn unload_models(State(state): State<AppState>) -> Result<Json<UnloadResponse>, ApiError> {
    let embedding_executor = Arc::clone(&state.embedding_executor);
    let audio_executor = Arc::clone(&state.audio_executor);
    let current_request = Arc::clone(&state.current_request);
    let job = InferenceJob::new(move || {
        let _guard = CurrentRequestGuard::new(&current_request, "admin unload models".to_string());
        let embedding_count = embedding_executor.unload_all()?;
        let audio_count = audio_executor.unload_all()?;
        Ok(InferenceOutput::ModelsUnloaded(
            embedding_count + audio_count,
        ))
    });

    let output = state.queue.submit(job).await.map_err(queue_error_to_api)?;
    let InferenceOutput::ModelsUnloaded(unloaded_model_count) = output else {
        return Err(ApiError::internal("unexpected non-unload inference output"));
    };

    Ok(Json(UnloadResponse {
        object: "model_unload",
        unloaded_model_count,
    }))
}

async fn create_embeddings(
    State(state): State<AppState>,
    JsonExtractor(request): JsonExtractor<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let model = state
        .config
        .models
        .iter()
        .find(|model| model.enabled && model.id == request.model)
        .cloned()
        .ok_or_else(|| ApiError::model_not_found(format!("model not found: {}", request.model)))?;

    if model.model_type != ModelType::Embedding {
        return Err(ApiError::model_not_found(format!(
            "model is not an embedding model: {}",
            model.id
        )));
    }

    let input = normalize_input(request.input)
        .map_err(|error| ApiError::invalid_request(error.to_string()))?;

    let executor = Arc::clone(&state.embedding_executor);
    let current_request = Arc::clone(&state.current_request);
    let queue_timeout = Duration::from_secs(model.queue_timeout_sec);
    let job_input = input.clone();
    let job = InferenceJob::new(move || {
        let _guard = CurrentRequestGuard::new(&current_request, format!("embedding {}", model.id));
        let embeddings = executor.embed(&model, &job_input)?;
        Ok(InferenceOutput::Embeddings(embeddings))
    });

    let output = state
        .queue
        .submit_with_timeout(job, queue_timeout)
        .await
        .map_err(queue_error_to_api)?;
    let InferenceOutput::Embeddings(embeddings) = output else {
        return Err(ApiError::internal(
            "unexpected non-embedding inference output",
        ));
    };

    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            object: "embedding",
            index,
            embedding,
        })
        .collect();
    let token_estimate = input
        .iter()
        .map(|value| value.split_whitespace().count())
        .sum();

    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model: request.model,
        usage: EmbeddingUsage {
            prompt_tokens: token_estimate,
            total_tokens: token_estimate,
        },
    }))
}

fn queue_error_to_api(error: QueueError) -> ApiError {
    match error {
        QueueError::Full => ApiError::queue_full("queue is full"),
        QueueError::Closed => ApiError::internal("queue worker is closed"),
        QueueError::Timeout => ApiError::queue_timeout("queue wait timed out"),
        QueueError::Job(error) => ApiError::internal(error.to_string()),
    }
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    openvino: OpenVinoStatus,
    configured_models: Vec<String>,
    loaded_models: Vec<String>,
    queue_size: usize,
    current_request: Option<String>,
    uptime_sec: u64,
}

#[derive(Debug, Serialize)]
struct UnloadResponse {
    object: &'static str,
    unloaded_model_count: usize,
}

struct CurrentRequestGuard<'a> {
    current_request: &'a Mutex<Option<String>>,
}

impl<'a> CurrentRequestGuard<'a> {
    fn new(current_request: &'a Mutex<Option<String>>, value: String) -> Self {
        if let Ok(mut current) = current_request.lock() {
            *current = Some(value);
        }
        Self { current_request }
    }
}

impl Drop for CurrentRequestGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut current) = self.current_request.lock() {
            *current = None;
        }
    }
}

#[derive(Debug, Deserialize)]
struct LogsQuery {
    lines: Option<usize>,
}

async fn logs(
    State(state): State<AppState>,
    Query(query): Query<LogsQuery>,
) -> Json<crate::logs::LogTail> {
    let path = state.config.logging.directory.join("ai2npu.log");
    Json(tail_log_file(path, query.lines.unwrap_or(200)))
}

async fn not_found() -> ApiError {
    ApiError::not_found("endpoint not found")
}
