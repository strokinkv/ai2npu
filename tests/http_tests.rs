use ai2npu::audio::AudioOutput;
use ai2npu::config::{AppConfig, ModelConfig};
use ai2npu::http::build_router_with_executors;
use ai2npu::inference::{AudioExecutor, AudioInferenceOptions, EmbeddingExecutor};
use ai2npu::openvino_backend::OpenVinoStatus;
use anyhow::{bail, Result};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;

#[derive(Debug, Clone)]
struct StaticAudioExecutor {
    output: AudioOutput,
    loaded_models: Vec<String>,
    unload_count: Arc<Mutex<usize>>,
}

impl StaticAudioExecutor {
    fn new(output: AudioOutput) -> Self {
        Self {
            output,
            loaded_models: Vec::new(),
            unload_count: Arc::new(Mutex::new(0)),
        }
    }

    fn with_loaded_models(output: AudioOutput, loaded_models: Vec<String>) -> Self {
        Self {
            output,
            loaded_models,
            unload_count: Arc::new(Mutex::new(0)),
        }
    }
}

impl AudioExecutor for StaticAudioExecutor {
    fn transcribe(
        &self,
        _model: &ModelConfig,
        _samples: &[f32],
        _options: &AudioInferenceOptions,
    ) -> Result<AudioOutput> {
        Ok(self.output.clone())
    }

    fn loaded_models(&self) -> Vec<String> {
        self.loaded_models.clone()
    }

    fn unload_all(&self) -> Result<usize> {
        let mut count = self.unload_count.lock().unwrap();
        *count += 1;
        Ok(self.loaded_models.len())
    }
}

#[derive(Debug, Clone)]
struct StaticEmbeddingExecutor {
    embeddings: Vec<Vec<f32>>,
    loaded_models: Vec<String>,
    block_embed_rx: Arc<Mutex<Option<std::sync::mpsc::Receiver<()>>>>,
    unload_count: Arc<Mutex<usize>>,
}

impl StaticEmbeddingExecutor {
    fn new(embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            embeddings,
            loaded_models: Vec::new(),
            block_embed_rx: Arc::new(Mutex::new(None)),
            unload_count: Arc::new(Mutex::new(0)),
        }
    }

    fn with_loaded_models(embeddings: Vec<Vec<f32>>, loaded_models: Vec<String>) -> Self {
        Self {
            embeddings,
            loaded_models,
            block_embed_rx: Arc::new(Mutex::new(None)),
            unload_count: Arc::new(Mutex::new(0)),
        }
    }

    fn with_blocking_embed(
        embeddings: Vec<Vec<f32>>,
        release_rx: std::sync::mpsc::Receiver<()>,
    ) -> Self {
        Self {
            embeddings,
            loaded_models: vec!["BAAI/bge-m3".to_string()],
            block_embed_rx: Arc::new(Mutex::new(Some(release_rx))),
            unload_count: Arc::new(Mutex::new(0)),
        }
    }
}

impl EmbeddingExecutor for StaticEmbeddingExecutor {
    fn embed(&self, _model: &ModelConfig, input: &[String]) -> Result<Vec<Vec<f32>>> {
        if let Some(rx) = self.block_embed_rx.lock().unwrap().take() {
            rx.recv().unwrap();
        }
        if self.embeddings.len() == input.len() {
            return Ok(self.embeddings.clone());
        }
        if self.embeddings.len() == 1 {
            return Ok(vec![self.embeddings[0].clone(); input.len()]);
        }
        bail!("static embedding count does not match input count")
    }

    fn loaded_models(&self) -> Vec<String> {
        self.loaded_models.clone()
    }

    fn unload_all(&self) -> Result<usize> {
        let mut count = self.unload_count.lock().unwrap();
        *count += 1;
        Ok(self.loaded_models.len())
    }
}

fn example_config() -> AppConfig {
    let text = std::fs::read_to_string("config.example.toml").unwrap();
    toml::from_str(&text).unwrap()
}

fn example_config_with_body_limit(limit_mb: u64) -> AppConfig {
    let mut config = example_config();
    config.server.request_body_limit_mb = limit_mb;
    config
}

fn status_with_npu() -> OpenVinoStatus {
    OpenVinoStatus {
        runtime_available: true,
        devices: vec!["CPU".to_string(), "GPU".to_string(), "NPU".to_string()],
        npu_available: true,
        error: None,
    }
}

async fn get_json(path: &str) -> (StatusCode, Value) {
    let app = build_router_with_executors(
        example_config(),
        status_with_npu(),
        Arc::new(StaticEmbeddingExecutor::new(vec![vec![0.1, 0.2, 0.3]])),
        Arc::new(StaticAudioExecutor::new(AudioOutput {
            text: String::new(),
            language: None,
            duration: 0.0,
            segments: Vec::new(),
        })),
    );
    let response = app
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

async fn get_json_with_loaded_models(path: &str) -> (StatusCode, Value) {
    let app = build_router_with_executors(
        example_config(),
        status_with_npu(),
        Arc::new(StaticEmbeddingExecutor::with_loaded_models(
            vec![vec![0.1, 0.2, 0.3]],
            vec!["BAAI/bge-m3".to_string()],
        )),
        Arc::new(StaticAudioExecutor::with_loaded_models(
            AudioOutput {
                text: String::new(),
                language: None,
                duration: 0.0,
                segments: Vec::new(),
            },
            vec!["openai/whisper-large-v3-turbo".to_string()],
        )),
    );
    let response = app
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

async fn post_embeddings_json(body: Value) -> (StatusCode, Value) {
    let app = build_router_with_executors(
        example_config(),
        status_with_npu(),
        Arc::new(StaticEmbeddingExecutor::new(vec![vec![0.1, 0.2, 0.3]])),
        Arc::new(StaticAudioExecutor::new(AudioOutput {
            text: String::new(),
            language: None,
            duration: 0.0,
            segments: Vec::new(),
        })),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

async fn post_embeddings_raw_with_config(config: AppConfig, body: Vec<u8>) -> (StatusCode, Value) {
    let app = build_router_with_executors(
        config,
        status_with_npu(),
        Arc::new(StaticEmbeddingExecutor::new(vec![vec![0.1, 0.2, 0.3]])),
        Arc::new(StaticAudioExecutor::new(AudioOutput {
            text: String::new(),
            language: None,
            duration: 0.0,
            segments: Vec::new(),
        })),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::CONTENT_LENGTH, body.len().to_string())
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

fn wav_header(
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    audio_format: u16,
    data_len: u32,
) -> Vec<u8> {
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let riff_size = 36 + data_len;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&audio_format.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    bytes.resize(bytes.len() + data_len as usize, 0);
    bytes
}

async fn post_audio_multipart_raw(
    path: &str,
    wav: Vec<u8>,
    extra_fields: &[(&str, &str)],
) -> (StatusCode, String, Option<String>) {
    let boundary = "ai2npu-test-boundary";
    let mut body = Vec::new();
    for (name, value) in extra_fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body.extend_from_slice(&wav);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let app = build_router_with_executors(
        example_config(),
        status_with_npu(),
        Arc::new(StaticEmbeddingExecutor::new(vec![vec![0.1, 0.2, 0.3]])),
        Arc::new(StaticAudioExecutor::new(AudioOutput {
            text: String::new(),
            language: None,
            duration: 0.0,
            segments: Vec::new(),
        })),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(path)
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    (status, body, content_type)
}

async fn post_audio_multipart(
    path: &str,
    wav: Vec<u8>,
    extra_fields: &[(&str, &str)],
) -> (StatusCode, Value) {
    let (status, body, _) = post_audio_multipart_raw(path, wav, extra_fields).await;
    let json = serde_json::from_str(&body).unwrap();
    (status, json)
}

#[tokio::test]
async fn models_returns_enabled_configured_models() {
    let (status, json) = get_json("/v1/models").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["id"], "BAAI/bge-m3");
    assert_eq!(json["data"][1]["id"], "openai/whisper-large-v3-turbo");
}

#[tokio::test]
async fn health_returns_service_and_npu_status() {
    let (status, json) = get_json("/health").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "ok");
    assert_eq!(json["openvino"]["npu_available"], true);
}

#[tokio::test]
async fn health_reports_loaded_models_from_executors() {
    let (status, json) = get_json_with_loaded_models("/health").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["loaded_models"],
        serde_json::json!(["BAAI/bge-m3", "openai/whisper-large-v3-turbo"])
    );
}

#[tokio::test]
async fn admin_unload_waits_for_active_request_and_unloads_all_models() {
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let embedding_executor = Arc::new(StaticEmbeddingExecutor::with_blocking_embed(
        vec![vec![0.1, 0.2, 0.3]],
        release_rx,
    ));
    let audio_executor = Arc::new(StaticAudioExecutor::with_loaded_models(
        AudioOutput {
            text: String::new(),
            language: None,
            duration: 0.0,
            segments: Vec::new(),
        },
        vec!["openai/whisper-large-v3-turbo".to_string()],
    ));
    let app = build_router_with_executors(
        example_config(),
        status_with_npu(),
        embedding_executor.clone(),
        audio_executor.clone(),
    );

    let active_app = app.clone();
    let active = tokio::spawn(async move {
        active_app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/embeddings")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "BAAI/bge-m3",
                            "input": "hello"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let unload_app = app.clone();
    let unload = tokio::spawn(async move {
        unload_app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/admin/models/unload")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!unload.is_finished());

    release_tx.send(()).unwrap();
    let active_response = active.await.unwrap();
    assert_eq!(active_response.status(), StatusCode::OK);
    let unload_response = unload.await.unwrap();
    assert_eq!(unload_response.status(), StatusCode::OK);
    let bytes = unload_response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    let json: Value = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json["object"], "model_unload");
    assert_eq!(json["unloaded_model_count"], 2);
    assert_eq!(*embedding_executor.unload_count.lock().unwrap(), 1);
    assert_eq!(*audio_executor.unload_count.lock().unwrap(), 1);
}

#[tokio::test]
async fn unknown_endpoint_returns_openai_like_error() {
    let (status, json) = get_json("/missing").await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn embeddings_returns_openai_like_response() {
    let (status, json) = post_embeddings_json(serde_json::json!({
        "model": "BAAI/bge-m3",
        "input": "hello"
    }))
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "list");
    assert_eq!(json["model"], "BAAI/bge-m3");
    assert_eq!(json["data"][0]["object"], "embedding");
    assert_eq!(json["data"][0]["index"], 0);
    assert_eq!(
        json["data"][0]["embedding"],
        serde_json::json!([0.1, 0.2, 0.3])
    );
}

#[tokio::test]
async fn embeddings_unknown_model_returns_model_not_found() {
    let (status, json) = post_embeddings_json(serde_json::json!({
        "model": "missing",
        "input": "hello"
    }))
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn embeddings_invalid_input_returns_invalid_request() {
    let (status, json) = post_embeddings_json(serde_json::json!({
        "model": "BAAI/bge-m3",
        "input": []
    }))
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn oversized_request_returns_openai_like_payload_too_large() {
    let (status, json) =
        post_embeddings_raw_with_config(example_config_with_body_limit(1), vec![b' '; 1_048_577])
            .await;

    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "payload_too_large");
}

#[tokio::test]
async fn audio_rejects_invalid_wav_with_stable_error_code() {
    let wav = wav_header(2, 16_000, 16, 1, 64_000);

    let (status, json) = post_audio_multipart(
        "/v1/audio/transcriptions",
        wav,
        &[("model", "openai/whisper-large-v3-turbo")],
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "invalid_audio_format");
}

#[tokio::test]
async fn audio_transcription_returns_json_response() {
    let wav = wav_header(1, 16_000, 16, 1, 32_000);

    let (status, json) = post_audio_multipart(
        "/v1/audio/transcriptions",
        wav,
        &[
            ("model", "openai/whisper-large-v3-turbo"),
            ("response_format", "json"),
        ],
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["text"], "");
}

#[tokio::test]
async fn audio_transcription_suppresses_silence_hallucination() {
    let wav = wav_header(1, 16_000, 16, 1, 32_000);
    let app = build_router_with_executors(
        example_config(),
        status_with_npu(),
        Arc::new(StaticEmbeddingExecutor::new(vec![vec![0.1, 0.2, 0.3]])),
        Arc::new(StaticAudioExecutor::new(AudioOutput {
            text: "Продолжение следует...".to_string(),
            language: None,
            duration: 1.0,
            segments: Vec::new(),
        })),
    );
    let boundary = "ai2npu-test-boundary";
    let mut body = Vec::new();
    for (name, value) in [
        ("model", "openai/whisper-large-v3-turbo"),
        ("response_format", "json"),
    ] {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body.extend_from_slice(&wav);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/audio/transcriptions")
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["text"], "");
}

#[tokio::test]
async fn audio_transcription_text_response_is_not_json_wrapped() {
    let wav = wav_header(1, 16_000, 16, 1, 32_000);

    let (status, body, content_type) = post_audio_multipart_raw(
        "/v1/audio/transcriptions",
        wav,
        &[
            ("model", "openai/whisper-large-v3-turbo"),
            ("response_format", "text"),
        ],
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "");
    assert!(!content_type
        .as_deref()
        .unwrap_or_default()
        .starts_with("application/json"));
}

#[tokio::test]
async fn audio_rejects_unsupported_response_format() {
    let wav = wav_header(1, 16_000, 16, 1, 32_000);

    let (status, json) = post_audio_multipart(
        "/v1/audio/transcriptions",
        wav,
        &[
            ("model", "openai/whisper-large-v3-turbo"),
            ("response_format", "diarized_json"),
        ],
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn audio_translations_reject_timestamp_granularities() {
    let wav = wav_header(1, 16_000, 16, 1, 32_000);

    let (status, json) = post_audio_multipart(
        "/v1/audio/translations",
        wav,
        &[
            ("model", "openai/whisper-large-v3-turbo"),
            ("timestamp_granularities", "segment"),
        ],
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn audio_rejects_embedding_model() {
    let wav = wav_header(1, 16_000, 16, 1, 32_000);

    let (status, json) =
        post_audio_multipart("/v1/audio/transcriptions", wav, &[("model", "BAAI/bge-m3")]).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["code"], "model_not_found");
}
