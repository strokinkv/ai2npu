use ai2npu::config::AppConfig;
use ai2npu::model_registry::{required_bundle_files, ModelRegistry};
use std::path::Path;

#[test]
fn example_config_is_valid() {
    let text = std::fs::read_to_string("config.example.toml").unwrap();
    let cfg: AppConfig = toml::from_str(&text).unwrap();

    cfg.validate().unwrap();
}

#[test]
fn example_config_preloads_whisper_without_queue_timeout() {
    let text = std::fs::read_to_string("config.example.toml").unwrap();
    let cfg: AppConfig = toml::from_str(&text).unwrap();

    let whisper = cfg
        .models
        .iter()
        .find(|model| model.id == "openai/whisper-large-v3-turbo")
        .expect("example config must include Whisper");

    assert!(whisper.preload);
    assert_eq!(whisper.queue_timeout_sec, 0);
}

#[test]
fn rejects_non_loopback_host() {
    let text = r#"
[server]
host = "0.0.0.0"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "logs"
max_file_size_mb = 10
max_files = 10

[[models]]
id = "BAAI/bge-m3"
type = "embedding"
path = "models/strokinkv/bge-m3-int8-ov"
enabled = true
preload = false
idle_timeout_sec = 0
queue_timeout_sec = 600
normalize = true
"#;
    let cfg: AppConfig = toml::from_str(text).unwrap();

    let err = cfg.validate().unwrap_err().to_string();

    assert!(err.contains("server.host must be 127.0.0.1"));
}

#[test]
fn rejects_duplicate_model_ids() {
    let text = r#"
[server]
host = "127.0.0.1"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "logs"
max_file_size_mb = 10
max_files = 10

[[models]]
id = "BAAI/bge-m3"
type = "embedding"
path = "models/strokinkv/bge-m3-int8-ov"
enabled = true
preload = false
idle_timeout_sec = 0
queue_timeout_sec = 600
normalize = true

[[models]]
id = "BAAI/bge-m3"
type = "whisper"
path = "models/openai/whisper-base-int8-ov"
enabled = true
preload = false
idle_timeout_sec = 0
queue_timeout_sec = 600
max_audio_duration_sec = 30
"#;
    let cfg: AppConfig = toml::from_str(text).unwrap();

    let err = cfg.validate().unwrap_err().to_string();

    assert!(err.contains("models.id must be unique"));
    assert!(err.contains("BAAI/bge-m3"));
}

#[test]
fn allows_no_configured_models() {
    let text = r#"
[server]
host = "127.0.0.1"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "logs"
max_file_size_mb = 10
max_files = 10
"#;
    let cfg: AppConfig = toml::from_str(text).unwrap();

    cfg.validate().unwrap();
}

#[test]
fn parses_streaming_config_section() {
    let text = r#"
[server]
host = "127.0.0.1"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "logs"
max_file_size_mb = 10
max_files = 10

[streaming]
enabled = true
vad_model_path = "models/silero_vad.onnx"
default_min_silence_ms = 400
default_max_segment_ms = 30000
max_input_buffer_sec = 30
"#;
    let cfg: AppConfig = toml::from_str(text).unwrap();
    let streaming = cfg.streaming.as_ref().unwrap();

    assert!(streaming.enabled);
    assert_eq!(
        streaming.vad_model_path,
        Path::new("models/silero_vad.onnx")
    );
    assert_eq!(streaming.default_min_silence_ms, 400);
    assert_eq!(streaming.default_max_segment_ms, 30000);
    assert_eq!(streaming.max_input_buffer_sec, 30);
    cfg.validate().unwrap();
}

#[test]
fn rejects_zero_streaming_min_silence() {
    let text = r#"
[server]
host = "127.0.0.1"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "logs"
max_file_size_mb = 10
max_files = 10

[streaming]
enabled = true
vad_model_path = "models/silero_vad.onnx"
default_min_silence_ms = 0
default_max_segment_ms = 30000
max_input_buffer_sec = 30
"#;
    let cfg: AppConfig = toml::from_str(text).unwrap();

    let err = cfg.validate().unwrap_err().to_string();

    assert!(err.contains("streaming.default_min_silence_ms must be greater than 0"));
}

#[test]
fn rejects_partial_silence_ge_min_silence() {
    let text = r#"
[server]
host = "127.0.0.1"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "logs"
max_file_size_mb = 10
max_files = 10

[streaming]
enabled = true
vad_model_path = "models/silero_vad.onnx"
default_min_silence_ms = 400
default_max_segment_ms = 30000
max_input_buffer_sec = 30
partial_silence_ms = 400
"#;
    let cfg: AppConfig = toml::from_str(text).unwrap();

    let err = cfg.validate().unwrap_err().to_string();

    assert!(err.contains("partial_silence_ms"), "got: {err}");
}

#[test]
fn partial_silence_ms_defaults_to_zero_when_absent() {
    let text = r#"
[server]
host = "127.0.0.1"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "logs"
max_file_size_mb = 10
max_files = 10

[streaming]
enabled = true
vad_model_path = "models/silero_vad.onnx"
default_min_silence_ms = 400
default_max_segment_ms = 30000
max_input_buffer_sec = 30
"#;
    let cfg: AppConfig = toml::from_str(text).unwrap();

    assert_eq!(cfg.streaming.unwrap().partial_silence_ms, 0);
}

#[test]
fn allows_zero_model_queue_timeout_as_unbounded_wait() {
    let text = r#"
[server]
host = "127.0.0.1"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "logs"
max_file_size_mb = 10
max_files = 10

[[models]]
id = "openai/whisper-large-v3-turbo"
type = "whisper"
path = "models/OpenVINO/whisper-large-v3-turbo-int8-ov"
enabled = true
preload = true
queue_timeout_sec = 0
max_audio_duration_sec = 1800
"#;
    let cfg: AppConfig = toml::from_str(text).unwrap();

    cfg.validate().unwrap();
    assert_eq!(cfg.models[0].queue_timeout_sec, 0);
    assert!(cfg.models[0].preload);
}

#[test]
fn rejects_unknown_model_type_during_deserialization() {
    let text = r#"
[server]
host = "127.0.0.1"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "logs"
max_file_size_mb = 10
max_files = 10

[[models]]
id = "BAAI/bge-m3"
type = "reranker"
path = "models/strokinkv/bge-m3-int8-ov"
enabled = true
preload = false
idle_timeout_sec = 0
queue_timeout_sec = 600
"#;

    let err = toml::from_str::<AppConfig>(text).unwrap_err().to_string();

    assert!(err.contains("unknown variant"));
    assert!(err.contains("reranker"));
}

#[test]
fn required_bundle_files_include_embedding_openvino_artifacts() {
    let files = required_bundle_files("embedding");

    assert!(files.iter().any(|file| file == "model.xml"));
    assert!(files.iter().any(|file| file == "model.bin"));
    assert!(files.iter().any(|file| file == "tokenizer.json"));
    assert!(files.iter().any(|file| file == "sentencepiece.bpe.model"));
}

#[test]
fn required_bundle_files_include_whisper_openvino_artifacts() {
    let files = required_bundle_files("whisper");

    assert!(files
        .iter()
        .any(|file| file == "openvino_encoder_model.xml"));
    assert!(files
        .iter()
        .any(|file| file == "openvino_decoder_model.xml"));
    assert!(files.iter().any(|file| file == "openvino_tokenizer.xml"));
    assert!(files.iter().any(|file| file == "openvino_detokenizer.xml"));
}

#[test]
fn prepared_model_bundles_are_valid() {
    if std::env::var("AI2NPU_RUN_MODEL_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping downloaded model test; set AI2NPU_RUN_MODEL_TESTS=1");
        return;
    }

    let text = std::fs::read_to_string("config.example.toml").unwrap();
    let cfg: AppConfig = toml::from_str(&text).unwrap();
    let registry = ModelRegistry::new(cfg);

    let statuses = registry.validate_bundles();

    assert_eq!(statuses.len(), 2);
    for status in statuses {
        assert!(
            status.valid,
            "{} missing {:?}",
            status.id, status.missing_files
        );
    }
}
