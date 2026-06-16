use ai2npu::config::AppConfig;
use ai2npu::model_registry::{required_bundle_files, ModelRegistry};

#[test]
fn example_config_is_valid() {
    let text = std::fs::read_to_string("config.example.toml").unwrap();
    let cfg: AppConfig = toml::from_str(&text).unwrap();

    cfg.validate().unwrap();
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
