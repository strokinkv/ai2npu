use ai2npu::bge_embeddings::BgeEmbeddingExecutor;
use ai2npu::config::AppConfig;
use ai2npu::embeddings::{normalize_input, EmbeddingRequest};
use ai2npu::inference::EmbeddingExecutor;

#[test]
fn accepts_single_string_input() {
    let req: EmbeddingRequest = serde_json::from_value(serde_json::json!({
        "model": "BAAI/bge-m3",
        "input": "hello"
    }))
    .unwrap();

    assert_eq!(normalize_input(req.input).unwrap(), vec!["hello"]);
}

#[test]
fn accepts_batch_string_input() {
    let req: EmbeddingRequest = serde_json::from_value(serde_json::json!({
        "model": "BAAI/bge-m3",
        "input": ["first", "second"]
    }))
    .unwrap();

    assert_eq!(normalize_input(req.input).unwrap(), vec!["first", "second"]);
}

#[test]
fn rejects_empty_batch() {
    let req: EmbeddingRequest = serde_json::from_value(serde_json::json!({
        "model": "BAAI/bge-m3",
        "input": []
    }))
    .unwrap();

    let err = normalize_input(req.input).unwrap_err().to_string();
    assert!(err.contains("input must not be empty"));
}

#[test]
fn rust_bge_executor_returns_1024_vector_when_enabled() {
    if std::env::var("AI2NPU_RUN_NPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping live NPU test; set AI2NPU_RUN_NPU_TESTS=1");
        return;
    }

    let cfg: AppConfig =
        toml::from_str(&std::fs::read_to_string("config.example.toml").unwrap()).unwrap();
    let model = cfg
        .models
        .iter()
        .find(|model| model.id == "BAAI/bge-m3")
        .unwrap();
    let executor = BgeEmbeddingExecutor::new().unwrap();
    let embeddings = executor.embed(model, &["hello world".to_string()]).unwrap();

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 1024);
}
