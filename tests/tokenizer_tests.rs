use ai2npu::bge_tokenizer::BgeTokenizer;

const BGE_MODEL_DIR: &str = "models/strokinkv/bge-m3-int8-ov";

fn model_tests_enabled() -> bool {
    std::env::var("AI2NPU_RUN_MODEL_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn tokenizer_pads_to_512_tokens() {
    if !model_tests_enabled() {
        eprintln!("skipping downloaded model test; set AI2NPU_RUN_MODEL_TESTS=1");
        return;
    }

    let tokenizer = BgeTokenizer::from_model_dir(BGE_MODEL_DIR).unwrap();
    let encoded = tokenizer.encode("hello world").unwrap();

    assert_eq!(encoded.input_ids.len(), 512);
    assert_eq!(encoded.attention_mask.len(), 512);
    assert!(encoded.attention_mask.contains(&1));
}

#[test]
fn tokenizer_truncates_long_input_to_512_tokens() {
    if !model_tests_enabled() {
        eprintln!("skipping downloaded model test; set AI2NPU_RUN_MODEL_TESTS=1");
        return;
    }

    let tokenizer = BgeTokenizer::from_model_dir(BGE_MODEL_DIR).unwrap();
    let text = std::iter::repeat_n("hello", 2000)
        .collect::<Vec<_>>()
        .join(" ");
    let encoded = tokenizer.encode(&text).unwrap();

    assert_eq!(encoded.input_ids.len(), 512);
    assert_eq!(encoded.attention_mask.len(), 512);
}
