use ai2npu::openvino_c::OpenVinoRuntime;

const BGE_MODEL_DIR: &str = "models/strokinkv/bge-m3-int8-ov";

fn live_npu_tests_enabled() -> bool {
    std::env::var("AI2NPU_RUN_NPU_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn openvino_runtime_lists_npu_device() {
    if !live_npu_tests_enabled() {
        eprintln!("skipping live NPU test; set AI2NPU_RUN_NPU_TESTS=1");
        return;
    }

    let runtime = OpenVinoRuntime::new().unwrap();
    let devices = runtime.available_devices().unwrap();
    assert!(devices.iter().any(|device| device == "NPU"), "{devices:?}");
}

#[test]
fn openvino_compiles_bge_model_on_npu() {
    if !live_npu_tests_enabled() {
        eprintln!("skipping live NPU test; set AI2NPU_RUN_NPU_TESTS=1");
        return;
    }

    let runtime = OpenVinoRuntime::new().unwrap();
    let model = runtime
        .read_model(format!("{BGE_MODEL_DIR}/model.xml"))
        .unwrap();
    let compiled = runtime.compile_model(&model, "NPU").unwrap();

    assert!(compiled
        .output_names()
        .unwrap()
        .iter()
        .any(|name| name == "sentence_embedding"));
}
