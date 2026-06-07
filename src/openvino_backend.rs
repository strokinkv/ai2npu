#[derive(Debug, Clone, serde::Serialize)]
pub struct OpenVinoStatus {
    pub runtime_available: bool,
    pub devices: Vec<String>,
    pub npu_available: bool,
    pub error: Option<String>,
}

impl OpenVinoStatus {
    pub fn detect() -> Self {
        match detect_with_openvino_c() {
            Ok(devices) => {
                let npu_available = devices.iter().any(|device| device == "NPU");
                Self {
                    runtime_available: true,
                    devices,
                    npu_available,
                    error: None,
                }
            }
            Err(error) => Self::unavailable(error.to_string()),
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            runtime_available: false,
            devices: Vec::new(),
            npu_available: false,
            error: Some(message.into()),
        }
    }
}

fn detect_with_openvino_c() -> anyhow::Result<Vec<String>> {
    let runtime = crate::openvino_c::OpenVinoRuntime::new()?;
    runtime.available_devices()
}
