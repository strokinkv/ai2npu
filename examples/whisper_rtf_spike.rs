//! Throwaway latency spike for Task 1.0 of the streaming-dictation plan.
//!
//! Measures the real-time factor (RTF = decode_time / audio_length) of the
//! configured Whisper model on the NPU for short/medium/long phrases. The result
//! gates the streaming window size, the default silence timeout, and whether
//! word-level timestamps are affordable (design spec §8/§14).
//!
//! Run inside the OpenVINO SDK + MSVC environment, e.g.:
//! ```powershell
//! . .\scripts\setup-openvino-sdk.ps1 -SdkRoot "C:\path\to\openvino_sdk"
//! cargo run --release --example whisper_rtf_spike -- <model_dir> <phrase.wav>...
//! ```
//! With no arguments it uses the bundled turbo model and three sample phrases.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ai2npu::audio::{wav_pcm_s16le_as_f32, AudioEndpoint};
use ai2npu::config::{ModelConfig, ModelType};
use ai2npu::inference::{AudioExecutor, AudioInferenceOptions, NativeWhisperExecutor};

const DEFAULT_MODEL_DIR: &str = "models/OpenVINO/whisper-large-v3-turbo-int8-ov";
const DEFAULT_WAVS: &[&str] = &[
    "phrase_2s.wav",
    "phrase_5s.wav",
    "phrase_10s.wav",
];

fn model_config(model_dir: &Path) -> ModelConfig {
    ModelConfig {
        id: "openai/whisper-large-v3-turbo".to_string(),
        model_type: ModelType::Whisper,
        path: model_dir.to_path_buf(),
        enabled: true,
        preload: false,
        queue_timeout_sec: 60,
        normalize: None,
        max_audio_duration_sec: None,
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let wavs: Vec<PathBuf> = {
        let rest: Vec<PathBuf> = args.map(PathBuf::from).collect();
        if rest.is_empty() {
            DEFAULT_WAVS.iter().map(PathBuf::from).collect()
        } else {
            rest
        }
    };

    let model = model_config(&model_dir);
    let executor = NativeWhisperExecutor::new()?;
    let options = AudioInferenceOptions {
        endpoint: AudioEndpoint::Transcriptions,
        language: Some("ru".to_string()),
        prompt: None,
        temperature: None,
        return_timestamps: false,
    };

    // Warm up: the first decode pays one-time session/model-load (NPU compile)
    // cost that must not be charged to the RTF measurement. Use real speech, not
    // silence — decoding pure zeros makes Whisper degenerate into a long
    // repetition loop and never produces a representative warm-up.
    let warmup_wav = wavs
        .first()
        .ok_or_else(|| anyhow::anyhow!("need at least one wav to warm up"))?;
    println!(
        "Loading model from {} and warming up on {}...",
        model_dir.display(),
        warmup_wav.display()
    );
    let warmup = {
        let bytes = std::fs::read(warmup_wav)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", warmup_wav.display()))?;
        wav_pcm_s16le_as_f32(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to decode {}: {e}", warmup_wav.display()))?
    };
    let started = Instant::now();
    executor.transcribe(&model, &warmup, &options)?;
    println!("Warm-up decode took {:.2}s\n", started.elapsed().as_secs_f64());

    println!(
        "{:<16} {:>10} {:>12} {:>8}   transcript",
        "phrase", "audio_s", "decode_s", "RTF"
    );
    println!("{}", "-".repeat(72));

    for wav in &wavs {
        let bytes = std::fs::read(wav)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", wav.display()))?;
        let samples = wav_pcm_s16le_as_f32(&bytes)
            .map_err(|e| anyhow::anyhow!("failed to decode {}: {e}", wav.display()))?;
        let audio_secs = samples.len() as f64 / 16_000.0;

        let started = Instant::now();
        let output = executor.transcribe(&model, &samples, &options)?;
        let decode_secs = started.elapsed().as_secs_f64();
        let rtf = decode_secs / audio_secs;

        let name = wav
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| wav.display().to_string());
        println!(
            "{:<16} {:>10.2} {:>12.2} {:>8.3}   {}",
            name,
            audio_secs,
            decode_secs,
            rtf,
            output.text.trim()
        );
    }

    Ok(())
}
