use ai2npu::audio::{
    is_effectively_empty_audio, validate_wav, wav_pcm_s16le_as_f32, AudioEndpoint, WavInfo,
};
use ai2npu::config::AppConfig;
use ai2npu::inference::{
    audio_executor_kind_from_env, AudioExecutor, AudioExecutorKind, AudioInferenceOptions,
    NativeWhisperExecutor,
};
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

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

#[test]
fn accepts_pcm_mono_16khz_s16le_wav() {
    let wav = wav_header(1, 16_000, 16, 1, 32_000);

    let info = validate_wav(&wav, 30).unwrap();

    assert_eq!(
        info,
        WavInfo {
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
            duration_sec: 1.0
        }
    );
}

#[test]
fn converts_pcm_mono_wav_samples_to_normalized_f32() {
    let mut wav = wav_header(1, 16_000, 16, 1, 4);
    let data_start = wav.len() - 4;
    wav[data_start..data_start + 2].copy_from_slice(&i16::MIN.to_le_bytes());
    wav[data_start + 2..data_start + 4].copy_from_slice(&i16::MAX.to_le_bytes());

    let samples = wav_pcm_s16le_as_f32(&wav).unwrap();

    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0], -1.0);
    assert!((samples[1] - 0.9999695).abs() < 0.000001);
}

#[test]
fn detects_short_audio_as_effectively_empty() {
    let wav = wav_header(1, 16_000, 16, 1, 1_600);
    let info = validate_wav(&wav, 30).unwrap();

    assert!(is_effectively_empty_audio(&wav, &info).unwrap());
}

#[test]
fn detects_silent_audio_as_effectively_empty() {
    let wav = wav_header(1, 16_000, 16, 1, 32_000);
    let info = validate_wav(&wav, 30).unwrap();

    assert!(is_effectively_empty_audio(&wav, &info).unwrap());
}

#[test]
fn does_not_treat_audible_audio_as_effectively_empty() {
    let mut wav = wav_header(1, 16_000, 16, 1, 32_000);
    let data_start = wav.len() - 32_000;
    for sample in wav[data_start..].chunks_exact_mut(2) {
        sample.copy_from_slice(&1_000i16.to_le_bytes());
    }
    let info = validate_wav(&wav, 30).unwrap();

    assert!(!is_effectively_empty_audio(&wav, &info).unwrap());
}

#[test]
fn rejects_stereo_wav() {
    let wav = wav_header(2, 16_000, 16, 1, 64_000);

    let err = validate_wav(&wav, 30).unwrap_err().to_string();

    assert!(err.contains("mono"));
}

#[test]
fn rejects_non_16khz_wav() {
    let wav = wav_header(1, 44_100, 16, 1, 88_200);

    let err = validate_wav(&wav, 30).unwrap_err().to_string();

    assert!(err.contains("16 kHz"));
}

#[test]
fn rejects_non_pcm_wav() {
    let wav = wav_header(1, 16_000, 16, 3, 32_000);

    let err = validate_wav(&wav, 30).unwrap_err().to_string();

    assert!(err.contains("PCM"));
}

#[test]
fn rejects_audio_longer_than_limit() {
    let wav = wav_header(1, 16_000, 16, 1, 32_000 * 31);

    let err = validate_wav(&wav, 30).unwrap_err().to_string();

    assert!(err.contains("duration"));
}

#[test]
fn native_whisper_executor_returns_text_when_enabled() {
    if std::env::var("AI2NPU_RUN_NATIVE_GENAI_TESTS")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skipping live native GenAI test; set AI2NPU_RUN_NATIVE_GENAI_TESTS=1");
        return;
    }

    let cfg: AppConfig =
        toml::from_str(&std::fs::read_to_string("config.example.toml").unwrap()).unwrap();
    let model = cfg
        .models
        .iter()
        .find(|model| model.id == "openai/whisper-large-v3-turbo")
        .unwrap();
    let wav = wav_header(1, 16_000, 16, 1, 32_000);
    let samples = wav_pcm_s16le_as_f32(&wav).unwrap();
    let executor = NativeWhisperExecutor::new().unwrap();
    let output = executor
        .transcribe(
            model,
            &samples,
            &AudioInferenceOptions {
                endpoint: AudioEndpoint::Transcriptions,
                language: None,
                prompt: None,
                temperature: None,
                return_timestamps: false,
            },
        )
        .unwrap();

    assert!(output.duration > 0.0);
}

#[test]
fn default_audio_executor_is_native_genai() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::remove_var("AI2NPU_AUDIO_EXECUTOR");

    assert_eq!(
        audio_executor_kind_from_env().unwrap(),
        AudioExecutorKind::NativeGenAi
    );
}
