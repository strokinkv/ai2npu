use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

const MIN_TRANSCRIBABLE_DURATION_SEC: f64 = 0.30;
const SILENCE_RMS_THRESHOLD: f64 = 0.001;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioEndpoint {
    Transcriptions,
    Translations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioResponseFormat {
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}

impl AudioResponseFormat {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("json") {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "srt" => Ok(Self::Srt),
            "verbose_json" => Ok(Self::VerboseJson),
            "vtt" => Ok(Self::Vtt),
            value => bail!("unsupported response_format: {value}"),
        }
    }
}

#[derive(Debug, Default)]
pub struct AudioMultipartRequest {
    pub model: Option<String>,
    pub file: Option<Vec<u8>>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
    pub temperature: Option<String>,
    pub timestamp_granularities: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct AudioJsonResponse {
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AudioSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AudioOutput {
    pub text: String,
    pub language: Option<String>,
    pub duration: f64,
    #[serde(default)]
    pub segments: Vec<AudioSegment>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub duration_sec: f64,
}

pub fn validate_wav(bytes: &[u8], max_duration_sec: u64) -> Result<WavInfo> {
    let parsed = parse_wav(bytes)?;
    let fmt = parsed.fmt;
    let data_len = parsed.data.len();

    if fmt.audio_format != 1 {
        bail!("invalid WAV: only PCM format tag 1 is supported");
    }
    if fmt.channels != 1 {
        bail!("invalid WAV: audio must be mono");
    }
    if fmt.sample_rate != 16_000 {
        bail!("invalid WAV: sample rate must be 16 kHz");
    }
    if fmt.bits_per_sample != 16 {
        bail!("invalid WAV: sample format must be 16-bit signed little-endian");
    }
    if fmt.block_align != 2 {
        bail!("invalid WAV: block align must be 2 bytes for mono s16le");
    }

    let duration_sec = data_len as f64 / f64::from(fmt.sample_rate * u32::from(fmt.block_align));
    if duration_sec > max_duration_sec as f64 {
        bail!(
            "invalid WAV: duration {:.3}s exceeds limit {}s",
            duration_sec,
            max_duration_sec
        );
    }

    Ok(WavInfo {
        sample_rate: fmt.sample_rate,
        channels: fmt.channels,
        bits_per_sample: fmt.bits_per_sample,
        duration_sec,
    })
}

pub fn wav_pcm_s16le_as_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let parsed = parse_wav(bytes)?;
    let fmt = parsed.fmt;
    if fmt.audio_format != 1
        || fmt.channels != 1
        || fmt.sample_rate != 16_000
        || fmt.bits_per_sample != 16
        || fmt.block_align != 2
    {
        bail!("invalid WAV: expected PCM mono 16 kHz s16le audio");
    }
    if parsed.data.len() % 2 != 0 {
        bail!("invalid WAV: data chunk has odd byte length for s16le audio");
    }

    Ok(parsed
        .data
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
        .collect())
}

pub fn is_effectively_empty_audio(bytes: &[u8], info: &WavInfo) -> Result<bool> {
    if info.duration_sec < MIN_TRANSCRIBABLE_DURATION_SEC {
        return Ok(true);
    }

    let samples = wav_pcm_s16le_as_f32(bytes)?;
    if samples.is_empty() {
        return Ok(true);
    }

    let mean_square = samples
        .iter()
        .map(|sample| f64::from(*sample).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    Ok(mean_square.sqrt() < SILENCE_RMS_THRESHOLD)
}

struct ParsedWav<'a> {
    fmt: FmtChunk,
    data: &'a [u8],
}

fn parse_wav(bytes: &[u8]) -> Result<ParsedWav<'_>> {
    if bytes.len() < 12 {
        bail!("invalid WAV: file is too small");
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("invalid WAV: expected RIFF/WAVE header");
    }

    let mut offset = 12usize;
    let mut fmt = None;
    let mut data = None;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_len = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start
            .checked_add(chunk_len)
            .ok_or_else(|| anyhow::anyhow!("invalid WAV: chunk size overflow"))?;
        if chunk_end > bytes.len() {
            bail!("invalid WAV: chunk exceeds file size");
        }

        match chunk_id {
            b"fmt " => fmt = Some(parse_fmt_chunk(&bytes[chunk_start..chunk_end])?),
            b"data" => data = Some(&bytes[chunk_start..chunk_end]),
            _ => {}
        }

        offset = chunk_end + (chunk_len % 2);
    }

    let fmt = fmt.ok_or_else(|| anyhow::anyhow!("invalid WAV: missing fmt chunk"))?;
    let data = data.ok_or_else(|| anyhow::anyhow!("invalid WAV: missing data chunk"))?;
    Ok(ParsedWav { fmt, data })
}

#[derive(Debug, Clone, Copy)]
struct FmtChunk {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

fn parse_fmt_chunk(bytes: &[u8]) -> Result<FmtChunk> {
    if bytes.len() < 16 {
        bail!("invalid WAV: fmt chunk is too small");
    }

    Ok(FmtChunk {
        audio_format: u16::from_le_bytes(bytes[0..2].try_into()?),
        channels: u16::from_le_bytes(bytes[2..4].try_into()?),
        sample_rate: u32::from_le_bytes(bytes[4..8].try_into()?),
        block_align: u16::from_le_bytes(bytes[12..14].try_into()?),
        bits_per_sample: u16::from_le_bytes(bytes[14..16].try_into()?),
    })
}
