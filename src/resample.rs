use anyhow::{bail, Result};
use rubato::{FftFixedIn, Resampler};

/// Sample rate expected by the Whisper pipeline.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Input frames consumed per resampler chunk. Fixed so the FFT resampler is
/// fully deterministic regardless of the total input length.
const RESAMPLE_CHUNK_FRAMES: usize = 1024;

/// Downmix `channels`-interleaved f32 samples to mono (average of channels),
/// then resample from `input_rate` to 16 kHz. Returns 16 kHz mono f32.
pub fn resample_to_16k_mono(samples: &[f32], input_rate: u32, channels: u16) -> Result<Vec<f32>> {
    if input_rate == 0 {
        bail!("invalid audio: input sample rate must be greater than zero");
    }
    if channels == 0 {
        bail!("invalid audio: channel count must be greater than zero");
    }
    if !samples.len().is_multiple_of(channels as usize) {
        bail!(
            "invalid audio: sample count {} is not a multiple of channel count {}",
            samples.len(),
            channels
        );
    }

    let mono = downmix_to_mono(samples, channels);

    if input_rate == TARGET_SAMPLE_RATE {
        return Ok(mono);
    }

    resample_mono(&mono, input_rate, TARGET_SAMPLE_RATE)
}

/// Average interleaved channels into a single mono frame each.
fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    let channels = channels as usize;
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample a mono signal from `input_rate` to `target_rate` using a
/// deterministic FFT resampler. The final partial chunk is zero-padded and the
/// output is trimmed to the rate-scaled length.
fn resample_mono(mono: &[f32], input_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
    if mono.is_empty() {
        return Ok(Vec::new());
    }

    let mut resampler = FftFixedIn::<f32>::new(
        input_rate as usize,
        target_rate as usize,
        RESAMPLE_CHUNK_FRAMES,
        1,
        1,
    )?;

    let expected = (mono.len() as u64 * u64::from(target_rate) / u64::from(input_rate)) as usize;

    // Keep feeding chunks (zero-padding past the real input) until the resampler
    // has emitted enough frames to cover its internal delay plus the full
    // rate-scaled output, then trim to the exact expected length.
    let mut output = Vec::new();
    let mut pos = 0usize;
    while output.len() < expected {
        let needed = resampler.input_frames_next();

        let mut frame = vec![0f32; needed];
        if pos < mono.len() {
            let take = needed.min(mono.len() - pos);
            frame[..take].copy_from_slice(&mono[pos..pos + take]);
        }
        pos += needed;

        let resampled = resampler.process(&[frame], None)?;
        output.extend_from_slice(&resampled[0]);
    }

    output.truncate(expected);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine(rate: u32, freq: f64, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (2.0 * PI * freq * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    #[test]
    fn downsample_48k_to_16k_length() {
        let frames = 48_000;
        let input = sine(48_000, 220.0, frames);
        let out = resample_to_16k_mono(&input, 48_000, 1).unwrap();

        let expected = frames / 3;
        let diff = (out.len() as i64 - expected as i64).abs();
        assert!(
            diff <= 2,
            "expected ~{} frames, got {}",
            expected,
            out.len()
        );
    }

    #[test]
    fn passthrough_16k_is_identity() {
        let input = sine(16_000, 100.0, 1_600);
        let out = resample_to_16k_mono(&input, 16_000, 1).unwrap();

        assert_eq!(out.len(), input.len());
        assert_eq!(out, input);
    }

    #[test]
    fn stereo_downmix_average_is_exact_on_passthrough() {
        // Interleaved L/R pairs whose averages are exact in binary float.
        let input = vec![0.0, 1.0, -0.5, 0.5, 0.25, 0.75];
        let out = resample_to_16k_mono(&input, 16_000, 2).unwrap();

        assert_eq!(out, vec![0.5, 0.0, 0.5]);
    }

    #[test]
    fn rejects_zero_sample_rate() {
        assert!(resample_to_16k_mono(&[0.0], 0, 1).is_err());
    }

    #[test]
    fn rejects_zero_channels() {
        assert!(resample_to_16k_mono(&[0.0], 16_000, 0).is_err());
    }

    #[test]
    fn rejects_non_multiple_of_channels() {
        // 3 samples cannot be split into stereo frames.
        assert!(resample_to_16k_mono(&[0.0, 1.0, 0.0], 16_000, 2).is_err());
    }

    #[test]
    fn empty_input_returns_empty() {
        let out = resample_to_16k_mono(&[], 48_000, 1).unwrap();
        assert!(out.is_empty());
    }
}
