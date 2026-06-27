use std::path::Path;

use anyhow::{bail, Result};
use voice_activity_detector::VoiceActivityDetector;

/// Sample rate expected by Silero VAD V5.
pub const SAMPLE_RATE: u32 = 16_000;
/// Window size required by Silero VAD V5 at 16 kHz.
pub const WINDOW_SAMPLES: usize = 512;
/// Duration of one VAD window at 16 kHz.
pub const WINDOW_MS: u64 = 32;

/// Source of speech probabilities for fixed-size VAD windows.
pub trait SpeechProb {
    fn prob(&mut self, window: &[f32]) -> f32;
}

/// Silero V5 probability source backed by `voice_activity_detector`.
pub struct SileroSpeechProb {
    detector: VoiceActivityDetector,
}

impl SileroSpeechProb {
    pub fn new() -> Result<Self> {
        let detector = VoiceActivityDetector::builder()
            .sample_rate(i64::from(SAMPLE_RATE))
            .chunk_size(WINDOW_SAMPLES)
            .build()?;
        Ok(Self { detector })
    }
}

impl SpeechProb for SileroSpeechProb {
    fn prob(&mut self, window: &[f32]) -> f32 {
        self.detector.predict(window.iter().copied())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    SpeechStart {
        at_ms: u64,
    },
    SpeechPartial {
        start_ms: u64,
        at_ms: u64,
        samples: Vec<f32>,
    },
    SpeechEnd {
        start_ms: u64,
        end_ms: u64,
        samples: Vec<f32>,
    },
}

pub struct VadSegmenter<P = SileroSpeechProb> {
    prob: P,
    min_silence_ms: u64,
    threshold: f32,
    max_segment_ms: u64,
    pending: Vec<f32>,
    in_speech: bool,
    speech_start_ms: u64,
    last_speech_end_ms: u64,
    current_ms: u64,
    silence_ms: u64,
    partial_silence_ms: u64,
    partial_emitted_for_pause: bool,
    speech_samples: Vec<f32>,
}

impl VadSegmenter<SileroSpeechProb> {
    /// Build the production Silero-backed segmenter.
    ///
    /// `voice_activity_detector` 0.2.1 embeds the Silero V5 ONNX model, so the
    /// path is accepted to keep the public API aligned with streaming config.
    pub fn new(
        _model_path: &Path,
        min_silence_ms: u64,
        threshold: f32,
        max_segment_ms: u64,
    ) -> Result<Self> {
        Self::with_probability_source(
            SileroSpeechProb::new()?,
            min_silence_ms,
            threshold,
            max_segment_ms,
        )
    }
}

impl<P: SpeechProb> VadSegmenter<P> {
    pub fn with_probability_source(
        prob: P,
        min_silence_ms: u64,
        threshold: f32,
        max_segment_ms: u64,
    ) -> Result<Self> {
        validate_config(min_silence_ms, threshold, max_segment_ms)?;
        Ok(Self {
            prob,
            min_silence_ms,
            threshold,
            max_segment_ms,
            pending: Vec::new(),
            in_speech: false,
            speech_start_ms: 0,
            last_speech_end_ms: 0,
            current_ms: 0,
            silence_ms: 0,
            partial_silence_ms: 0,
            partial_emitted_for_pause: false,
            speech_samples: Vec::new(),
        })
    }

    pub fn push(&mut self, samples_16k: &[f32]) -> Vec<VadEvent> {
        self.pending.extend_from_slice(samples_16k);

        let complete_windows = self.pending.len() / WINDOW_SAMPLES;
        if complete_windows == 0 {
            return Vec::new();
        }

        let process_len = complete_windows * WINDOW_SAMPLES;
        let ready: Vec<f32> = self.pending.drain(..process_len).collect();
        let mut events = Vec::new();
        for window in ready.chunks_exact(WINDOW_SAMPLES) {
            self.process_window(window, &mut events);
        }
        events
    }

    pub fn flush(&mut self) -> Option<VadEvent> {
        self.pending.clear();
        self.finish_segment()
    }

    pub fn set_min_silence_ms(&mut self, ms: u64) {
        self.min_silence_ms = ms;
    }

    pub fn set_partial_silence_ms(&mut self, ms: u64) {
        self.partial_silence_ms = ms;
    }

    fn process_window(&mut self, window: &[f32], events: &mut Vec<VadEvent>) {
        let is_speech = self.prob.prob(window) >= self.threshold;

        if is_speech && self.should_force_cut_before_next_speech_window() {
            if let Some(event) = self.finish_segment() {
                events.push(event);
            }
        }

        if is_speech {
            if !self.in_speech {
                self.start_segment(events);
            } else if self.partials_enabled()
                && !self.partial_emitted_for_pause
                && self.silence_ms >= self.partial_silence_ms
                && self.silence_ms < self.min_silence_ms
                && !self.speech_samples.is_empty()
            {
                self.partial_emitted_for_pause = true;
                events.push(VadEvent::SpeechPartial {
                    start_ms: self.speech_start_ms,
                    at_ms: self.current_ms - self.silence_ms + self.partial_silence_ms,
                    samples: self.speech_samples.clone(),
                });
            }
            self.silence_ms = 0;
            self.partial_emitted_for_pause = false;
            self.speech_samples.extend_from_slice(window);
            self.last_speech_end_ms = self.current_ms + WINDOW_MS;
        } else if self.in_speech {
            self.silence_ms += WINDOW_MS;
            if self.silence_ms >= self.min_silence_ms {
                if let Some(event) = self.finish_segment() {
                    events.push(event);
                }
            }
        }

        self.current_ms += WINDOW_MS;
    }

    fn should_force_cut_before_next_speech_window(&self) -> bool {
        self.in_speech
            && self.current_ms.saturating_sub(self.speech_start_ms) >= self.max_segment_ms
    }

    fn partials_enabled(&self) -> bool {
        self.partial_silence_ms > 0 && self.partial_silence_ms < self.min_silence_ms
    }

    fn start_segment(&mut self, events: &mut Vec<VadEvent>) {
        self.in_speech = true;
        self.speech_start_ms = self.current_ms;
        self.last_speech_end_ms = self.current_ms;
        self.silence_ms = 0;
        self.partial_emitted_for_pause = false;
        self.speech_samples.clear();
        events.push(VadEvent::SpeechStart {
            at_ms: self.speech_start_ms,
        });
    }

    fn finish_segment(&mut self) -> Option<VadEvent> {
        if !self.in_speech {
            return None;
        }

        self.in_speech = false;
        self.silence_ms = 0;
        Some(VadEvent::SpeechEnd {
            start_ms: self.speech_start_ms,
            end_ms: self.last_speech_end_ms,
            samples: std::mem::take(&mut self.speech_samples),
        })
    }
}

fn validate_config(min_silence_ms: u64, threshold: f32, max_segment_ms: u64) -> Result<()> {
    if min_silence_ms == 0 {
        bail!("VAD min_silence_ms must be greater than zero");
    }
    if !(0.0..=1.0).contains(&threshold) {
        bail!("VAD threshold must be between 0.0 and 1.0");
    }
    if max_segment_ms == 0 {
        bail!("VAD max_segment_ms must be greater than zero");
    }
    Ok(())
}
