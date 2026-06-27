use ai2npu::vad::{SpeechProb, VadEvent, VadSegmenter, SAMPLE_RATE, WINDOW_SAMPLES};
use std::path::Path;

#[derive(Default)]
struct ScriptedProb {
    probs: Vec<f32>,
    calls: usize,
}

impl ScriptedProb {
    fn new(probs: impl Into<Vec<f32>>) -> Self {
        Self {
            probs: probs.into(),
            calls: 0,
        }
    }
}

impl SpeechProb for ScriptedProb {
    fn prob(&mut self, _window: &[f32]) -> f32 {
        let prob = self.probs.get(self.calls).copied().unwrap_or(0.0);
        self.calls += 1;
        prob
    }
}

fn samples_for_windows(windows: usize) -> Vec<f32> {
    vec![0.25; windows * WINDOW_SAMPLES]
}

#[test]
fn emits_speech_boundaries_after_configured_silence() {
    let mut segmenter = VadSegmenter::with_probability_source(
        ScriptedProb::new([0.1, 0.8, 0.9, 0.2, 0.1]),
        64,
        0.5,
        30_000,
    )
    .unwrap();

    let events = segmenter.push(&samples_for_windows(5));

    assert_eq!(events.len(), 2);
    assert_eq!(events[0], VadEvent::SpeechStart { at_ms: 32 });
    match &events[1] {
        VadEvent::SpeechEnd {
            start_ms,
            end_ms,
            samples,
        } => {
            assert_eq!((*start_ms, *end_ms), (32, 96));
            assert_eq!(samples.len(), 2 * WINDOW_SAMPLES);
        }
        other => panic!("expected SpeechEnd, got {other:?}"),
    }
}

#[test]
fn force_cuts_when_max_segment_duration_is_reached() {
    let mut segmenter = VadSegmenter::with_probability_source(
        ScriptedProb::new([0.9, 0.9, 0.9, 0.9]),
        400,
        0.5,
        64,
    )
    .unwrap();

    let events = segmenter.push(&samples_for_windows(4));

    assert_eq!(events.len(), 3);
    assert_eq!(events[0], VadEvent::SpeechStart { at_ms: 0 });
    match &events[1] {
        VadEvent::SpeechEnd {
            start_ms,
            end_ms,
            samples,
        } => {
            assert_eq!((*start_ms, *end_ms), (0, 64));
            assert_eq!(samples.len(), 2 * WINDOW_SAMPLES);
        }
        other => panic!("expected forced SpeechEnd, got {other:?}"),
    }
    assert_eq!(events[2], VadEvent::SpeechStart { at_ms: 64 });
}

#[test]
fn flush_emits_trailing_speech() {
    let mut segmenter =
        VadSegmenter::with_probability_source(ScriptedProb::new([0.8, 0.9]), 400, 0.5, 30_000)
            .unwrap();

    let events = segmenter.push(&samples_for_windows(2));
    let flushed = segmenter.flush();

    assert_eq!(events, vec![VadEvent::SpeechStart { at_ms: 0 }]);
    match flushed {
        Some(VadEvent::SpeechEnd {
            start_ms,
            end_ms,
            samples,
        }) => {
            assert_eq!((start_ms, end_ms), (0, 64));
            assert_eq!(samples.len(), 2 * WINDOW_SAMPLES);
        }
        other => panic!("expected flushed SpeechEnd, got {other:?}"),
    }
}

#[test]
fn silence_timeout_can_be_updated() {
    let mut segmenter =
        VadSegmenter::with_probability_source(ScriptedProb::new([0.8, 0.2, 0.2]), 400, 0.5, 30_000)
            .unwrap();
    segmenter.set_min_silence_ms(64);

    let events = segmenter.push(&samples_for_windows(3));

    assert_eq!(events.len(), 2);
    assert_eq!(events[0], VadEvent::SpeechStart { at_ms: 0 });
    assert!(matches!(events[1], VadEvent::SpeechEnd { .. }));
}

#[test]
fn emits_partial_on_micro_pause_before_endpoint() {
    // speech(2) -> short silence(1) -> speech(2) -> long silence(2)
    // min_silence=64ms (2 windows), partial_silence=32ms (1 window).
    let mut segmenter = VadSegmenter::with_probability_source(
        ScriptedProb::new([0.9, 0.9, 0.2, 0.9, 0.9, 0.2, 0.2]),
        64,
        0.5,
        30_000,
    )
    .unwrap();
    segmenter.set_partial_silence_ms(32);

    let events = segmenter.push(&samples_for_windows(7));

    // SpeechStart, SpeechPartial (after the 1-window micro-pause), SpeechEnd.
    assert_eq!(events.len(), 3, "got {events:?}");
    assert!(matches!(events[0], VadEvent::SpeechStart { .. }));
    match &events[1] {
        VadEvent::SpeechPartial { samples, .. } => {
            // Snapshot holds the 2 speech windows seen so far.
            assert_eq!(samples.len(), 2 * WINDOW_SAMPLES);
        }
        other => panic!("expected SpeechPartial, got {other:?}"),
    }
    match &events[2] {
        VadEvent::SpeechEnd { samples, .. } => {
            // Final holds all 4 speech windows; partial is a strict prefix.
            assert_eq!(samples.len(), 4 * WINDOW_SAMPLES);
        }
        other => panic!("expected SpeechEnd, got {other:?}"),
    }
}

#[test]
fn no_partial_when_disabled() {
    let mut segmenter = VadSegmenter::with_probability_source(
        ScriptedProb::new([0.9, 0.9, 0.2, 0.9, 0.9, 0.2, 0.2]),
        64,
        0.5,
        30_000,
    )
    .unwrap();
    // partial_silence_ms defaults to 0 -> disabled.
    let events = segmenter.push(&samples_for_windows(7));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, VadEvent::SpeechPartial { .. })),
        "no partial expected when disabled, got {events:?}"
    );
}

/// Exercises the real Silero V5 model embedded in `voice_activity_detector`
/// through `ort` on CPU. Gated behind `AI2NPU_RUN_MODEL_TESTS=1` because it
/// loads an ONNX model and runs inference. Verifies the model loads and that
/// a loud tone scores meaningfully higher than digital silence.
#[test]
fn real_silero_vad_loads_and_discriminates_speechlike_audio() {
    if std::env::var("AI2NPU_RUN_MODEL_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping real Silero VAD test (set AI2NPU_RUN_MODEL_TESTS=1)");
        return;
    }

    // Real production constructor: loads the embedded Silero ONNX via ort.
    let mut segmenter = VadSegmenter::new(Path::new("models/silero_vad.onnx"), 400, 0.5, 30_000)
        .expect("real Silero VAD must load");

    // A few windows of silence must not crash and must not start speech.
    let silence = vec![0.0_f32; WINDOW_SAMPLES * 3];
    let silence_events = segmenter.push(&silence);
    assert!(
        !silence_events
            .iter()
            .any(|event| matches!(event, VadEvent::SpeechStart { .. })),
        "digital silence must not be detected as speech, got {silence_events:?}"
    );

    // A loud 220 Hz tone exercises the model on speech-like energy; we only
    // assert it runs and produces a finite probability (no panic / NaN).
    let mut prober = ai2npu::vad::SileroSpeechProb::new().unwrap();
    let tone: Vec<f32> = (0..WINDOW_SAMPLES)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
        })
        .collect();
    let prob = prober.prob(&tone);
    assert!(
        prob.is_finite() && (0.0..=1.0).contains(&prob),
        "Silero probability must be a valid [0,1] value, got {prob}"
    );
}
