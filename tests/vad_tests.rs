use ai2npu::vad::{SpeechProb, VadEvent, VadSegmenter, WINDOW_SAMPLES};

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
