use std::sync::{Arc, Mutex};

use ai2npu::streaming::{ClientEvent, SessionGuard};
use serde_json::json;

#[test]
fn deserializes_realtime_transcription_session_update() {
    let event: ClientEvent = serde_json::from_value(json!({
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "openai/whisper-large-v3-turbo",
                "language": "ru",
                "prompt": "Запиши Открой Telegram"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 400
            },
            "sample_rate": 48000,
            "max_segment_ms": 30000,
            "word_timestamps": true
        }
    }))
    .unwrap();

    let ClientEvent::TranscriptionSessionUpdate { session } = event else {
        panic!("expected transcription_session.update");
    };
    let transcription = session.input_audio_transcription.unwrap();
    let turn_detection = session.turn_detection.unwrap();

    assert_eq!(session.input_audio_format.as_deref(), Some("pcm16"));
    assert_eq!(transcription.model, "openai/whisper-large-v3-turbo");
    assert_eq!(transcription.language.as_deref(), Some("ru"));
    assert_eq!(
        transcription.prompt.as_deref(),
        Some("Запиши Открой Telegram")
    );
    assert_eq!(turn_detection.kind, "server_vad");
    assert_eq!(turn_detection.threshold, Some(0.5));
    assert_eq!(turn_detection.silence_duration_ms, Some(400));
    assert_eq!(session.sample_rate, Some(48000));
    assert_eq!(session.max_segment_ms, Some(30000));
    assert_eq!(session.word_timestamps, Some(true));
}

#[test]
fn deserializes_audio_append_event() {
    let event: ClientEvent = serde_json::from_value(json!({
        "type": "input_audio_buffer.append",
        "audio": "AQIDBA=="
    }))
    .unwrap();

    let ClientEvent::InputAudioBufferAppend { audio } = event else {
        panic!("expected input_audio_buffer.append");
    };

    assert_eq!(audio, "AQIDBA==");
}

#[test]
fn session_guard_allows_only_one_active_session() {
    let active_session = Arc::new(Mutex::new(None));
    let first = SessionGuard::try_acquire(Arc::clone(&active_session), 1).unwrap();

    assert_eq!(*active_session.lock().unwrap(), Some(1));

    let err = SessionGuard::try_acquire(Arc::clone(&active_session), 2)
        .expect_err("second session should be rejected");
    assert_eq!(err.code(), "streaming_busy");

    let error_event = err.into_server_event();
    let json = serde_json::to_value(error_event).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["error"]["code"], "streaming_busy");

    drop(first);
    assert_eq!(*active_session.lock().unwrap(), None);

    let second = SessionGuard::try_acquire(Arc::clone(&active_session), 2).unwrap();
    assert_eq!(*active_session.lock().unwrap(), Some(2));
    drop(second);
    assert_eq!(*active_session.lock().unwrap(), None);
}
