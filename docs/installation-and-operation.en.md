# Installation and Operation

## Overview

`ai2npu` runs local OpenVINO inference on Intel NPU and exposes OpenAI-like HTTP endpoints on `127.0.0.1:9555`. It runs as the Windows service `ai2npuService`, with configuration and logs stored under `C:\ProgramData\ai2npu`.

## Installation

Run the Inno Setup installer:

```powershell
dist\ai2npu-setup-<version>.exe
```

The installer:

- installs binaries and runtime DLLs to `C:\Program Files\ai2npu`;
- creates `C:\ProgramData\ai2npu`;
- creates `C:\ProgramData\ai2npu\config.toml` if it does not already exist;
- installs and starts the Windows service;
- adds `C:\Program Files\ai2npu` to the system `PATH`;
- optionally installs and synchronizes models selected in the wizard.

Microsoft Visual C++ Redistributable 2015-2022 x64 is required by the native OpenVINO GenAI bridge.

## Model Installation

The installer provides model tasks:

- `BAAI/bge-m3`: downloaded from Hugging Face over HTTPS using a bundled manifest with SHA256 verification.
- `openai/whisper-large-v3-turbo`: downloaded from Hugging Face over HTTPS using a bundled manifest with SHA256 verification.

Models are synchronized from:

```text
https://huggingface.co/strokinkv/bge-m3-int8-ov
https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov
```

to:

```text
C:\ProgramData\ai2npu\models\strokinkv\bge-m3-int8-ov
C:\ProgramData\ai2npu\models\OpenVINO\whisper-large-v3-turbo-int8-ov
```

If the model folder already exists, the installer verifies files by size and SHA256, skips matching files, redownloads damaged or outdated files, and removes stale files from the model directory. Git and Git LFS are not used.

## Configuration

The production config is:

```text
C:\ProgramData\ai2npu\config.toml
```

Key defaults:

```toml
[server]
host = "127.0.0.1"
port = 9555
request_body_limit_mb = 100
thread_count = 16

[queue]
max_pending_requests = 10
default_timeout_sec = 600

[logging]
level = "info"
directory = "C:\\ProgramData\\ai2npu\\logs"
max_file_size_mb = 10
max_files = 10

[streaming]
enabled = true
vad_model_path = "models/silero_vad.onnx"
default_min_silence_ms = 400
default_max_segment_ms = 30000
max_input_buffer_sec = 30
partial_silence_ms = 250
```

Each model is configured with `[[models]]`. Current paths:

```toml
path = "C:/ProgramData/ai2npu/models/strokinkv/bge-m3-int8-ov"
path = "C:/ProgramData/ai2npu/models/OpenVINO/whisper-large-v3-turbo-int8-ov"
```

Loaded model sessions stay alive after first use; run `ai2npu.exe unload` to release resources without stopping the service.

After editing the config, restart the service:

```powershell
ai2npu.exe restart-service
```

## Service Commands

```powershell
ai2npu.exe validate-config
ai2npu.exe check-npu
ai2npu.exe list-models
ai2npu.exe unload
ai2npu.exe start-service
ai2npu.exe stop-service
ai2npu.exe restart-service
ai2npu.exe uninstall-service
```

`ai2npu.exe unload` sends a local request to the running service and unloads all loaded models from memory. If an inference request is active, the command waits for it to finish before releasing resources. The next inference request loads the required model again.

For development:

```powershell
ai2npu.exe run --config config.example.toml
```

## HTTP API

Base URL:

```text
http://127.0.0.1:9555
```

Endpoints:

- `GET /health`
- `GET /v1/models`
- `GET /logs?lines=200`
- `POST /v1/embeddings`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `GET /v1/realtime` (WebSocket)
- `POST /admin/models/unload`

`POST /admin/models/unload` is used by `ai2npu.exe unload`. It is a local administrative operation for releasing model resources without stopping the service.

Embedding example:

```powershell
Invoke-RestMethod http://127.0.0.1:9555/v1/embeddings `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"model":"BAAI/bge-m3","input":"hello"}'
```

Whisper accepts WAV only: mono, 16 kHz, PCM signed 16-bit little-endian.

```powershell
curl.exe http://127.0.0.1:9555/v1/audio/transcriptions `
  -F "model=openai/whisper-large-v3-turbo" `
  -F "language=en" `
  -F "response_format=json" `
  -F "file=@audio.wav"
```

Supported audio `response_format` values:

- `json`
- `text`
- `srt`
- `verbose_json`
- `vtt`

Streaming transcription is available over WebSocket at
`ws://127.0.0.1:9555/v1/realtime`. The server accepts text JSON frames with
base64 PCM16, runs server-side VAD, emits `speech_started/stopped`, append-only
`conversation.item.input_audio_transcription.delta` on VAD micro-pauses, and the
authoritative `conversation.item.input_audio_transcription.completed` final. Only
one streaming session can be active at a time; see `docs/streaming-api.md` for the
full contract.

## How It Works

At startup the service reads `config.toml`, initializes file logging, detects OpenVINO devices, and starts the local HTTP server. Models are loaded lazily on first request. Inference is serialized through a shared FIFO queue so only one model request runs at a time.

The `ai2npu.exe unload` command enqueues the unload operation in the same queue. It does not interrupt an active request; it runs after that request and clears all loaded executor/session caches.

Embeddings use OpenVINO Runtime and the BGE OpenVINO tokenizer/model bundle on `NPU`. Whisper uses a native C++ bridge around OpenVINO GenAI `WhisperPipeline`, also targeting `NPU`. CPU/GPU fallback for model inference is not used.

## Logs

Logs are written to:

```text
C:\ProgramData\ai2npu\logs\ai2npu.log
```

Logs rotate by size during startup according to:

- `logging.max_file_size_mb`
- `logging.max_files`

Recent logs can also be read through:

```text
GET http://127.0.0.1:9555/logs?lines=200
```

## Troubleshooting

Check service and NPU state:

```powershell
ai2npu.exe check-npu
Invoke-RestMethod http://127.0.0.1:9555/health
```

Validate config:

```powershell
ai2npu.exe validate-config
```

If a model was not downloaded or needs refresh, rerun the installer with the model task selected, or run the model installer command manually:

```powershell
& "C:\Program Files\ai2npu\ai2npu.exe" install-model `
  --model "BAAI/bge-m3" `
  --model-dir "C:\ProgramData\ai2npu\models\strokinkv\bge-m3-int8-ov"
```

```powershell
& "C:\Program Files\ai2npu\ai2npu.exe" install-model `
  --model "openai/whisper-large-v3-turbo" `
  --model-dir "C:\ProgramData\ai2npu\models\OpenVINO\whisper-large-v3-turbo-int8-ov"
```
