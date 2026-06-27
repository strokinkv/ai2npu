# Установка и эксплуатация

## Обзор

`ai2npu` запускает локальный OpenVINO-инференс на Intel NPU и публикует OpenAI-подобные HTTP-эндпоинты на `127.0.0.1:9555`. Сервис работает как Windows-служба `ai2npuService`, а конфигурация и логи хранятся в `C:\ProgramData\ai2npu`.

## Установка

Запустите installer Inno Setup:

```powershell
dist\ai2npu-setup-<version>.exe
```

Installer:

- устанавливает бинарники и runtime DLL в `C:\Program Files\ai2npu`;
- создаёт `C:\ProgramData\ai2npu`;
- создаёт `C:\ProgramData\ai2npu\config.toml`, если файла ещё нет;
- ставит и запускает Windows-службу;
- добавляет `C:\Program Files\ai2npu` в системный `PATH`;
- по выбору загружает и синхронизирует модели из мастера установки.

Для нативного OpenVINO GenAI bridge требуется Microsoft Visual C++ Redistributable 2015-2022 x64.

## Установка моделей

В installer доступны задачи моделей:

- `BAAI/bge-m3`: загружается из Hugging Face по HTTPS через встроенный manifest с SHA256-проверкой файлов.
- `openai/whisper-large-v3-turbo`: загружается из Hugging Face по HTTPS через встроенный manifest с SHA256-проверкой файлов.

Модели синхронизируются из:

```text
https://huggingface.co/strokinkv/bge-m3-int8-ov
https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov
```

в:

```text
C:\ProgramData\ai2npu\models\strokinkv\bge-m3-int8-ov
C:\ProgramData\ai2npu\models\OpenVINO\whisper-large-v3-turbo-int8-ov
```

Если папка модели уже существует, installer проверяет файлы по размеру и SHA256, пропускает совпадающие файлы, заново скачивает повреждённые или устаревшие и удаляет лишние файлы из каталога модели. Git и Git LFS не используются.

## Конфигурация

Рабочий конфиг:

```text
C:\ProgramData\ai2npu\config.toml
```

Ключевые значения по умолчанию:

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

Каждая модель задаётся через `[[models]]`. Текущие пути моделей:

```toml
path = "C:/ProgramData/ai2npu/models/strokinkv/bge-m3-int8-ov"
path = "C:/ProgramData/ai2npu/models/OpenVINO/whisper-large-v3-turbo-int8-ov"
```

Загруженная модель удерживается в памяти после первого обращения; для освобождения ресурсов используйте `ai2npu.exe unload`.

После изменения конфига перезапустите службу:

```powershell
ai2npu.exe restart-service
```

## Команды службы

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

`ai2npu.exe unload` отправляет локальный запрос в работающую службу и выгружает все загруженные модели из памяти. Если сейчас выполняется inference-запрос, команда ждёт его завершения и только потом освобождает ресурсы. Следующий inference-запрос загрузит нужную модель заново.

Для разработки:

```powershell
ai2npu.exe run --config config.example.toml
```

## HTTP API

Base URL:

```text
http://127.0.0.1:9555
```

Эндпоинты:

- `GET /health`
- `GET /v1/models`
- `GET /logs?lines=200`
- `POST /v1/embeddings`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `GET /v1/realtime` (WebSocket)
- `POST /admin/models/unload`

`POST /admin/models/unload` используется командой `ai2npu.exe unload`. Это локальная административная операция для освобождения ресурсов модели без остановки службы.

Пример запроса эмбеддингов:

```powershell
Invoke-RestMethod http://127.0.0.1:9555/v1/embeddings `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"model":"BAAI/bge-m3","input":"hello"}'
```

Whisper принимает только WAV: mono, 16 kHz, PCM signed 16-bit little-endian.

```powershell
curl.exe http://127.0.0.1:9555/v1/audio/transcriptions `
  -F "model=openai/whisper-large-v3-turbo" `
  -F "language=en" `
  -F "response_format=json" `
  -F "file=@audio.wav"
```

Поддерживаемые значения `response_format` для аудио:

- `json`
- `text`
- `srt`
- `verbose_json`
- `vtt`

Потоковая транскрибация доступна по WebSocket `ws://127.0.0.1:9555/v1/realtime`.
Сервер принимает текстовые JSON-кадры с base64 PCM16, использует server-side VAD,
отправляет `speech_started/stopped`, append-only `conversation.item.input_audio_transcription.delta`
при микропаузах и финальный `conversation.item.input_audio_transcription.completed`.
Одна streaming-сессия активна одновременно; подробный контракт описан в
`docs/streaming-api.md`.

## Как это работает

При старте сервис читает `config.toml`, инициализирует файловое логирование, определяет устройства OpenVINO и запускает локальный HTTP-сервер. Модели загружаются лениво по первому запросу. Инференс проходит через общую FIFO-очередь, поэтому одновременно выполняется только один запрос к модели.

Команда `ai2npu.exe unload` ставит операцию выгрузки в эту же очередь. Поэтому она не прерывает активный запрос, а выполняется после него и очищает все загруженные executor/session caches.

Эмбеддинги используют OpenVINO Runtime и BGE OpenVINO tokenizer/model bundle на `NPU`. Whisper работает через нативный C++ bridge поверх OpenVINO GenAI `WhisperPipeline`, тоже с таргетом `NPU`. CPU/GPU fallback для инференса моделей не используется.

## Логи

Логи пишутся в:

```text
C:\ProgramData\ai2npu\logs\ai2npu.log
```

Ротация логов выполняется при старте по параметрам:

- `logging.max_file_size_mb`
- `logging.max_files`

Свежие логи также можно прочитать через:

```text
GET http://127.0.0.1:9555/logs?lines=200
```

## Диагностика

Проверка состояния службы и NPU:

```powershell
ai2npu.exe check-npu
Invoke-RestMethod http://127.0.0.1:9555/health
```

Проверка конфига:

```powershell
ai2npu.exe validate-config
```

Если модель не была загружена или требует обновления, повторно запустите installer с выбранной задачей модели либо вызовите команду установки вручную:

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
