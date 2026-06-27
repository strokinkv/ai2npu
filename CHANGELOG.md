# Changelog

Все заметные изменения проекта документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
проект придерживается [Semantic Versioning](https://semver.org/lang/ru/).

## [1.2.0] - 2026-06-27

### Добавлено
- WebSocket `/v1/realtime` для потоковой ASR-транскрибации в формате подмножества OpenAI Realtime.
- Server-side VAD на Silero V5 с событиями `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped` и `input_audio_buffer.committed`.
- Промежуточные append-only события `conversation.item.input_audio_transcription.delta` по микропаузам VAD.
- Поле `content_index: 0` в событиях `conversation.item.input_audio_transcription.delta` и `conversation.item.input_audio_transcription.completed`.
- Секция `[streaming]` в конфиге: включение `/v1/realtime`, длительность тишины, максимальная длина сегмента, размер входного буфера и `partial_silence_ms`.
- `streaming.partial_silence_ms`: `0` выключает partial-результаты, ненулевое значение включает VAD micro-pause partials.
- `transcription_session.update`, `input_audio_buffer.append` и `input_audio_buffer.commit` для Realtime-сессий.
- Опциональные `word_timestamps` в протоколе Realtime; поле `words` появляется в `completed`, когда таймстампы включены и backend вернул слова.
- 16 kHz mono ресэмплер для входного PCM16 в streaming ASR.
- Кооперативная отмена streaming-сессии при закрытии WebSocket.
- Gated live NPU smoke-тест для `/v1/realtime` (`AI2NPU_RUN_NPU_TESTS=1`, `AI2NPU_SMOKE_WAV=<16-bit PCM WAV>`).

### Изменено
- Whisper в дефолтном конфиге прогревается при старте службы (`preload = true`) и ждёт inference-очередь без таймаута (`queue_timeout_sec = 0`), чтобы Marvin получал постоянно прогретый ASR.
- `queue_timeout_sec = 0` теперь валидное значение и означает ожидание очереди без таймера.
- OpenVINO GenAI SDK обновлён до `2026.2.0.0`.
- В дистрибутив добавлен совместимый `onnxruntime.dll` для запуска VAD.
- BGE предупреждает о truncation входа до статического лимита NPU-модели в 512 токенов.
- Ротация логов стала потокобезопасной.
- Лимит размера HTTP body применяется и к streamed requests.
- Очередь инференса корректно считает running job в `pending_len`.
- Streaming decoder не зависает на втором сегменте: `initial_prompt` и `word_timestamps` нейтрализуются на NPU-границе из-за текущих ограничений OpenVINO GenAI `WhisperPipeline`.

### Документация
- Добавлена документация Realtime-контракта: `docs/streaming-api.md`.
- Обновлены README, Marvin-контекст и operation docs под текущий streaming ASR, partial `delta`, `content_index`, `partial_silence_ms` и always-warm Whisper.

### Ограничения
- В default/CI тестах live NPU smoke самопропускается без `AI2NPU_RUN_NPU_TESTS=1` и `AI2NPU_SMOKE_WAV`.
- Пословные таймстампы в Realtime-протоколе подготовлены, но на NPU пока могут быть пустыми из-за ограничений одного общего `WhisperPipeline`.

## [1.1.0] - 2026-06-16

### Добавлено
- Форматы ответа `srt` и `vtt` для `/v1/audio/transcriptions` и `/v1/audio/translations` (субтитры с таймкодами из сегментов).
- Параметр `temperature` (`0.0`–`1.0`) для Whisper: передаётся в generation config пайплайна; при выходе за диапазон возвращается `400 Bad Request`.
- `server.thread_count` теперь реально задаёт число worker-потоков runtime.
- `validate-config` печатает статус бандлов моделей (наличие требуемых файлов).
- Возвращаются документированные коды ошибок `npu_unavailable`, `openvino_unavailable`, `model_load_failed`, `inference_failed`.

### Изменено
- Токенизатор BGE кэшируется рядом со скомпилированной моделью (раньше `tokenizer.json` перечитывался на каждый запрос embeddings).
- WAV парсится один раз за запрос (раньше до трёх раз).
- `/health` для embedding-моделей в `loaded_models` возвращает `id` модели, а не путь к файлу.
- Лог ротируется в рантайме по размеру, а не только при старте службы.
- Ошибки preload пробрасываются вместо паники; отравлённые mutex восстанавливаются вместо паники.

### Удалено
- Поле конфигурации `idle_timeout_sec` (ранее требовалось значение `0`). Существующие конфиги с этим полем продолжают загружаться — поле игнорируется.

### CI
- Release-workflow проверяет версию пакета в `Cargo.lock`.
- `cargo clippy` запускается с `--all-targets` (ловит линты и в тестах).

## [1.0.1]

### Добавлено
- Команда `ai2npu unload` и эндпоинт `POST /admin/models/unload` для выгрузки загруженных моделей без остановки службы.

## [1.0.0]

### Добавлено
- Первый релиз: локальная Windows-служба с OpenAI-совместимым API для Whisper transcription/translation и BGE-M3 embeddings через OpenVINO на Intel NPU.
- Эндпоинты `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/embeddings`, `/v1/models`, `/health`, `/logs`.
- Установщик на Inno Setup, загрузка моделей через `ai2npu install-model`.

[1.2.0]: https://github.com/strokinkv/ai2npu/releases/tag/v1.2.0
[1.1.0]: https://github.com/strokinkv/ai2npu/releases/tag/v1.1.0
[1.0.1]: https://github.com/strokinkv/ai2npu/releases/tag/v1.0.1
[1.0.0]: https://github.com/strokinkv/ai2npu/releases/tag/v1.0.0
