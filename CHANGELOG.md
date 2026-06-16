# Changelog

Все заметные изменения проекта документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
проект придерживается [Semantic Versioning](https://semver.org/lang/ru/).

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

[1.1.0]: https://github.com/strokinkv/ai2npu/releases/tag/v1.1.0
[1.0.1]: https://github.com/strokinkv/ai2npu/releases/tag/v1.0.1
[1.0.0]: https://github.com/strokinkv/ai2npu/releases/tag/v1.0.0
