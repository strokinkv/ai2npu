# ai2npu

`ai2npu` — локальная Windows-служба для OpenVINO-инференса на Intel NPU с OpenAI-совместимым HTTP API.

Проект закрывает три основных сценария:

- транскрибация и перевод аудио через Whisper;
- получение эмбеддингов через BGE-M3;
- локальные эндпоинты для проверки здоровья, списка моделей и логов.

Служба слушает `127.0.0.1:9555`, использует один TOML-конфиг и предназначена для работы как `ai2npuService` на Windows 11 с поддержкой Intel NPU.

## Модели

- `BAAI/bge-m3` для `/v1/embeddings`.
- `openai/whisper-large-v3-turbo` для `/v1/audio/transcriptions` и `/v1/audio/translations`.

BGE и Whisper OpenVINO-модели синхронизируются во время установки из Hugging Face:

```text
C:\ProgramData\ai2npu\models\strokinkv\bge-m3-int8-ov
C:\ProgramData\ai2npu\models\OpenVINO\whisper-large-v3-turbo-int8-ov
```

Сами файлы моделей не входят в GitHub-репозиторий и не коммитятся. Локальная папка `models/` используется только для разработки и тестов.

## Установка и проверка

Готовый installer публикуется в GitHub Releases как `ai2npu-setup-<version>.exe`. Для версии `1.0.1` ожидаемый файл: `ai2npu-setup-1.0.1.exe`.

После установки проверьте службу:

```powershell
& "C:\Program Files\ai2npu\ai2npu.exe" version
curl.exe http://127.0.0.1:9555/health
curl.exe http://127.0.0.1:9555/v1/models
```

Для освобождения ресурсов без остановки службы:

```powershell
& "C:\Program Files\ai2npu\ai2npu.exe" unload
```

Проверка embeddings из PowerShell:

```powershell
$response = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:9555/v1/embeddings" `
  -ContentType "application/json" `
  -Body (@{ model="BAAI/bge-m3"; input="test text" } | ConvertTo-Json -Compress)
$response.data[0].embedding.Count
```

Ожидаемый размер embedding для BGE-M3: `1024`.

> **Лимит длины входа.** NPU-модель имеет статическую форму входа в **512 токенов**.
> Хотя BGE-M3 поддерживает до 8192 токенов, более длинный текст молча усекается до 512.
> При усечении в лог пишется предупреждение (`embedding input truncated ...`). Для длинных
> документов разбивайте текст на части на стороне клиента.

## API эндпоинты

Служба слушает `http://127.0.0.1:9555` и предоставляет OpenAI-совместимые и служебные эндпоинты:

- `POST /v1/embeddings` — эмбеддинги BGE-M3.
- `POST /v1/audio/transcriptions` — транскрибация Whisper (форматы ответа: `json`, `verbose_json`, `text`, `srt`, `vtt`).
- `POST /v1/audio/translations` — перевод аудио в английский текст.
- `GET /v1/models` — список включённых моделей.
- `GET /health` — статус службы, NPU/OpenVINO, загруженные модели и очередь.
- `GET /logs?lines=N` — последние строки лог-файла (по умолчанию 200).
- `POST /admin/models/unload` — выгрузка загруженных моделей без остановки службы (см. `ai2npu unload`).

Все эндпоинты доступны только с loopback-адреса и без аутентификации, что рассчитано на локальное использование.

## Ограничения

- Нужны Windows 11, Intel NPU driver и доступный OpenVINO NPU device.
- При первой установке нужен интернет: модели загружаются из Hugging Face по HTTPS.
- Первый Whisper-запрос может быть медленнее из-за cold start.
- GitHub Actions проверяет сборку и installer, но не выполняет live NPU tests на hosted runner.

## Документация

Русская версия:
- [docs/installation-and-operation.md](docs/installation-and-operation.md)
- [docs/development-setup-and-build.md](docs/development-setup-and-build.md)
- [docs/openvino-sdk-setup.md](docs/openvino-sdk-setup.md)

English version:
- [docs/installation-and-operation.en.md](docs/installation-and-operation.en.md)
- [docs/development-setup-and-build.en.md](docs/development-setup-and-build.en.md)
- [docs/openvino-sdk-setup.en.md](docs/openvino-sdk-setup.en.md)

## Изменения

История версий — в [CHANGELOG.md](CHANGELOG.md).

## Разработка

Команды сборки и тестирования для участников проекта описаны в [AGENTS.md](AGENTS.md).

## Лицензия

[MIT](LICENSE)
