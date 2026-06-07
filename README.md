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

## Документация

Русская версия:
- [docs/installation-and-operation.md](docs/installation-and-operation.md)
- [docs/development-setup-and-build.md](docs/development-setup-and-build.md)
- [docs/openvino-sdk-setup.md](docs/openvino-sdk-setup.md)

English version:
- [docs/installation-and-operation.en.md](docs/installation-and-operation.en.md)
- [docs/development-setup-and-build.en.md](docs/development-setup-and-build.en.md)
- [docs/openvino-sdk-setup.en.md](docs/openvino-sdk-setup.en.md)

## Разработка

Команды сборки и тестирования для участников проекта описаны в [AGENTS.md](AGENTS.md).

## Лицензия

[MIT](LICENSE)
