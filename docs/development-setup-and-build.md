# Подготовка среды разработки и сборка

## Что нужно установить

Для разработки на Windows нужны:

- Python не нужен;
- Rust toolchain `stable-x86_64-pc-windows-msvc`;
- Visual Studio Build Tools 2022 с workload `Desktop development with C++`;
- Windows 10/11 SDK;
- MSYS2, если требуется GNU-сборка или вспомогательные Unix-утилиты;
- Inno Setup 6 для сборки installer;
- Доступ к интернету для загрузки BGE и Whisper моделей из Hugging Face по HTTPS.

Для OpenVINO-зависимостей используйте архив SDK:

```powershell
$env:OPENVINO_SDK_ROOT = "C:\path\to\openvino_sdk"
$env:OPENVINO_LIB_DIR = "$env:OPENVINO_SDK_ROOT\runtime\lib\intel64\Release"
$env:PATH = "$env:OPENVINO_SDK_ROOT\runtime\bin\intel64\Release;$env:OPENVINO_SDK_ROOT\runtime\3rdparty\tbb\bin;$env:PATH"
```

## Подготовка окружения

1. Убедитесь, что `rustup`, `cargo`, `link.exe`, `cl.exe` и `ISCC.exe` доступны.
2. Проверьте, что установлен Intel NPU driver и устройство видно в системе.

Если нужен локальный Windows build, работайте через `VsDevCmd.bat`, чтобы подхватились `LIB`, `INCLUDE` и SDK libraries.

## Проверка сборки

Сначала проверьте код и тесты:

```powershell
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo fmt"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo test"
```

Обычный `cargo test` не требует скачанных моделей и живого NPU. Проверки, зависящие от локальной папки `models/`, включаются отдельно:

```powershell
$env:AI2NPU_RUN_MODEL_TESTS = "1"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo test"
```

Живые NPU-проверки включаются отдельно:

```powershell
$env:AI2NPU_RUN_NPU_TESTS = "1"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo test"
```

Для запуска сервиса из исходников:

```powershell
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo run -- run --config config.example.toml"
```

## Сборка дистрибутива

Сборка дистрибутива идёт через `packaging\prepare-dist.ps1`:

```powershell
$env:LIB = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64;$env:LIB"
rustup run stable-x86_64-pc-windows-msvc powershell -ExecutionPolicy Bypass -File packaging\prepare-dist.ps1
```

Скрипт:

- собирает `ai2npu.exe`;
- копирует OpenVINO runtime DLL;
- использует `ai2npu.exe install-model` для HTTPS-загрузки BGE и Whisper моделей;
- не включает Git, Git LFS или отдельный downloader-скрипт.

Папки `dist/`, `target/`, `build/`, `tools/` и `models/` являются локальными артефактами и не публикуются в GitHub.

GitHub Actions скачивает официальный Windows archive OpenVINO GenAI из OpenVINO storage и кэширует распакованный SDK в `tools/` для следующих запусков. У hosted runner нет Intel NPU, поэтому model/live NPU tests остаются opt-in и не включаются в CI.

Installer собирается через Inno Setup:

```powershell
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" packaging\ai2npu.iss
```

Результат появляется в `dist\ai2npu-setup-<version>.exe`.

## GitHub Release

Автоматический release запускается при push тега вида `vMAJOR.MINOR.PATCH`, например:

```powershell
git tag -a v1.0.1 -m "ai2npu v1.0.1"
git push origin v1.0.1
```

Перед созданием тега обновите версию в `Cargo.toml` и `packaging/ai2npu.iss`. Workflow проверяет совпадение версии с тегом, собирает release binary, native bridge, installer, считает SHA256 и публикует GitHub Release с `ai2npu-setup-<version>.exe`.

## Модели и синхронизация

BGE берётся из Hugging Face репозитория `strokinkv/bge-m3-int8-ov` по HTTPS и устанавливается в:

```text
C:\ProgramData\ai2npu\models\strokinkv\bge-m3-int8-ov
```

Whisper берётся из Hugging Face репозитория `OpenVINO/whisper-large-v3-turbo-int8-ov` по HTTPS и устанавливается в:

```text
C:\ProgramData\ai2npu\models\OpenVINO\whisper-large-v3-turbo-int8-ov
```

Для обновления BGE вручную используйте:

```powershell
& "C:\Program Files\ai2npu\ai2npu.exe" install-model `
  --model "BAAI/bge-m3" `
  --model-dir "C:\ProgramData\ai2npu\models\strokinkv\bge-m3-int8-ov"
```

Для обновления Whisper вручную используйте:

```powershell
powershell -ExecutionPolicy Bypass `
& "C:\Program Files\ai2npu\ai2npu.exe" install-model `
  --model "openai/whisper-large-v3-turbo" `
  --model-dir "C:\ProgramData\ai2npu\models\OpenVINO\whisper-large-v3-turbo-int8-ov"
```

## Полезные проверки

- `ai2npu.exe validate-config`
- `ai2npu.exe check-npu`
- `Invoke-RestMethod http://127.0.0.1:9555/health`

Логи пишутся в `C:\ProgramData\ai2npu\logs\ai2npu.log`.
