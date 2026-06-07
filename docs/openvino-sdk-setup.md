# Подготовка OpenVINO SDK archive

## Назначение

Этот скрипт настраивает текущую PowerShell-сессию для сборки и запуска `ai2npu` с OpenVINO SDK archive, без Python wheel.

## Что должно быть скачано

Нужен распакованный OpenVINO SDK archive для Windows. Внутри должны быть каталоги:

- `runtime\bin\intel64\Release`
- `runtime\lib\intel64\Release`
- `runtime\3rdparty\tbb\bin`
- `runtime\cmake`

## Как настроить сессию

```powershell
. \scripts\setup-openvino-sdk.ps1 -SdkRoot "C:\path\to\openvino_sdk"
```

Скрипт задаёт:

- `OPENVINO_SDK_ROOT`
- `OPENVINO_LIB_DIR`
- `PATH` с OpenVINO runtime и TBB

## После настройки

Проверьте сборку:

```powershell
rustup run stable-x86_64-pc-windows-msvc cargo test --test config_tests prepared_model_bundles_are_valid
```

Для полного тестового прогона нужен тот же `PATH` в текущей сессии.

