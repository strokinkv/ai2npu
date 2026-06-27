# Development Environment Setup and Build

## What to Install

For Windows development you need:

- Python is not required;
- Rust toolchain `stable-x86_64-pc-windows-msvc`;
- Visual Studio Build Tools 2022 with the `Desktop development with C++` workload;
- Windows 10/11 SDK;
- MSYS2, if you also need GNU builds or Unix-style helper tools;
- Inno Setup 6 for installer builds;
- Internet access for downloading the BGE and Whisper models from Hugging Face over HTTPS.

For OpenVINO dependencies, use the SDK archive:

```powershell
$env:OPENVINO_SDK_ROOT = "C:\path\to\openvino_sdk"
$env:OPENVINO_LIB_DIR = "$env:OPENVINO_SDK_ROOT\runtime\lib\intel64\Release"
$env:PATH = "$env:OPENVINO_SDK_ROOT\runtime\bin\intel64\Release;$env:OPENVINO_SDK_ROOT\runtime\3rdparty\tbb\bin;$env:PATH"
```

## Environment Preparation

1. Make sure `rustup`, `cargo`, `link.exe`, `cl.exe`, and `ISCC.exe` are available.
2. Confirm that the Intel NPU driver is installed and the device is visible to Windows.

For local Windows builds, run commands through `VsDevCmd.bat` so `LIB`, `INCLUDE`, and SDK libraries are set correctly.

## Build Verification

Start with formatting and tests:

```powershell
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo fmt"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo test"
```

Default `cargo test` does not require downloaded models or a live NPU. Tests that depend on the local `models/` directory are opt-in:

```powershell
$env:AI2NPU_RUN_MODEL_TESTS = "1"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo test"
```

Live NPU checks are opt-in:

```powershell
$env:AI2NPU_RUN_NPU_TESTS = "1"
$env:AI2NPU_SMOKE_WAV = "C:\path\to\16bit-pcm.wav"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo test"
```

To run the service from source:

```powershell
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo run -- run --config config.example.toml"
```

## Building the Distribution

Build the distributable with `packaging\prepare-dist.ps1`:

```powershell
$env:LIB = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64;$env:LIB"
rustup run stable-x86_64-pc-windows-msvc powershell -ExecutionPolicy Bypass -File packaging\prepare-dist.ps1
```

The script:

- builds `ai2npu.exe`;
- copies the OpenVINO runtime DLLs;
- uses `ai2npu.exe install-model` for HTTPS-based BGE and Whisper model downloads;
- does not bundle Git, Git LFS, or a separate downloader script.

`dist/`, `target/`, `build/`, `tools/`, and `models/` are local artifacts and are not published to GitHub.

GitHub Actions downloads the official OpenVINO GenAI Windows archive from OpenVINO storage and caches the extracted SDK under `tools/` for later runs. The hosted runner does not have Intel NPU hardware, so model and live NPU tests stay opt-in and are not enabled in CI.

Build the installer with Inno Setup:

```powershell
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" packaging\ai2npu.iss
```

The output is written to `dist\ai2npu-setup-<version>.exe`.

## GitHub Release

The automated release runs when pushing a `vMAJOR.MINOR.PATCH` tag, for example:

```powershell
git tag -a v1.2.0 -m "ai2npu v1.2.0"
git push origin v1.2.0
```

Before creating the tag, update the version in `Cargo.toml`, `Cargo.lock`, and `packaging/ai2npu.iss`. The workflow verifies that the version matches the tag, builds the release binary, native bridge, installer, computes SHA256, and publishes a GitHub Release with `ai2npu-setup-<version>.exe`.

## Models and Synchronization

BGE is pulled from the Hugging Face repository `strokinkv/bge-m3-int8-ov` over HTTPS and installed into:

```text
C:\ProgramData\ai2npu\models\strokinkv\bge-m3-int8-ov
```

Whisper is pulled from the Hugging Face repository `OpenVINO/whisper-large-v3-turbo-int8-ov` over HTTPS and installed into:

```text
C:\ProgramData\ai2npu\models\OpenVINO\whisper-large-v3-turbo-int8-ov
```

To refresh BGE manually:

```powershell
& "C:\Program Files\ai2npu\ai2npu.exe" install-model `
  --model "BAAI/bge-m3" `
  --model-dir "C:\ProgramData\ai2npu\models\strokinkv\bge-m3-int8-ov"
```

To refresh Whisper manually:

```powershell
powershell -ExecutionPolicy Bypass `
& "C:\Program Files\ai2npu\ai2npu.exe" install-model `
  --model "openai/whisper-large-v3-turbo" `
  --model-dir "C:\ProgramData\ai2npu\models\OpenVINO\whisper-large-v3-turbo-int8-ov"
```

## Useful Checks

- `ai2npu.exe validate-config`
- `ai2npu.exe check-npu`
- `Invoke-RestMethod http://127.0.0.1:9555/health`

Logs are written to `C:\ProgramData\ai2npu\logs\ai2npu.log`.
