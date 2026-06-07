# OpenVINO SDK Archive Setup

## Purpose

This helper configures the current PowerShell session for building and running `ai2npu` against the OpenVINO SDK archive, without relying on the Python wheel.

## What Must Be Present

Use an extracted Windows OpenVINO SDK archive. It should contain:

- `runtime\bin\intel64\Release`
- `runtime\lib\intel64\Release`
- `runtime\3rdparty\tbb\bin`
- `runtime\cmake`

## Session Setup

```powershell
. \scripts\setup-openvino-sdk.ps1 -SdkRoot "C:\path\to\openvino_sdk"
```

The script sets:

- `OPENVINO_SDK_ROOT`
- `OPENVINO_LIB_DIR`
- `PATH` with the OpenVINO runtime and TBB bins

## After Setup

Verify the build:

```powershell
rustup run stable-x86_64-pc-windows-msvc cargo test --test config_tests prepared_model_bundles_are_valid
```

For a full test run, keep the same `PATH` in the current session.

