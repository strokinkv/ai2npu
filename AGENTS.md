# Repository Guidelines

## Project Layout

Core code is in `src/`. `http.rs` owns routes, `inference.rs` selects executors, `openvino_c.rs` wraps the OpenVINO C API, and `windows_service.rs` handles SCM integration. Tests live in `tests/`. Downloaded models may exist locally in `models/`, but that directory is ignored and must not be committed. Native bridge code is in `native/ai2npu_genai_bridge/`. Packaging scripts are in `packaging/`. Developer helpers are in `scripts/`. Local operational notes may live in ignored `docs/superpowers/`.

## Build And Run

Use the OpenVINO SDK archive helper first:

```powershell
. \scripts\setup-openvino-sdk.ps1 -SdkRoot "C:\path\to\openvino_sdk"
```

Then use the MSVC toolchain:

```powershell
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo fmt"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo test"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo run -- run --config config.example.toml"
```

Use `packaging\prepare-dist.ps1` to build `dist\ai2npu`, then `ISCC.exe packaging\ai2npu.iss` to build the installer.

## Runtime And Service

`cargo run -- run --config config.example.toml` starts the service in-process. The Windows service commands are `install-service`, `start-service`, `stop-service`, `restart-service`, and `uninstall-service`. `init-config` creates `C:\ProgramData\ai2npu\config.toml` when the installer needs a first-run config.

## Style And Tests

Follow Rust 2021 and `rustfmt`. Use `snake_case` for functions and modules, `PascalCase` for types, and `anyhow::Context` at subsystem boundaries. Add or update tests in `tests/*_tests.rs` for behavior changes. Keep downloaded-model tests behind `AI2NPU_RUN_MODEL_TESTS=1` and live NPU tests behind `AI2NPU_RUN_NPU_TESTS=1` so default `cargo test` stays deterministic.

## Packaging And Configuration

`config.example.toml` is the canonical config template. Keep the service bound to loopback unless a reviewed change explicitly broadens exposure. Do not commit `.venv/`, `target/`, `build/`, `dist/`, `models/`, `logs/`, `docs/superpowers/`, or downloaded SDK archives. The installer now uses the OpenVINO SDK archive path from `scripts/setup-openvino-sdk.ps1`; there is no Python runtime path in the active build.
