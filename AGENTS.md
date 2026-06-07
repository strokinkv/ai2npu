# Repository Guidelines

## Communication

Think in English. Answer user-facing summaries and questions in Russian. Keep responses brief and practical.

## Project Layout

Core Rust code is in `src/`. HTTP routes live in `src/http.rs`, executor selection and model runtime behavior in `src/inference.rs`, BGE embeddings in `src/bge_embeddings.rs`, OpenVINO C bindings in `src/openvino_c.rs`, CLI definitions in `src/cli.rs`, and Windows service integration in `src/windows_service.rs`. Tests live in `tests/`. Native C++ bridge code is in `native/ai2npu_genai_bridge/`. Packaging is in `packaging/`, helper scripts in `scripts/`, and documentation in `docs/`.

Ignored local artifacts must not be committed: `target/`, `build/`, `dist/`, `models/`, `tools/`, `logs/`, `.venv/`, and `docs/superpowers/`.

## Feature Development Process

Track new ideas in GitHub Issues, not `TODO.md`. Use labels such as `idea`, `enhancement`, `bug`, and `release-blocker`. For behavior changes, define the CLI/API contract, runtime behavior, error handling, tests, documentation impact, and acceptance criteria before implementation.

Use TDD for new features and bug fixes. Add a failing test first, verify it fails for the expected reason, implement the smallest change that passes, then refactor. Do not add production behavior without a corresponding test.

## Build And Test

Use the MSVC toolchain and the OpenVINO SDK archive environment:

```powershell
. \scripts\setup-openvino-sdk.ps1 -SdkRoot "C:\path\to\openvino_sdk"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo fmt -- --check"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo test"
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 && rustup run stable-x86_64-pc-windows-msvc cargo clippy -- -D warnings"
```

Default tests must stay deterministic. Keep downloaded-model tests behind `AI2NPU_RUN_MODEL_TESTS=1` and live NPU tests behind `AI2NPU_RUN_NPU_TESTS=1`.

## Packaging

Build the distribution with:

```powershell
rustup run stable-x86_64-pc-windows-msvc powershell -ExecutionPolicy Bypass -File packaging\prepare-dist.ps1
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" packaging\ai2npu.iss
```

`config.example.toml` is the canonical config template. Keep the service bound to loopback unless a reviewed change explicitly broadens exposure. Python is not part of the active runtime or build.

## Release Process

Before a release, update versions in `Cargo.toml`, `Cargo.lock`, and `packaging/ai2npu.iss`. Run `cargo fmt -- --check`, `cargo test`, and `cargo clippy -- -D warnings`. Commit with `Release MAJOR.MINOR.PATCH`.

Push `main`, then create and push an annotated tag:

```powershell
git tag -a v1.0.1 -m "ai2npu v1.0.1"
git push origin v1.0.1
```

GitHub Actions verifies version metadata, restores/downloads the official OpenVINO GenAI archive, runs format/tests/clippy, builds the Rust binary, native bridge, distribution, installer, SHA256, and GitHub Release. Verify the release asset and checksum with `gh release view vMAJOR.MINOR.PATCH --json assets,url`.

After the release workflow succeeds, download the produced installer from GitHub Releases, install it on the Windows test machine, and run the runtime checks below against the installed service. Do not consider a release validated only because GitHub Actions passed.

## Runtime Checks

After installing a release, verify:

```powershell
& "C:\Program Files\ai2npu\ai2npu.exe" version
curl.exe http://127.0.0.1:9555/health
curl.exe http://127.0.0.1:9555/v1/models
```

Also test embeddings and Whisper manually when release behavior touches inference, model loading, packaging, or OpenVINO runtime files.
