# ai2npu_genai_bridge

Native Whisper bridge for the Rust service. It exposes a small C ABI over the
OpenVINO GenAI C++ `WhisperPipeline`, so production builds do not need to ship a
Python runtime.

Build prerequisites:

- OpenVINO GenAI Archive SDK for Windows, not the Python wheel.
- CMake and a C++17 compiler.

MSVC example:

```powershell
.\scripts\setup-openvino-sdk.ps1 -SdkRoot "C:\path\to\openvino_sdk"
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake -S tools\openvino_genai_src -B build\openvino_genai_msvc_clean -G "Visual Studio 17 2022" -A x64 `
    -DOpenVINO_DIR="$env:OPENVINO_SDK_ROOT\runtime\cmake" `
    -DENABLE_PYTHON=OFF -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF `
    -DPCRE2_SUPPORT_LIBBZ2=OFF -DPCRE2_SUPPORT_LIBZ=OFF
& $cmake --build build\openvino_genai_msvc_clean --target openvino_genai --config Release --parallel 4
& $cmake -S native\ai2npu_genai_bridge -B build\ai2npu_genai_bridge -G "Visual Studio 17 2022" -A x64 `
    -DOpenVINO_DIR="$env:OPENVINO_SDK_ROOT\runtime\cmake" `
    -DOpenVINOGenAI_DIR="$PWD\build\openvino_genai_msvc_clean"
& $cmake --build build\ai2npu_genai_bridge --config Release --parallel 4
```

At runtime the Rust service loads `ai2npu_genai_bridge.dll` from the executable
directory, or from `AI2NPU_GENAI_BRIDGE_DLL` when that environment variable is
set. Native audio supports only `NPU`; CPU, GPU, AUTO, and HETERO fallback are
intentionally unsupported.
