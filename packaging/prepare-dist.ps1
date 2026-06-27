param(
    [string]$Configuration = "release",
    [string]$RustToolchain = "stable-x86_64-pc-windows-msvc",
    [string]$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    [string]$DistDir = "dist\ai2npu",
    [string]$OpenVinoGenAiRoot = "tools\openvino_genai_2026.2.0.0_archive\openvino_genai_windows_2026.2.0.0_x86_64",
    [string]$BridgeDll = "build\ai2npu_genai_bridge_archive\Release\ai2npu_genai_bridge.dll",
    # ONNX Runtime version required by the `ort` crate (streaming VAD). Must match
    # ort 2.0.0-rc.10's expected ORT 1.22.x.
    [string]$OnnxRuntimeVersion = "1.22.0",
    [string]$OnnxRuntimeUrl = "",
    [switch]$SkipCargoBuild
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$archive = Resolve-Path $OpenVinoGenAiRoot
$runtimeBin = Join-Path $archive "runtime\bin\intel64\Release"
$runtimeLib = Join-Path $archive "runtime\lib\intel64\Release"
$tbbBin = Join-Path $archive "runtime\3rdparty\tbb\bin"

foreach ($path in @($runtimeBin, $runtimeLib, $tbbBin)) {
    if (-not (Test-Path $path)) {
        throw "OpenVINO SDK path not found: $path"
    }
}

$env:OPENVINO_SDK_ROOT = $archive.Path
$env:OPENVINO_LIB_DIR = $runtimeLib
$env:PATH = "$runtimeBin;$tbbBin;$env:PATH"

if (-not $SkipCargoBuild) {
    if ($RustToolchain -like "*windows-msvc") {
        if (-not (Test-Path $VsDevCmd)) {
            throw "Visual Studio Developer Command Prompt not found at $VsDevCmd"
        }
        & cmd.exe /d /s /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 && rustup run $RustToolchain cargo build --release"
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build --release failed with exit code $LASTEXITCODE"
        }
    } else {
        $env:Path = "C:\msys64\ucrt64\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
        rustup run $RustToolchain cargo build --release
    }
}

$dist = Join-Path $ProjectRoot $DistDir
if (Test-Path $dist) {
    Remove-Item -LiteralPath $dist -Recurse -Force
}
New-Item -ItemType Directory -Force $dist | Out-Null
New-Item -ItemType Directory -Force (Join-Path $dist "models") | Out-Null

$exe = Join-Path $ProjectRoot "target\$Configuration\ai2npu.exe"
if (-not (Test-Path $exe)) {
    throw "ai2npu.exe not found at $exe"
}
Copy-Item $exe (Join-Path $dist "ai2npu.exe")
Copy-Item (Join-Path $ProjectRoot "config.example.toml") (Join-Path $dist "config.example.toml")

$bridge = Resolve-Path $BridgeDll
Copy-Item $bridge (Join-Path $dist "ai2npu_genai_bridge.dll")

$runtimeDlls = @(
    "openvino.dll",
    "openvino_c.dll",
    "openvino_genai.dll",
    "openvino_genai_c.dll",
    # WhisperPipeline targets NPU, but OpenVINO GenAI still needs CPU runtime
    # support for internal pipeline steps such as preprocessing/token handling.
    "openvino_intel_cpu_plugin.dll",
    "openvino_tokenizers.dll",
    "openvino_intel_npu_plugin.dll",
    "openvino_intel_npu_compiler.dll",
    "openvino_ir_frontend.dll",
    "openvino_onnx_frontend.dll",
    "openvino_paddle_frontend.dll",
    "openvino_pytorch_frontend.dll",
    "openvino_tensorflow_frontend.dll",
    "openvino_tensorflow_lite_frontend.dll"
)

foreach ($dll in $runtimeDlls) {
    $source = Join-Path $runtimeBin $dll
    if (-not (Test-Path $source)) {
        throw "runtime DLL not found: $source"
    }
    Copy-Item $source (Join-Path $dist $dll)
}

Get-ChildItem $tbbBin -Filter "*.dll" |
    Where-Object { $_.Name -notlike "*_debug.dll" } |
    Copy-Item -Destination $dist

# ONNX Runtime for the streaming VAD (`ort` load-dynamic). Ship the ort-compatible
# 1.22.x onnxruntime.dll next to ai2npu.exe so it wins the Windows DLL search over
# an incompatible System32 onnxruntime.dll (Windows ML ships 1.17). Static linking
# is not possible (esaxx-rs /MT vs prebuilt ORT /MD).
$ortArchive = Join-Path $ProjectRoot "tools\onnxruntime-win-x64-$OnnxRuntimeVersion"
$ortDll = Join-Path $ortArchive "lib\onnxruntime.dll"
if (-not (Test-Path $ortDll)) {
    $ortZip = Join-Path $ProjectRoot "tools\onnxruntime-win-x64-$OnnxRuntimeVersion.zip"
    if (-not (Test-Path $ortZip)) {
        $url = if ($OnnxRuntimeUrl) {
            $OnnxRuntimeUrl
        } else {
            "https://github.com/microsoft/onnxruntime/releases/download/v$OnnxRuntimeVersion/onnxruntime-win-x64-$OnnxRuntimeVersion.zip"
        }
        Write-Host "Downloading ONNX Runtime $OnnxRuntimeVersion from $url"
        Invoke-WebRequest -Uri $url -OutFile $ortZip
    }
    Expand-Archive -Path $ortZip -DestinationPath (Join-Path $ProjectRoot "tools") -Force
}
if (-not (Test-Path $ortDll)) {
    throw "onnxruntime.dll not found after extraction: $ortDll"
}
Copy-Item $ortDll (Join-Path $dist "onnxruntime.dll")

if ($RustToolchain -like "*windows-gnu") {
    $msysDlls = @(
        "C:\msys64\ucrt64\bin\libstdc++-6.dll",
        "C:\msys64\ucrt64\bin\libgcc_s_seh-1.dll",
        "C:\msys64\ucrt64\bin\libwinpthread-1.dll"
    )
    foreach ($dll in $msysDlls) {
        if (Test-Path $dll) {
            Copy-Item $dll $dist
        }
    }
}

Write-Host "Prepared distribution: $dist"
