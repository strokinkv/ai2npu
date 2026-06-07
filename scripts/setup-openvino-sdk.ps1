param(
    [Parameter(Mandatory = $true)]
    [string]$SdkRoot
)

$ErrorActionPreference = "Stop"

$resolved = Resolve-Path $SdkRoot
$runtimeBin = Join-Path $resolved.Path "runtime\bin\intel64\Release"
$runtimeLib = Join-Path $resolved.Path "runtime\lib\intel64\Release"
$tbbBin = Join-Path $resolved.Path "runtime\3rdparty\tbb\bin"
$cmakeDir = Join-Path $resolved.Path "runtime\cmake"

foreach ($path in @($runtimeBin, $runtimeLib, $tbbBin, $cmakeDir)) {
    if (-not (Test-Path $path)) {
        throw "OpenVINO SDK path not found: $path"
    }
}

$env:OPENVINO_SDK_ROOT = $resolved.Path
$env:OPENVINO_LIB_DIR = $runtimeLib
$env:PATH = "$runtimeBin;$tbbBin;$env:PATH"

Write-Host "OPENVINO_SDK_ROOT=$($env:OPENVINO_SDK_ROOT)"
Write-Host "OPENVINO_LIB_DIR=$($env:OPENVINO_LIB_DIR)"
Write-Host "Updated PATH with OpenVINO runtime and TBB bins."
