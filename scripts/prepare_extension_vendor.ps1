# Prepare ONNX Runtime Web vendor files for the Chrome extension dev build.

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path $Root "node_modules\onnxruntime-web\dist"
$TargetDir = Join-Path $Root "extension\dist\vendor\onnxruntime-web"

if (-not (Test-Path $SourceDir)) {
  throw "onnxruntime-web dist directory not found. Run npm install first."
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Copy-Item (Join-Path $SourceDir "ort.min.js") (Join-Path $TargetDir "ort.min.js") -Force
Copy-Item (Join-Path $SourceDir "*.wasm") $TargetDir -Force
Copy-Item (Join-Path $SourceDir "*.mjs") $TargetDir -Force -ErrorAction SilentlyContinue

Write-Host "ONNX Runtime Web vendor files copied to extension/dist/vendor/onnxruntime-web"
