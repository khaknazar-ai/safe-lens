param(
  [int]$Port = 8790
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Starting SafeLens restricted detector service on 127.0.0.1:$Port"
python -m uvicorn restricted_detector_service.app:app --host 127.0.0.1 --port $Port
