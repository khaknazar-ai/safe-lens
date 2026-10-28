param(
  [int]$Port = 8787
)

Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $ProjectRoot

Write-Host "Starting SafeLens generation service..."
Write-Host "Project root: $ProjectRoot"
Write-Host "URL: http://127.0.0.1:$Port"

python -m uvicorn generation_service.app:app --host 127.0.0.1 --port $Port
