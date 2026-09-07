$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
}
$env:HF_HUB_DISABLE_TELEMETRY = '1'
# Bound CPU parallelism; the local Windows worker was severely slower at its default.
# Respect explicit user settings for other machines or measured workloads.
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '4' }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = '4' }
& '.\.venv\Scripts\python.exe' -m uvicorn app:app --host 127.0.0.1 --port 8000 --no-access-log
exit $LASTEXITCODE
