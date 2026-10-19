param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Chưa có môi trường. Chạy .\setup.ps1 trước.'
}
$models = @()
if (Test-Path -LiteralPath 'artifacts') {
    $models = Get-ChildItem -LiteralPath 'artifacts' -Directory |
        Where-Object { $_.Name -notlike '*smoke*' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'metadata.json')) }
}
if (-not $models) {
    & $pythonPath scripts/run_pipeline.py --models logreg
    if ($LASTEXITCODE -ne 0) { throw 'Pipeline chưa hoàn thành; xem lỗi phía trên.' }
}
& $pythonPath -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port $Port --browser.gatherUsageStats false
