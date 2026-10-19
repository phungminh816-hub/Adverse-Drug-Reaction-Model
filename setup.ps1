param([switch]$UseSystemPackages)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($UseSystemPackages) { python -m venv --system-site-packages .venv }
    else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Không tạo được môi trường Python.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -e . pytest
if ($LASTEXITCODE -ne 0) { throw 'Cài dependency thất bại; xem lỗi phía trên.' }
Write-Host 'Sẵn sàng. Chạy: .\run.ps1'
