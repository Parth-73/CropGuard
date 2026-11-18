$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$archive = Join-Path $projectRoot "data/raw/weedsgalore/original.zip"
New-Item -ItemType Directory -Force (Split-Path -Parent $archive) | Out-Null
if (!(Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip" -OutFile $archive
}
& "$projectRoot/.venv/Scripts/python.exe" "$PSScriptRoot/prepare_weedsgalore.py" --archive $archive
if ($LASTEXITCODE -ne 0) { throw "Dataset conversion failed" }
