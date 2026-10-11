<#
.SYNOPSIS
  Regenerates every ML result of the project from the raw sessions (one command).

.EXAMPLE
  .\ml\run_all.ps1
  .\ml\run_all.ps1 -RawDir C:\proyecto__ciber\ml\data\raw -WithCnn

  -RawDir    folder with <participant>\{clicks,moves,breaks,sessions}.csv (default: ml\data\raw)
  -WithCnn   also install the optional PyTorch CPU wheel (1D-CNN baseline)
  -SkipJava  do not rebuild/test the plugin after exporting the ONNX model
#>
param(
    [string]$RawDir = (Join-Path $PSScriptRoot "data\raw"),
    [switch]$WithCnn,
    [switch]$SkipJava
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($name, [scriptblock]$block) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    # Python logs to stderr; in Windows PowerShell 5.1 that is not an error. Fail on exit code only.
    $ErrorActionPreference = "Continue"
    & $block 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $name (exit $LASTEXITCODE)" }
}

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Step "create venv" { python -m venv .venv }
}
Step "install requirements (exact lock)" { & $py -m pip install -q -r requirements-lock.txt }
if ($WithCnn) {
    Step "install torch (CPU)" { & $py -m pip install -q -r requirements-cnn.txt --index-url https://download.pytorch.org/whl/cpu }
}

$env:ML_RAW_DIR = (Resolve-Path $RawDir -ErrorAction SilentlyContinue)
if (-not $env:ML_RAW_DIR) { Write-Warning "Raw data folder $RawDir not found: only synthetic (DEMO) results will be produced."; $env:ML_RAW_DIR = $RawDir }
$env:PYTHONHASHSEED = "0"

Step "unit tests"                   { & $py -m pytest -q }
Step "A data layer (real data)"     { & $py -m src.data }
Step "B synthetic data"             { & $py -m src.synthetic }
Step "C features"                   { & $py -m src.features }
Step "D models + evaluation"        { & $py -m src.evaluate }
Step "E adversarial experiment"     { & $py -m src.adversarial }
Step "F simulated lag"              { & $py -m src.lag }
Step "G explainability"             { & $py -m src.explain }
Step "H export ONNX + Java fixtures" { & $py -m src.export_onnx }

if (-not $SkipJava) {
    Push-Location (Join-Path $PSScriptRoot "..")
    try {
        Step "plugin build + Java parity tests" { & .\gradlew.bat build --console=plain }
    } finally { Pop-Location }
}
Write-Host "`nDone. Results in ml\results\ (figures, tables, phase_*_results.md, data_report.md)." -ForegroundColor Green
