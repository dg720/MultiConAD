param(
    [ValidateSet("run", "smoke", "status")]
    [string]$Mode = "run",
    [switch]$Resume,
    [int[]]$Seeds
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SuiteScript = Join-Path $RepoRoot "experiments\multiseed_suite.py"
$SuiteDir = Join-Path $RepoRoot "tables\experiment-results\multiseed-suite"
$LauncherLogDir = Join-Path $SuiteDir "launcher_logs"

if (-not (Test-Path $Python)) {
    throw "Python not found at $Python"
}

if (-not (Test-Path $SuiteScript)) {
    throw "Suite script not found at $SuiteScript"
}

New-Item -ItemType Directory -Force -Path $LauncherLogDir | Out-Null

$ArgsList = @($SuiteScript, $Mode)
if ($Resume) {
    $ArgsList += "--resume"
}
if ($Seeds) {
    $ArgsList += "--seeds"
    $ArgsList += ($Seeds | ForEach-Object { "$_" })
}

if ($Mode -eq "status") {
    & $Python @ArgsList
    exit $LASTEXITCODE
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ConsoleLog = Join-Path $LauncherLogDir "${Mode}_${Timestamp}.log"
$InnerLogName = "run.log"
if ($Mode -eq "smoke") {
    $InnerLogName = "smoke.log"
}

Write-Host "Repo root: $RepoRoot"
Write-Host "Suite mode: $Mode"
Write-Host "Console log: $ConsoleLog"
Write-Host "Inner run log: $(Join-Path $SuiteDir $InnerLogName)"

$previousEap = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $Python @ArgsList 2>&1 | Tee-Object -FilePath $ConsoleLog
    exit $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousEap
}
