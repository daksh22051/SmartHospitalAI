param(
    [string]$ImageName = "smart-hospital-orchestration-test",
    [string]$ProjectPath = ".",
    [switch]$SkipRun
)

$ErrorActionPreference = "Stop"

function Fail {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

Write-Host "=== Docker Verification ==="
Write-Host "Image: $ImageName"
Write-Host "Project: $ProjectPath"

try {
    docker version | Out-Null
}
catch {
    Write-Host "Docker daemon is not reachable." -ForegroundColor Yellow
    Write-Host "Start Docker Desktop and re-run this script." -ForegroundColor Yellow
    Write-Host "Suggested command: Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'"
    Fail "Docker daemon check failed"
}

Write-Host "PASS daemon reachable"

Write-Host "Building image..."
docker build -t $ImageName $ProjectPath
if ($LASTEXITCODE -ne 0) {
    Fail "docker build failed"
}
Write-Host "PASS build"

if ($SkipRun) {
    Write-Host "Skipping container run (--SkipRun set)."
    exit 0
}

Write-Host "Running container..."
docker run --rm $ImageName
if ($LASTEXITCODE -ne 0) {
    Fail "docker run failed"
}

Write-Host "PASS run"
Write-Host "Docker verification completed successfully." -ForegroundColor Green
