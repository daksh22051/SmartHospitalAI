param(
    [string]$RemoteName = "hf",
    [string]$Branch = "master"
)

$ErrorActionPreference = "Stop"

Write-Host "[HF DEPLOY] Preparing push to remote '$RemoteName' branch '$Branch'"

if (-not (Test-Path "./Dockerfile")) {
    throw "Dockerfile not found in current directory. Run from project root."
}

git status --short

Write-Host "[HF DEPLOY] Pushing latest commits..."
git push $RemoteName $Branch

Write-Host "[HF DEPLOY] Done. Check Space build logs in Hugging Face UI."
