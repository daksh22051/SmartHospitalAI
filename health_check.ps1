param(
    [string]$BaseUrl = "http://127.0.0.1:5000",
    [string]$Task = "medium",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$failures = New-Object System.Collections.Generic.List[string]

function Add-Failure {
    param([string]$Message)
    $script:failures.Add($Message)
}

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )
    if (-not $Condition) {
        Add-Failure $Message
    }
}

function Get-Json {
    param([string]$Path)
    Invoke-RestMethod -Uri ($BaseUrl + $Path)
}

function Post-Json {
    param(
        [string]$Path,
        [string]$Body = "{}"
    )
    Invoke-RestMethod -Method Post -Uri ($BaseUrl + $Path) -ContentType "application/json" -Body $Body
}

Write-Host "=== Smart Hospital Health Check ==="
Write-Host "Base URL: $BaseUrl"
Write-Host "Task: $Task | Seed: $Seed"

try {
    Write-Host "\n[1/3] Checking page availability..."
    $pages = @("/", "/controls", "/analytics", "/ai_lab", "/3d_view", "/clinical_ops")
    foreach ($p in $pages) {
        try {
            $resp = Invoke-WebRequest -Uri ($BaseUrl + $p) -UseBasicParsing -TimeoutSec 20
            if ($resp.StatusCode -eq 200) {
                Write-Host ("PASS page {0} -> {1}" -f $p, $resp.StatusCode)
            }
            else {
                Add-Failure ("Page {0} returned {1}" -f $p, $resp.StatusCode)
            }
        }
        catch {
            Add-Failure ("Page {0} failed: {1}" -f $p, $_.Exception.Message)
        }
    }

    Write-Host "\n[2/3] Checking API availability..."
    $apiPaths = @(
        "/api/status",
        "/api/get_state",
        "/api/episode_history",
        "/api/tasks",
        "/api/health",
        "/api/weather",
        "/api/drone_status"
    )

    foreach ($api in $apiPaths) {
        try {
            $j = Get-Json $api
            if ($j -ne $null) {
                Write-Host ("PASS api {0}" -f $api)
            }
            else {
                Add-Failure ("API {0} returned null" -f $api)
            }
        }
        catch {
            Add-Failure ("API {0} failed: {1}" -f $api, $_.Exception.Message)
        }
    }

    Write-Host "\n[3/3] Checking dynamic data changes..."

    # Initialize deterministic environment and clear timeline for clean assertions.
    $initBody = @{ task = $Task; seed = $Seed } | ConvertTo-Json -Compress
    Post-Json "/api/init" $initBody | Out-Null
    Post-Json "/api/timeline/clear" "{}" | Out-Null
    $resetBody = @{ seed = $Seed } | ConvertTo-Json -Compress
    Post-Json "/api/reset" $resetBody | Out-Null

    $beforeState = Get-Json "/api/get_state"
    $beforeHist = Get-Json "/api/episode_history"
    $beforeTimeline = Get-Json "/api/timeline?limit=600&offset=0"

    $stepResult = Post-Json "/api/step" '{"action":1}'
    $aiResult = Post-Json "/api/ai_action" "{}"

    $afterState = Get-Json "/api/get_state"
    $afterHist = Get-Json "/api/episode_history"
    $afterTimeline = Get-Json "/api/timeline?limit=600&offset=0"

    $beforeStep = [int]$beforeState.state.step
    $afterStep = [int]$afterState.state.step
    $beforeHistSteps = [int]$beforeHist.total_steps
    $afterHistSteps = [int]$afterHist.total_steps
    $beforeTimelineCount = @($beforeTimeline.frames).Count
    $afterTimelineCount = @($afterTimeline.frames).Count

    Assert-True ($stepResult.success -eq $true) "Step API did not succeed"
    Assert-True ($aiResult.success -eq $true) "AI action API did not succeed"
    Assert-True ($afterStep -gt $beforeStep) ("State step did not increase ({0} -> {1})" -f $beforeStep, $afterStep)
    Assert-True ($afterHistSteps -gt $beforeHistSteps) ("Episode history steps did not increase ({0} -> {1})" -f $beforeHistSteps, $afterHistSteps)
    Assert-True ($afterTimelineCount -gt $beforeTimelineCount) ("Timeline frames did not increase ({0} -> {1})" -f $beforeTimelineCount, $afterTimelineCount)

    Write-Host ("INFO state.step: {0} -> {1}" -f $beforeStep, $afterStep)
    Write-Host ("INFO episode_history.total_steps: {0} -> {1}" -f $beforeHistSteps, $afterHistSteps)
    Write-Host ("INFO timeline.frames: {0} -> {1}" -f $beforeTimelineCount, $afterTimelineCount)

}
catch {
    Add-Failure ("Unexpected health-check failure: {0}" -f $_.Exception.Message)
}

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "HEALTH CHECK PASSED" -ForegroundColor Green
    exit 0
}
else {
    Write-Host "HEALTH CHECK FAILED" -ForegroundColor Red
    foreach ($f in $failures) {
        Write-Host (" - {0}" -f $f) -ForegroundColor Yellow
    }
    exit 1
}
