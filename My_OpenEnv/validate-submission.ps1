#!/usr/bin/env pwsh
<#
.SYNOPSIS
    validate-submission.ps1 — OpenEnv Submission Validator (Windows)

.DESCRIPTION
    Checks that your HF Space is live, Docker image builds, and openenv validate passes.

.PARAMETERS
    PingUrl     Your HuggingFace Space URL (e.g. https://your-space.hf.space)
    RepoDir     Path to your repo (default: current directory)

.EXAMPLES
    .\validate-submission.ps1 https://my-team.hf.space
    .\validate-submission.ps1 https://my-team.hf.space -RepoDir ./my-repo

.NOTES
    Prerequisites:
      - Docker: https://docs.docker.com/get-docker/
      - openenv-core: pip install openenv-core
      - curl (usually pre-installed)
#>

param(
    [Parameter(Position=0, Mandatory=$true)]
    [string]$PingUrl,
    
    [Parameter(Position=1, Mandatory=$false)]
    [string]$RepoDir = "."
)

$ErrorActionPreference = "Stop"

# Colors
$RED     = "$([char]27)[0;31m"
$GREEN   = "$([char]27)[0;32m"
$YELLOW  = "$([char]27)[1;33m"
$BOLD    = "$([char]27)[1m"
$NC      = "$([char]27)[0m"

$DOCKER_BUILD_TIMEOUT = 600
$PASS = 0

# Validate repo directory
try {
    $RepoDir = Resolve-Path $RepoDir -ErrorAction Stop
}
catch {
    Write-Host "Error: directory '$RepoDir' not found"
    exit 1
}

$PingUrl = $PingUrl.TrimEnd('/')

function log {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message"
}

function pass {
    param([string]$Message)
    log "${GREEN}PASSED${NC} -- $Message"
    $script:PASS++
}

function fail {
    param([string]$Message)
    log "${RED}FAILED${NC} -- $Message"
}

function hint {
    param([string]$Message)
    Write-Host "  ${YELLOW}Hint:${NC} $Message"
}

function stop_at {
    param([string]$Step)
    Write-Host ""
    Write-Host "${RED}${BOLD}Validation stopped at $Step.${NC} Fix the above before continuing."
    exit 1
}

# Header
Write-Host ""
Write-Host "${BOLD}========================================${NC}"
Write-Host "${BOLD}  OpenEnv Submission Validator (Windows)${NC}"
Write-Host "${BOLD}========================================${NC}"
log "Repo:     $RepoDir"
log "Ping URL: $PingUrl"
Write-Host ""

# Step 1: Ping HF Space
log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PingUrl/reset) ..."

try {
    $response = Invoke-WebRequest -Uri "$PingUrl/reset" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body "{}" `
        -TimeoutSec 30 `
        -ErrorAction Stop
    
    if ($response.StatusCode -eq 200) {
        pass "HF Space is live and responds to /reset"
    }
    else {
        fail "HF Space /reset returned HTTP $($response.StatusCode) (expected 200)"
        hint "Make sure your Space is running and the URL is correct."
        hint "Try opening $PingUrl in your browser first."
        stop_at "Step 1"
    }
}
catch {
    fail "HF Space not reachable (connection failed or timed out)"
    hint "Check your network connection and that the Space is running."
    hint "Try: curl -s -X POST $PingUrl/reset"
    stop_at "Step 1"
}

# Step 2: Docker Build
log "${BOLD}Step 2/3: Running docker build${NC} ..."

$dockerPath = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerPath) {
    fail "docker command not found"
    hint "Install Docker: https://docs.docker.com/get-docker/"
    stop_at "Step 2"
}

$dockerfilePath = Join-Path $RepoDir "Dockerfile"
$dockerfileServerPath = Join-Path $RepoDir "server" "Dockerfile"

if (Test-Path $dockerfilePath) {
    $dockerContext = $RepoDir
}
elseif (Test-Path $dockerfileServerPath) {
    $dockerContext = Join-Path $RepoDir "server"
}
else {
    fail "No Dockerfile found in repo root or server/ directory"
    stop_at "Step 2"
}

log "  Found Dockerfile in $dockerContext"

try {
    $buildOutput = & docker build $dockerContext 2>&1
    pass "Docker build succeeded"
}
catch {
    fail "Docker build failed"
    $buildOutput | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
    stop_at "Step 2"
}

# Step 3: OpenEnv Validate
log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

$openenvPath = Get-Command openenv -ErrorAction SilentlyContinue
if (-not $openenvPath) {
    fail "openenv command not found"
    hint "Install it: pip install openenv-core"
    stop_at "Step 3"
}

try {
    Push-Location $RepoDir
    $validateOutput = & openenv validate 2>&1
    Pop-Location
    pass "openenv validate passed"
    if ($validateOutput) {
        log "  $validateOutput"
    }
}
catch {
    fail "openenv validate failed"
    Write-Host $_
    stop_at "Step 3"
}

# Success
Write-Host ""
Write-Host "${BOLD}========================================${NC}"
Write-Host "${GREEN}${BOLD}  All 3/3 checks passed!${NC}"
Write-Host "${GREEN}${BOLD}  Your submission is ready to submit.${NC}"
Write-Host "${BOLD}========================================${NC}"
Write-Host ""

exit 0
