[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$Apply,
    [string]$ExpectedContext
)

$ErrorActionPreference = "Stop"
$imageName = "telco-churn-api:lab11"
$deploymentName = "telco-churn-api"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($CheckOnly -and $Apply) {
    throw "Choose either -CheckOnly or -Apply, not both."
}

if ($Apply -and [string]::IsNullOrWhiteSpace($ExpectedContext)) {
    throw "Applying requires -ExpectedContext with the exact Kubernetes context name."
}

$requiredFiles = @(
    (Join-Path $projectRoot "deployment\model.pkl"),
    (Join-Path $projectRoot "deployment\preprocessor.pkl"),
    (Join-Path $projectRoot "deployment\model_metadata.json"),
    (Join-Path $projectRoot "Dockerfile"),
    (Join-Path $projectRoot "k8s\deployment.yaml")
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required deployment file is missing: $file"
    }
}

$metadata = Get-Content -LiteralPath (Join-Path $projectRoot "deployment\model_metadata.json") -Raw |
    ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($metadata.model_name) -or
    [string]::IsNullOrWhiteSpace([string]$metadata.model_version) -or
    [string]::IsNullOrWhiteSpace($metadata.run_id) -or
    [string]::IsNullOrWhiteSpace($metadata.model_id)) {
    throw "Deployment metadata is missing model name, version, or provenance."
}
if ($metadata.model_stage -ne "Production" -or
    $metadata.preprocessor_run_id -ne $metadata.run_id) {
    throw "Deployment bundle is not a matched MLflow Production model/preprocessor."
}

foreach ($artifactName in @("model.pkl", "preprocessor.pkl")) {
    $artifactPath = Join-Path (Join-Path $projectRoot "deployment") $artifactName
    $expectedHash = $metadata.sha256.PSObject.Properties[$artifactName].Value
    if ([string]::IsNullOrWhiteSpace($expectedHash) -or
        -not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
        throw "Deployment artifact or checksum is missing: $artifactName"
    }

    $actualHash = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash
    if ($actualHash -ne $expectedHash) {
        throw "Deployment artifact checksum mismatch: $artifactName"
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI is not installed or is not on PATH."
}
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    throw "kubectl is not installed or is not on PATH."
}

$previousErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$dockerServer = & docker info --format "{{.ServerVersion}}" 2>&1
$dockerExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorPreference
if ($dockerExitCode -ne 0 -or [string]::IsNullOrWhiteSpace(($dockerServer -join ""))) {
    throw "Docker engine is unavailable. Start Docker Desktop, then retry."
}

$ErrorActionPreference = "Continue"
$currentContext = & kubectl config current-context 2>&1
$kubectlContextExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorPreference
if ($kubectlContextExitCode -ne 0 -or [string]::IsNullOrWhiteSpace(($currentContext -join ""))) {
    throw "No Kubernetes context is configured. Configure and verify a context first."
}

if ($Apply -and $currentContext.Trim() -ne $ExpectedContext.Trim()) {
    throw "Current context '$currentContext' does not match expected context '$ExpectedContext'."
}

$ErrorActionPreference = "Continue"
$clusterInfo = & kubectl cluster-info 2>&1
$kubectlClusterInfoExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorPreference
if ($kubectlClusterInfoExitCode -ne 0) {
    throw "Kubernetes API is not reachable from context '$currentContext'."
}

if ($CheckOnly) {
    Write-Output "Preflight passed for model $($metadata.model_name) version $($metadata.model_version)."
    Write-Output "Image: $imageName; deployment/container: $deploymentName."
    Write-Output "No image was built and no Kubernetes resources were changed."
    exit 0
}

if (-not $Apply) {
    Write-Output "Preflight completed. No deployment was requested."
    Write-Output "To deploy, obtain approval and pass -Apply -ExpectedContext <exact-context>."
    exit 0
}

Push-Location $projectRoot
try {
    & docker build --tag $imageName --file Dockerfile .
    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed."
    }

    & kubectl apply --filename (Join-Path $projectRoot "k8s\deployment.yaml")
    if ($LASTEXITCODE -ne 0) {
        throw "Kubernetes manifest application failed."
    }

    & kubectl rollout status "deployment/$deploymentName" --timeout=120s
    if ($LASTEXITCODE -ne 0) {
        throw "Deployment rollout did not complete successfully."
    }
}
finally {
    Pop-Location
}

Write-Output "Deployment $deploymentName is available in context '$currentContext'."
