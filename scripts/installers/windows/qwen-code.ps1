$ErrorActionPreference = "Stop"

param(
    [string]$ApiKey,
    [string]$Model
)

$ScriptName = "Qwen Code Nordlys Installer"
$NodeMinVersion = 20
$NodeInstallVersion = "22"
$PackageName = "@qwen-code/qwen-code"
$ApiBaseUrl = "https://api.llmadaptive.uk/v1"
$DefaultModel = "nordlys/nordlys-code"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message"
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK]  $Message"
}

function Write-Failure {
    param([string]$Message)
    Write-Host "[ERR] $Message"
    exit 1
}

function Get-NodeMajorVersion {
    param([string]$Version)
    $clean = $Version.TrimStart("v")
    return [int]($clean.Split(".")[0])
}

function Ensure-Node {
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCmd) {
        $version = (& node --version)
        $major = Get-NodeMajorVersion $version
        if ($major -ge $NodeMinVersion) {
            Write-Success "Node.js detected: $version"
            return
        }
        Write-Info "Node.js $version detected but needs >= $NodeMinVersion. Installing $NodeInstallVersion..."
    } else {
        Write-Info "Node.js not found. Installing $NodeInstallVersion..."
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install OpenJS.NodeJS --version $NodeInstallVersion --silent --accept-package-agreements --accept-source-agreements | Out-Null
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        choco install nodejs --version=$NodeInstallVersion -y | Out-Null
    } else {
        Write-Failure "Node.js not found and neither winget nor Chocolatey are available. Install Node.js from https://nodejs.org and re-run."
    }

    $nodePath = Join-Path $env:ProgramFiles "nodejs"
    if (Test-Path $nodePath -and $env:Path -notlike "*$nodePath*") {
        $env:Path = "$nodePath;$env:Path"
    }

    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        Write-Failure "Node.js installation did not update PATH. Restart your terminal and re-run."
    }

    Write-Success "Node.js installed: $(node --version)"
}

function Install-QwenCode {
    if (Get-Command qwen -ErrorAction SilentlyContinue) {
        Write-Success "Qwen Code already installed: $(qwen --version)"
        return
    }

    Write-Info "Installing Qwen Code via npm..."
    npm install -g $PackageName | Out-Null
    if (-not (Get-Command qwen -ErrorAction SilentlyContinue)) {
        Write-Failure "Qwen Code installation failed."
    }
    Write-Success "Qwen Code installed."
}

function Ensure-ApiKey {
    if (-not $ApiKey) {
        $ApiKey = $env:NORDLYS_API_KEY
    }
    if (-not $ApiKey) {
        $ApiKey = $env:OPENAI_API_KEY
    }
    if (-not $ApiKey) {
        $ApiKey = Read-Host "Enter your Nordlys API key"
    }
    if (-not $ApiKey) {
        Write-Failure "API key is required."
    }
    return $ApiKey
}

Write-Host "=========================================="
Write-Host "  $ScriptName"
Write-Host "=========================================="

Ensure-Node
Install-QwenCode

$ApiKey = Ensure-ApiKey
if (-not $Model) { $Model = $env:NORDLYS_MODEL }
if (-not $Model) { $Model = $env:OPENAI_MODEL }
if (-not $Model) { $Model = $DefaultModel }

$env:OPENAI_API_KEY = $ApiKey
$env:OPENAI_BASE_URL = $ApiBaseUrl
$env:OPENAI_MODEL = $Model

[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $ApiKey, "User")
[Environment]::SetEnvironmentVariable("OPENAI_BASE_URL", $ApiBaseUrl, "User")
[Environment]::SetEnvironmentVariable("OPENAI_MODEL", $Model, "User")

Write-Success "Qwen Code configured for Nordlys."
Write-Host "Run: qwen --version"
