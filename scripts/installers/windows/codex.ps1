$ErrorActionPreference = "Stop"

param(
    [string]$ApiKey,
    [string]$Model,
    [string]$ModelProvider
)

$ScriptName = "OpenAI Codex Nordlys Installer"
$NodeMinVersion = 18
$NodeInstallVersion = "22"
$PackageName = "@openai/codex"
$ApiBaseUrl = "https://api.llmadaptive.uk/v1"
$DefaultModel = "nordlys/nordlys-code"
$DefaultModelProvider = "nordlys"

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

function Install-Codex {
    if (Get-Command codex -ErrorAction SilentlyContinue) {
        Write-Success "Codex already installed: $(codex --version)"
        return
    }

    Write-Info "Installing Codex via npm..."
    npm install -g $PackageName | Out-Null
    if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
        Write-Failure "Codex installation failed."
    }
    Write-Success "Codex installed."
}

function Ensure-ApiKey {
    if (-not $ApiKey) {
        $ApiKey = $env:NORDLYS_API_KEY
    }
    if (-not $ApiKey) {
        $ApiKey = Read-Host "Enter your Nordlys API key"
    }
    if (-not $ApiKey) {
        Write-Failure "API key is required."
    }
    return $ApiKey
}

function Write-Config {
    param([string]$ConfigPath, [string]$ModelValue, [string]$ProviderValue)

    if (Test-Path $ConfigPath) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupPath = "$ConfigPath.$timestamp.bak"
        Copy-Item $ConfigPath $backupPath -Force
        Write-Info "Backed up existing config to $backupPath"
    }

    $config = @"
# Nordlys Configuration
model = "$ModelValue"
model_provider = "$ProviderValue"
approval_policy = "untrusted"

[model_providers.nordlys]
name = "Nordlys"
base_url = "$ApiBaseUrl"
env_key = "NORDLYS_API_KEY"
wire_api = "chat"
"@

    $config | Set-Content -Path $ConfigPath -Encoding ASCII
}

Write-Host "=========================================="
Write-Host "  $ScriptName"
Write-Host "=========================================="

Ensure-Node
Install-Codex

$ApiKey = Ensure-ApiKey
$env:NORDLYS_API_KEY = $ApiKey
[Environment]::SetEnvironmentVariable("NORDLYS_API_KEY", $ApiKey, "User")
Write-Success "Stored NORDLYS_API_KEY in user environment."

if (-not $Model) { $Model = $env:NORDLYS_MODEL }
if (-not $Model) { $Model = $DefaultModel }
if (-not $ModelProvider) { $ModelProvider = $env:NORDLYS_MODEL_PROVIDER }
if (-not $ModelProvider) { $ModelProvider = $DefaultModelProvider }

$configDir = Join-Path $HOME ".codex"
New-Item -ItemType Directory -Force -Path $configDir | Out-Null
$configPath = Join-Path $configDir "config.toml"
Write-Config -ConfigPath $configPath -ModelValue $Model -ProviderValue $ModelProvider

Write-Success "Codex configured for Nordlys."
Write-Host "Config: $configPath"
Write-Host "Run: codex --version"
