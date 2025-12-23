$ErrorActionPreference = "Stop"

param(
    [string]$ApiKey,
    [string]$Model
)

$ScriptName = "Grok CLI Nordlys Installer"
$NodeMinVersion = 18
$NodeInstallVersion = "22"
$PackageName = "@vibe-kit/grok-cli@0.0.16"
$ApiBaseUrl = "https://api.llmadaptive.uk/v1"
$DefaultModel = "nordlys/nordlys-code"
$DefaultModels = @(
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-4-5-haiku",
    "anthropic/claude-opus-4-1-20250805",
    "openai/gpt-5-codex",
    "google/gemini-2-5-pro"
)

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

function Install-GrokCli {
    if (Get-Command grok -ErrorAction SilentlyContinue) {
        Write-Success "Grok CLI already installed: $(grok --version)"
        return
    }

    Write-Info "Installing Grok CLI via npm..."
    npm install -g $PackageName | Out-Null
    if (-not (Get-Command grok -ErrorAction SilentlyContinue)) {
        Write-Failure "Grok CLI installation failed."
    }
    Write-Success "Grok CLI installed."
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

function Write-Settings {
    param(
        [string]$SettingsPath,
        [hashtable]$Settings
    )

    if (Test-Path $SettingsPath) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupPath = "$SettingsPath.$timestamp.bak"
        Copy-Item $SettingsPath $backupPath -Force
        Write-Info "Backed up existing settings to $backupPath"
    }

    $json = $Settings | ConvertTo-Json -Depth 6
    $json | Set-Content -Path $SettingsPath -Encoding ASCII
}

Write-Host "=========================================="
Write-Host "  $ScriptName"
Write-Host "=========================================="

Ensure-Node
Install-GrokCli

$ApiKey = Ensure-ApiKey
if (-not $Model) { $Model = $env:NORDLYS_MODEL }
if (-not $Model) { $Model = $DefaultModel }

$settingsDir = Join-Path $HOME ".grok"
New-Item -ItemType Directory -Force -Path $settingsDir | Out-Null
$settingsPath = Join-Path $settingsDir "user-settings.json"

$settings = @{
    apiKey = $ApiKey
    baseURL = $ApiBaseUrl
    defaultModel = $Model
    models = $DefaultModels
}

Write-Settings -SettingsPath $settingsPath -Settings $settings

$env:NORDLYS_API_KEY = $ApiKey
$env:NORDLYS_BASE_URL = $ApiBaseUrl
[Environment]::SetEnvironmentVariable("NORDLYS_API_KEY", $ApiKey, "User")
[Environment]::SetEnvironmentVariable("NORDLYS_BASE_URL", $ApiBaseUrl, "User")

Write-Success "Grok CLI configured for Nordlys."
Write-Host "Config: $settingsPath"
Write-Host "Run: grok --version"
