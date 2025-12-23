$ErrorActionPreference = "Stop"

param(
    [string]$Model
)

$ScriptName = "OpenCode Nordlys Installer"
$NodeMinVersion = 18
$NodeInstallVersion = "22"
$PackageName = "opencode-ai"
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

function Install-OpenCode {
    if (Get-Command opencode -ErrorAction SilentlyContinue) {
        Write-Success "OpenCode already installed: $(opencode --version)"
        return
    }

    Write-Info "Installing OpenCode via npm..."
    npm install -g $PackageName | Out-Null
    if (-not (Get-Command opencode -ErrorAction SilentlyContinue)) {
        Write-Failure "OpenCode installation failed."
    }
    Write-Success "OpenCode installed."
}

function Write-Config {
    param([string]$ConfigPath, [string]$ModelValue)

    if (Test-Path $ConfigPath) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupPath = "$ConfigPath.$timestamp.bak"
        Copy-Item $ConfigPath $backupPath -Force
        Write-Info "Backed up existing config to $backupPath"
    }

    $config = @"
{
  "`$schema": "https://opencode.ai/config.json",
  "provider": {
    "nordlys": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nordlys",
      "options": {
        "baseURL": "$ApiBaseUrl",
        "headers": { "User-Agent": "opencode-nordlys-integration" }
      },
      "models": {}
    }
  },
  "model": "$ModelValue"
}
"@

    $config | Set-Content -Path $ConfigPath -Encoding ASCII
}

Write-Host "=========================================="
Write-Host "  $ScriptName"
Write-Host "=========================================="

Ensure-Node
Install-OpenCode

if (-not $Model) { $Model = $env:NORDLYS_MODEL }
if (-not $Model) { $Model = $DefaultModel }

$configPath = Join-Path (Get-Location) "opencode.json"
Write-Config -ConfigPath $configPath -ModelValue $Model

Write-Success "OpenCode configured for Nordlys."
Write-Host "Config: $configPath"
Write-Host "Run: opencode auth login"
