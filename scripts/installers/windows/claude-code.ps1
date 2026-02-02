param(
    [string]$ApiKey,
    [string]$PrimaryModel
)

$ErrorActionPreference = "Stop"

$ScriptName = "Claude Code Nordlys Installer"
$NodeMinVersion = 18
$NodeInstallVersion = "22"
$PackageName = "@anthropic-ai/claude-code"
$ApiBaseUrl = "https://api.nordlyslabs.com"
$ApiTimeoutMs = 3000000
$DefaultPrimaryModel = "nordlys/hypernova"

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

function Install-ClaudeCode {
    if (Get-Command claude -ErrorAction SilentlyContinue) {
        Write-Success "Claude Code already installed: $(claude --version)"
        return
    }

    Write-Info "Installing Claude Code via npm..."
    npm install -g $PackageName | Out-Null
    if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
        Write-Failure "Claude Code installation failed."
    }
    Write-Success "Claude Code installed."
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
        [hashtable]$EnvValues
    )

    if (Test-Path $SettingsPath) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupPath = "$SettingsPath.$timestamp.bak"
        Copy-Item $SettingsPath $backupPath -Force
        Write-Info "Backed up existing settings to $backupPath"
    }

    $settings = @{}
    if (Test-Path $SettingsPath) {
        try {
            $existing = Get-Content $SettingsPath -Raw | ConvertFrom-Json
            if ($existing) {
                $settings = $existing
            }
        } catch {
            Write-Info "Existing settings.json is not valid JSON. Replacing."
            $settings = @{}
        }
    }

    if (-not $settings.env) {
        $settings | Add-Member -MemberType NoteProperty -Name env -Value @{}
    }

    $settings.model = $EnvValues["_MODEL_OVERRIDE"]
    $EnvValues.Remove("_MODEL_OVERRIDE")

    foreach ($key in $EnvValues.Keys) {
        $settings.env.$key = $EnvValues[$key]
    }

    $json = $settings | ConvertTo-Json -Depth 10
    $json | Set-Content -Path $SettingsPath -Encoding ASCII
}

function Launch-Tool {
    Write-Info "Launching Claude Code..."
    
    $toolCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($toolCmd) {
        try {
            & claude
        }
        catch {
            Write-Host ""
            Write-Host "To launch manually, run:"
            Write-Host "   claude"
            Write-Host ""
        }
    }
    else {
        Write-Host ""
        Write-Host "To launch manually, run:"
        Write-Host "   claude"
        Write-Host ""
    }
}

Write-Host "=========================================="
Write-Host "  $ScriptName"
Write-Host "=========================================="

Ensure-Node
Install-ClaudeCode

$ApiKey = Ensure-ApiKey

if (-not $PrimaryModel) { $PrimaryModel = $env:NORDLYS_PRIMARY_MODEL }
if (-not $PrimaryModel) { $PrimaryModel = $DefaultPrimaryModel }

$configDir = Join-Path $HOME ".claude"
New-Item -ItemType Directory -Force -Path $configDir | Out-Null
$settingsPath = Join-Path $configDir "settings.json"

$envValues = @{
    ANTHROPIC_AUTH_TOKEN = $ApiKey
    ANTHROPIC_BASE_URL = $ApiBaseUrl
    API_TIMEOUT_MS = $ApiTimeoutMs
    _MODEL_OVERRIDE = $PrimaryModel
}

Write-Settings -SettingsPath $settingsPath -EnvValues $envValues

Write-Success "Claude Code configured for Nordlys."
Write-Host "Config: $settingsPath"
Write-Host "Run: claude"
Write-Host ""

Launch-Tool
