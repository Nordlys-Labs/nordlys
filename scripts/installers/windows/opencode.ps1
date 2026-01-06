$ErrorActionPreference = "Stop"

param(
    [string]$Model,
    [string]$ApiKey
)

$ScriptName = "OpenCode Nordlys Installer"
$NodeMinVersion = 18
$NodeInstallVersion = "22"
$PackageName = "opencode-ai"
$ApiBaseUrl = "https://api.nordlyslabs.com/v1"
$ApiKeyUrl = "https://nordlyslabs.com/api-platform/orgs"
$DefaultModel = "nordlys/hypernova"
$ConfigDir = Join-Path $env:USERPROFILE ".config\opencode"

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

function Validate-ApiKey {
    param([string]$Key)
    return $Key -match "^[A-Za-z0-9._-]{20,}$"
}

function Get-ApiKey {
    # Check parameter first
    if ($ApiKey) {
        if (Validate-ApiKey $ApiKey) {
            Write-Success "Using API key from parameter"
            return $ApiKey
        } else {
            Write-Failure "API key format appears invalid."
        }
    }

    # Check environment variable
    $envKey = $env:NORDLYS_API_KEY
    if ($envKey) {
        if (Validate-ApiKey $envKey) {
            Write-Success "Using API key from NORDLYS_API_KEY environment variable"
            return $envKey
        } else {
            Write-Failure "NORDLYS_API_KEY format appears invalid."
        }
    }

    # Interactive prompt
    Write-Host ""
    Write-Info "You can get your API key from: $ApiKeyUrl"
    $attempts = 0
    $maxAttempts = 3

    while ($attempts -lt $maxAttempts) {
        $secureKey = Read-Host "Enter your Nordlys API key" -AsSecureString
        $key = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey))

        if (-not $key) {
            Write-Host "[ERR] API key cannot be empty."
            $attempts++
            continue
        }

        if (Validate-ApiKey $key) {
            return $key
        }

        Write-Host "[ERR] API key format appears invalid."
        $attempts++
        if ($attempts -lt $maxAttempts) {
            Write-Info "Please try again ($($maxAttempts - $attempts) attempts remaining)..."
        }
    }

    Write-Failure "Maximum attempts reached. Please run the script again."
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
    param(
        [string]$ModelValue,
        [string]$ApiKeyValue
    )

    # Ensure config directory exists
    if (-not (Test-Path $ConfigDir)) {
        New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
    }

    # Check for existing config (json or jsonc)
    $configPath = Join-Path $ConfigDir "opencode.json"
    if (Test-Path (Join-Path $ConfigDir "opencode.jsonc")) {
        $configPath = Join-Path $ConfigDir "opencode.jsonc"
    }

    if (Test-Path $configPath) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupPath = "$configPath.$timestamp.bak"
        Copy-Item $configPath $backupPath -Force
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
        "apiKey": "$ApiKeyValue",
        "headers": { "User-Agent": "opencode-nordlys-integration" }
      },
      "models": {
        "$ModelValue": {
          "name": "Hypernova",
          "limit": {
            "context": 200000,
            "output": 65536
          }
        }
      }
    }
  },
  "model": "$ModelValue"
}
"@

    $config | Set-Content -Path $configPath -Encoding ASCII
    Write-Success "OpenCode configuration written: $configPath"
}

# Main script
Write-Host "=========================================="
Write-Host "  $ScriptName"
Write-Host "=========================================="
Write-Host "Configure OpenCode to use Nordlys's"
Write-Host "Mixture of Models for intelligent model selection"
Write-Host ""

Ensure-Node
Install-OpenCode

# Get API key (from param, env, or prompt)
$apiKeyValue = Get-ApiKey

# Get model
if (-not $Model) { $Model = $env:NORDLYS_MODEL }
if (-not $Model) { $Model = $DefaultModel }

Write-Info "Using model: $Model"

Write-Config -ModelValue $Model -ApiKeyValue $apiKeyValue

Write-Host ""
Write-Host "============================================"
Write-Host "  OpenCode + Nordlys Setup Complete"
Write-Host "============================================"
Write-Host ""
Write-Host "Quick Start:"
Write-Host "   opencode                    # open the TUI"
Write-Host "   /models                     # select 'nordlys/hypernova'"
Write-Host ""
Write-Host "Verify:"
Write-Host "   cat $ConfigDir\opencode.json"
Write-Host ""
Write-Host "Monitor:"
Write-Host "   Dashboard: $ApiKeyUrl"
Write-Host ""
Write-Host "Documentation: https://docs.nordlyslabs.com/developer-tools/opencode"
