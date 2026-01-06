$ErrorActionPreference = "Stop"

param(
    [string]$ApiKey
)

$ScriptName = "Zed Editor Nordlys Installer"
$ScriptVersion = "1.0.0"
$ConfigFile = "$env:APPDATA\\Zed\\settings.json"
$ApiBaseUrl = "https://api.nordlyslabs.com/v1"
$DefaultModel = "nordlys/hypernova"
$DefaultProvider = "openai_compatible"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK]  $Message" -ForegroundColor Green
}

function Write-Failure {
    param([string]$Message)
    Write-Host "[ERR] $Message" -ForegroundColor Red
    exit 1
}

function Test-ZedInstalled {
    # Check if Zed config directory exists (more reliable than checking PATH)
    $configDir = "$env:APPDATA\Zed"
    
    if (Test-Path $configDir) {
        Write-Success "Zed editor config directory found"
        return $true
    }
    
    # Also check if zed is in PATH as fallback
    $zedPath = Get-Command zed -ErrorAction SilentlyContinue
    if ($zedPath) {
        Write-Success "Zed editor is installed"
        return $true
    }
    
    Write-Failure "Zed editor is not installed. Please install from https://zed.dev and run it at least once"
    return $false
}

function Get-NordlysApiKey {
    if ($ApiKey) {
        return $ApiKey
    }
    
    if ($env:NORDLYS_API_KEY) {
        Write-Info "Using API key from NORDLYS_API_KEY environment variable"
        return $env:NORDLYS_API_KEY
    }
    
    Write-Info "Get your API key from: https://nordlyslabs.com/api-platform/orgs"
    $key = Read-Host "Enter your Nordlys API key"
    
    if (-not $key) {
        Write-Failure "API key is required"
    }
    
    return $key
}

function Backup-Config {
    param([string]$ConfigPath)
    
    if (Test-Path $ConfigPath) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupPath = "$ConfigPath.$timestamp.bak"
        Copy-Item $ConfigPath $backupPath -Force
        Write-Success "Config backed up to: $backupPath"
        Write-Info "To revert: Copy-Item \"$backupPath\" \"$ConfigPath\" -Force"
    }
}

function Update-ZedConfig {
    param(
        [string]$ConfigPath,
        [string]$ApiKey
    )
    
    $configDir = Split-Path -Parent $ConfigPath
    if (-not (Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
        Write-Info "Created config directory: $configDir"
    }
    
    Backup-Config $ConfigPath
    
    # Read existing config or create new
    $config = @{}
    if (Test-Path $ConfigPath) {
        try {
            $content = Get-Content $ConfigPath -Raw
            # Remove comments for JSON parsing
            $cleanContent = ($content -split "`n" | Where-Object { $_ -notmatch "^\s*//" }) -join "`n"
            if ($cleanContent.Trim()) {
                $config = $cleanContent | ConvertFrom-Json -AsHashtable
            }
        }
        catch {
            Write-Info "Warning: Existing config has invalid JSON, creating new config"
            $config = @{}
        }
    }
    
    # Ensure language_models section exists
    if (-not $config.ContainsKey('language_models')) {
        $config['language_models'] = @{}
    }
    
    # Add openai_compatible section
    if (-not $config['language_models'].ContainsKey('openai_compatible')) {
        $config['language_models']['openai_compatible'] = @{}
    }
    
    # Configure Nordlys provider
    $config['language_models']['openai_compatible']['Nordlys'] = @{
        api_url = $ApiBaseUrl
        available_models = @(
            @{
                name = $DefaultModel
                display_name = 'Hypernova (Nordlys MoM)'
                max_tokens = 200000
                capabilities = @{
                    tools = $true
                    images = $true
                    parallel_tool_calls = $true
                    prompt_cache_key = $false
                }
            }
        )
    }
    
    # Update agent default model if not set
    if (-not $config.ContainsKey('agent')) {
        $config['agent'] = @{}
    }
    
    if (-not $config['agent'].ContainsKey('default_model')) {
        $config['agent']['default_model'] = @{
            provider = $DefaultProvider
            model = $DefaultModel
            name = 'Nordlys'
        }
    }
    
    # Write updated config
    $config | ConvertTo-Json -Depth 10 | Set-Content $ConfigPath -Encoding UTF8
    Write-Success "Zed configuration updated successfully"
}

function Set-EnvironmentVariable {
    param([string]$ApiKey)
    
    # Set for current session
    $env:NORDLYS_API_KEY = $ApiKey
    
    # Set for user permanently
    [Environment]::SetEnvironmentVariable('NORDLYS_API_KEY', $ApiKey, 'User')
    Write-Success "NORDLYS_API_KEY environment variable set"
    Write-Info "Restart your terminal or Zed editor to apply changes"
}

function Main {
    Write-Host ""
    Write-Host "===============================================" -ForegroundColor Yellow
    Write-Host "  $ScriptName v$ScriptVersion" -ForegroundColor Yellow
    Write-Host "===============================================" -ForegroundColor Yellow
    Write-Host ""
    
    # Check if Zed is installed
    if (-not (Test-ZedInstalled)) {
        return
    }
    
    # Get API key
    $apiKey = Get-NordlysApiKey
    
    # Update Zed configuration
    Update-ZedConfig -ConfigPath $ConfigFile -ApiKey $apiKey
    
    # Setup environment variable
    Set-EnvironmentVariable -ApiKey $apiKey
    
    Write-Host ""
    Write-Success "Installation complete!"
    Write-Host ""
    Write-Info "Next steps:"
    Write-Host "  1. Restart Zed editor"
    Write-Host "  2. Open Agent Panel (Ctrl + Shift + A)"
    Write-Host "  3. Select 'Hypernova (Nordlys MoM)' from model dropdown"
    Write-Host "  4. Start using Nordlys Mixture of Models!"
    Write-Host ""
    Write-Info "Documentation: https://docs.nordlyslabs.com/developer-tools/zed"
    Write-Host ""
}

Main
