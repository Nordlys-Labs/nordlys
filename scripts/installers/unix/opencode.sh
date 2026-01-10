#!/bin/bash
# OpenCode + Nordlys one-shot installer
# Works on macOS/Linux/Windows (bash/PowerShell), needs curl

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="OpenCode Nordlys Installer"
SCRIPT_VERSION="1.1.0"

NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"

# ✅ Correct package for OpenCode:
OPENCODE_PACKAGE="opencode-ai"

CONFIG_DIR="$HOME/.config/opencode"
CONFIG_FILE="opencode.json"
API_BASE_URL="https://api.nordlyslabs.com/v1"
API_KEY_URL="https://nordlyslabs.com/api-platform/orgs"

# Model override defaults:
# - use nordlys/hypernova for Nordlys model
DEFAULT_MODEL="nordlys/hypernova"

# ========================
#       Logging
# ========================
log_info()    { echo "🔹 $*"; }
log_success() { echo "✅ $*"; }
log_error()   { echo "❌ $*" >&2; }

# ========================
#     Node.js helpers
# ========================
install_nodejs_windows() {
  log_info "Installing Node.js on Windows..."
  
  # Check if winget is available (Windows 10 1809+)
  if command -v winget >/dev/null 2>&1; then
    log_info "Installing Node.js via winget..."
    winget install OpenJS.NodeJS --version "$NODE_INSTALL_VERSION" --silent --accept-package-agreements --accept-source-agreements || {
      log_error "winget installation failed"
      return 1
    }
  # Check if chocolatey is available
  elif command -v choco >/dev/null 2>&1; then
    log_info "Installing Node.js via Chocolatey..."
    choco install nodejs --version="$NODE_INSTALL_VERSION" -y || {
      log_error "Chocolatey installation failed"
      return 1
    }
  else
    log_error "Neither winget nor Chocolatey found. Please install Node.js manually from https://nodejs.org/"
    log_info "After installing Node.js, re-run this script."
    exit 1
  fi
  
  # Refresh PATH
  log_info "Refreshing PATH environment..."
  export PATH="/c/Program Files/nodejs:$PATH"
  
  # Verify installation
  if command -v node >/dev/null 2>&1; then
    log_success "Node.js installed: $(node -v)"
    log_success "npm version: $(npm -v)"
  else
    log_error "Node.js installation verification failed. Please restart your terminal and try again."
    exit 1
  fi
}

install_nodejs() {
  local platform
  platform=$(uname -s)

  case "$platform" in
    Linux|Darwin)
      log_info "Installing Node.js on $platform..."
      log_info "Installing nvm ($NVM_VERSION)..."
      curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | bash

      # Load nvm in this shell
      if [ -s "$HOME/.nvm/nvm.sh" ]; then
        # shellcheck disable=SC1090
        . "$HOME/.nvm/nvm.sh"
      else
        log_error "nvm did not install correctly (missing ~/.nvm/nvm.sh)."
        exit 1
      fi

      log_info "Installing Node.js v$NODE_INSTALL_VERSION via nvm..."
      nvm install "$NODE_INSTALL_VERSION"

      node -v >/dev/null 2>&1 || { log_error "Node.js installation failed."; exit 1; }
      log_success "Node.js installed: $(node -v)"
      log_success "npm version: $(npm -v)"
      ;;
    MINGW*|MSYS*|CYGWIN*)
      log_info "Windows environment detected (Git Bash/MSYS2/Cygwin)..."
      install_nodejs_windows
      ;;
    *)
      log_error "Unsupported platform: $platform"
      log_info "Supported platforms: Linux, macOS, Windows (Git Bash/MSYS2/WSL)"
      exit 1
      ;;
  esac
}

check_nodejs() {
  if command -v node >/dev/null 2>&1; then
    local current_version major_version
    current_version=$(node -v | sed 's/^v//')
    major_version=$(echo "$current_version" | cut -d. -f1)
    if [ "$major_version" -ge "$NODE_MIN_VERSION" ]; then
      log_success "Node.js is already installed: v$current_version"
      return 0
    else
      log_info "Node v$current_version < $NODE_MIN_VERSION; updating..."
      install_nodejs
    fi
  else
    log_info "Node.js not found. Installing..."
    install_nodejs
  fi
}

# ========================
#     OpenCode install
# ========================
install_opencode() {
  if command -v opencode >/dev/null 2>&1; then
    log_success "OpenCode already installed: $(opencode --version 2>/dev/null || echo 'present')"
    return 0
  fi

  log_info "Installing OpenCode via npm (package: $OPENCODE_PACKAGE)..."
  npm install -g "$OPENCODE_PACKAGE" || {
    log_error "Failed to install OpenCode"
    exit 1
  }
  log_success "OpenCode installed via npm."
}

# ========================
#     Validation helpers
# ========================
validate_api_key() {
  local api_key="$1"
  [[ "$api_key" =~ ^[A-Za-z0-9._-]{20,}$ ]]
}

validate_model_override() {
  local model="$1"
  # Empty => falls back to nordlys/hypernova
  if [ -z "$model" ]; then return 0; fi
  [[ "$model" =~ ^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$ ]]
}

# ========================
#     API Key Configuration
# ========================
get_api_key() {
  local api_key="${NORDLYS_API_KEY:-}"

  # If API key is already set in env, validate and return it
  if [ -n "$api_key" ]; then
    if validate_api_key "$api_key"; then
      log_success "Using API key from NORDLYS_API_KEY environment variable" >&2
      echo "$api_key"
      return 0
    else
      log_error "NORDLYS_API_KEY format looks invalid. Re-check your key."
      exit 1
    fi
  fi

  # Check if running in non-interactive mode (e.g., piped from curl)
  if [ ! -t 0 ]; then
    echo "" >&2
    log_info "🎯 Interactive setup required for API key configuration" >&2
    echo "" >&2
    echo "📥 Option 1: Download and run interactively (Recommended)" >&2
    echo "   curl -o opencode.sh https://raw.githubusercontent.com/Nordlys-Labs/nordlys/main/scripts/installers/unix/opencode.sh" >&2
    echo "   chmod +x opencode.sh" >&2
    echo "   ./opencode.sh" >&2
    echo "" >&2
    echo "🔑 Option 2: Set API key via environment variable" >&2
    echo "   export NORDLYS_API_KEY='your-api-key-here'" >&2
    echo "   curl -fsSL https://raw.githubusercontent.com/Nordlys-Labs/nordlys/main/scripts/installers/unix/opencode.sh | bash" >&2
    echo "" >&2
    echo "🔗 Get your API key: $API_KEY_URL" >&2
    exit 1
  fi

  # Interactive mode - prompt for API key
  echo "" >&2
  log_info "You can get your API key from: $API_KEY_URL" >&2
  local attempts=0
  local max_attempts=3

  while [ $attempts -lt $max_attempts ]; do
    echo -n "🔑 Please enter your Nordlys API key: " >&2
    read -s api_key
    echo >&2

    if [ -z "$api_key" ]; then
      log_error "API key cannot be empty."
      ((attempts++))
      continue
    fi

    if validate_api_key "$api_key"; then
      echo "$api_key"
      return 0
    fi

    log_error "API key format appears invalid."
    ((attempts++))
    if [ $attempts -lt $max_attempts ]; then
      log_info "Please try again ($((max_attempts - attempts)) attempts remaining)..." >&2
    fi
  done

  log_error "Maximum attempts reached. Please run the script again."
  exit 1
}

# ========================
#     Config generator
# ========================
create_opencode_config() {
  local model="$1"
  local api_key="$2"

  log_info "Creating OpenCode configuration..."

  # Ensure config directory exists
  mkdir -p "$CONFIG_DIR"

  # Check for existing config (json or jsonc)
  local config_path=""
  if [ -f "$CONFIG_DIR/opencode.jsonc" ]; then
    config_path="$CONFIG_DIR/opencode.jsonc"
  else
    config_path="$CONFIG_DIR/opencode.json"
  fi

  # Backup existing config if present
  if [ -f "$config_path" ]; then
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_path="${config_path}.${timestamp}.bak"
    cp "$config_path" "$backup_path" || {
      log_error "Failed to create backup: $backup_path"
      exit 1
    }
    log_success "Config backed up to: $backup_path"
  fi

  # If user gave NORDLYS_MODEL, use it; else default to the Nordlys model id.
  local effective_model="$model"
  if [ -z "$effective_model" ]; then
    effective_model="$DEFAULT_MODEL"
  fi

  cat > "$config_path" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "nordlys": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nordlys",
      "options": {
        "baseURL": "$API_BASE_URL",
        "apiKey": "$api_key",
        "headers": { "User-Agent": "opencode-nordlys-integration" }
      },
      "models": {
        "${effective_model}": {
          "name": "Hypernova",
          "limit": {
            "context": 200000,
            "output": 65536
          }
        }
      }
    }
  },
  "model": "${effective_model}"
}
EOF

  # quick JSON sanity check (jq optional)
  if command -v jq >/dev/null 2>&1; then
    if ! jq . "$config_path" >/dev/null 2>&1; then
      log_error "Generated $config_path is not valid JSON."
      exit 1
    fi
  fi

  log_success "OpenCode configuration written: $config_path"
}

# ========================
#     Verification
# ========================
verify_installation() {
  log_info "Verifying installation..."

  if ! command -v opencode >/dev/null 2>&1; then
    log_error "OpenCode binary not found after install."
    return 1
  fi

  # Check for config in either json or jsonc format
  if [ ! -f "$CONFIG_DIR/opencode.json" ] && [ ! -f "$CONFIG_DIR/opencode.jsonc" ]; then
    log_error "Missing config in $CONFIG_DIR/"
    return 1
  fi

  log_success "Installation verification passed."
  return 0
}

# ========================
#        Main
# ========================
show_banner() {
  echo "=========================================="
  echo "  $SCRIPT_NAME v$SCRIPT_VERSION"
  echo "=========================================="
  echo "Configure OpenCode to use Nordlys's"
  echo "Mixture of Models for intelligent model selection"
  echo ""
}

main() {
  show_banner

  # 1) Node & npm
  check_nodejs

  # 2) OpenCode CLI
  install_opencode

  # 3) Get API key (from env or prompt)
  local api_key
  api_key=$(get_api_key)

  # 4) Read model override (optional)
  local model_override="${NORDLYS_MODEL:-}"
  local model="$DEFAULT_MODEL"

  if [ -n "$model_override" ]; then
    if ! validate_model_override "$model_override"; then
      log_error "Invalid NORDLYS_MODEL: '$model_override'. Use format: author/model_id (e.g., nordlys/hypernova)."
      exit 1
    fi
    log_info "Using custom model override: $model_override"
    model="$model_override"
  else
    log_info "Using nordlys/hypernova Mixture of Models."
  fi

  # 5) Create config in ~/.config/opencode/
  create_opencode_config "$model" "$api_key"

  # 6) Verify
  if verify_installation; then
    echo ""
    echo "╭──────────────────────────────────────────────╮"
    echo "│  🎉 OpenCode + Nordlys Setup Complete       │"
    echo "╰──────────────────────────────────────────────╯"
    echo ""
    echo "🚀 Quick Start"
    echo "   opencode                    # open the TUI"
    echo "   /models                     # select 'nordlys/hypernova'"
    echo ""
    echo "🔍 Verify"
    echo "   cat $CONFIG_DIR/$CONFIG_FILE"
    echo ""
    echo "📊 Monitor"
    echo "   Dashboard: $API_KEY_URL"
    echo ""
    echo "📖 Documentation: https://docs.nordlyslabs.com/developer-tools/opencode"
  else
    echo ""
    log_error "Installation verification failed."
    echo ""
    echo "🔧 Manual fallback:"
    echo "   mkdir -p $CONFIG_DIR"
    echo "   # Create $CONFIG_DIR/opencode.json with your config"
    exit 1
  fi
}

main "$@"
