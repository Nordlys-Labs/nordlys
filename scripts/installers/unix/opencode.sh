#!/bin/bash
# OpenCode + Nordlys one-shot installer
# Works on macOS/Linux/Windows (bash/PowerShell), needs curl

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="OpenCode Nordlys Installer"
SCRIPT_VERSION="1.0.1"

NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"

# ✅ Correct package for OpenCode:
OPENCODE_PACKAGE="opencode-ai"

CONFIG_FILE="opencode.json"
API_BASE_URL="https://api.llmadaptive.uk/v1"
API_KEY_URL="https://www.llmadaptive.uk/dashboard"

# Model override defaults:
# - use nordlys/nordlys-code for Nordlys model
DEFAULT_MODEL="nordlys/nordlys-code"

# ========================
#       Logging
# ========================
log_info()    { echo "🔹 $*"; }
log_success() { echo "✅ $*"; }
log_error()   { echo "❌ $*" >&2; }

create_config_backup() {
  local config_file="$1"

  if [ -f "$config_file" ]; then
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local timestamped_backup="${config_file}.${timestamp}.bak"

    # Create timestamped backup
    cp "$config_file" "$timestamped_backup" || {
      log_error "Failed to create timestamped backup: $timestamped_backup"
      exit 1
    }

    log_success "Config backed up to: $timestamped_backup"
    log_info "To revert: cp \"$timestamped_backup\" \"$config_file\""
  fi
}

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
  # Empty => falls back to nordlys/nordlys-code
  if [ -z "$model" ]; then return 0; fi
  [[ "$model" =~ ^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$ ]]
}

# ========================
#     Config generator
# ========================
create_opencode_config() {
  local config_file="$1"
  local model="$2"

  log_info "Creating OpenCode configuration..."
  create_config_backup "$config_file"

  # If user gave NORDLYS_MODEL, use it; else default to the Nordlys model id.
  local effective_model="$model"
  if [ -z "$effective_model" ]; then
    effective_model="$DEFAULT_MODEL"
  fi

  cat > "$config_file" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "nordlys": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nordlys",
      "options": {
        "baseURL": "$API_BASE_URL",
        "headers": { "User-Agent": "opencode-nordlys-integration" }
      },
      "models": {}
    }
  },
  "model": "${effective_model}"
}
EOF

  # quick JSON sanity check (jq optional)
  if command -v jq >/dev/null 2>&1; then
    if ! jq . "$config_file" >/dev/null 2>&1; then
      log_error "Generated $config_file is not valid JSON."
      exit 1
    fi
  fi

  log_success "OpenCode configuration written: $config_file"
}

# ========================
#     Auth guidance
# ========================
print_auth_instructions() {
  echo ""
  echo "⚠️  Authentication is interactive in OpenCode."
  echo "   Run the following to finish setup:"
  echo ""
  echo "   opencode auth login"
  echo "     → Select: Other"
  echo "     → Provider ID: nordlys"
  echo "     → Paste your API key (get it from $API_KEY_URL)"
  echo ""
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

  if [ ! -f "$PWD/$CONFIG_FILE" ]; then
    log_error "Missing $CONFIG_FILE in current directory."
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
  echo "Mixture of Models (save 60–80% costs)"
  echo ""
}

main() {
  show_banner

  # 1) Node & npm
  check_nodejs

  # 2) OpenCode CLI
  install_opencode

  # 3) Read env overrides (optional)
  local api_key="${NORDLYS_API_KEY:-}"
  local model_override="${NORDLYS_MODEL:-}"
  local model="$DEFAULT_MODEL"

  if [ -n "$model_override" ]; then
    if ! validate_model_override "$model_override"; then
      log_error "Invalid NORDLYS_MODEL: '$model_override'. Use format: author/model_id (e.g., nordlys/nordlys-code)."
      exit 1
    fi
    log_info "Using custom model override: $model_override"
    model="$model_override"
  else
    log_info "Using nordlys/nordlys-code Nordlys model (no explicit model override)."
  fi

  # 4) If API key is present, quick format check (we cannot inject it non-interactively)
  if [ -n "$api_key" ]; then
    if validate_api_key "$api_key"; then
      log_success "NORDLYS_API_KEY detected (format looks OK)."
      log_info "Note: OpenCode still requires an interactive 'auth login' to store the key."
    else
      log_error "NORDLYS_API_KEY format looks invalid. Re-check your key or omit the variable."
      exit 1
    fi
  else
    log_info "No NORDLYS_API_KEY in env. You can still complete auth interactively."
  fi

  # 5) Create per-project config
  create_opencode_config "$PWD/$CONFIG_FILE" "$model"

  # 6) Final instructions for auth
  print_auth_instructions

  # 7) Verify
  if verify_installation; then
    echo ""
    echo "╭──────────────────────────────────────────────╮"
    echo "│  🎉 OpenCode + Nordlys Setup Complete       │"
    echo "╰──────────────────────────────────────────────╯"
    echo ""
    echo "🚀 Quick Start"
    echo "   1) opencode auth login         # add 'nordlys' integration with your API key"
    echo "   2) opencode                    # open the TUI"
    echo "   3) /models                     # pick 'Nordlys / 🧠 Nordlys Model'"
    echo ""
    echo "🔍 Verify"
    echo "   opencode auth list             # should list 'nordlys'"
    echo "   cat $CONFIG_FILE               # see 'model': 'nordlys/nordlys-code'"
    echo ""
    echo "📊 Monitor"
    echo "   Dashboard: $API_KEY_URL"
    echo "   Config:    $PWD/$CONFIG_FILE"
  else
    echo ""
    log_error "Installation verification failed."
    echo ""
    echo "🔧 Manual fallback:"
    echo "   curl -o $CONFIG_FILE https://raw.githubusercontent.com/Egham-7/nordlys/main/examples/opencode.json"
    echo "   opencode auth login   # Other → integration id 'nordlys' → paste API key"
    exit 1
  fi
}

main "$@"
