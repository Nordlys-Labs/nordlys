#!/bin/bash

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="OpenAI Codex Nordlys Installer"
SCRIPT_VERSION="1.0.0"
CONFIG_DIR="$HOME/.codex"
API_BASE_URL="https://api.llmadaptive.uk/v1"
API_KEY_URL="https://www.llmadaptive.uk/dashboard"

# Model override defaults (can be overridden by environment variables)
# Use nordlys/nordlys-code to enable Nordlys model for optimal cost/performance
DEFAULT_MODEL="nordlys/nordlys-code"
DEFAULT_MODEL_PROVIDER="nordlys"

# ========================
#       Utility Functions
# ========================

log_info() {
  echo "🔹 $*"
}

log_success() {
  echo "✅ $*"
}

log_error() {
  echo "❌ $*" >&2
}

ensure_dir_exists() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    mkdir -p "$dir" || {
      log_error "Failed to create directory: $dir"
      exit 1
    }
  fi
}

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
#     Installation Detection
# ========================

detect_installation_method() {
  # Check if Codex is already installed
  if command -v codex &>/dev/null; then
    log_success "Codex is already installed: $(codex --version 2>/dev/null || echo 'installed')"
    return 0
  fi

  # Check for available installation methods (npm first as it's the official method)
  if command -v npm &>/dev/null; then
    log_info "npm detected - will use npm for installation (recommended)"
    INSTALL_METHOD="npm"
    return 0
  elif command -v brew &>/dev/null; then
    log_info "Homebrew detected - will use brew for installation"
    INSTALL_METHOD="brew"
    return 0
  elif command -v cargo &>/dev/null; then
    log_info "Cargo detected - will use cargo for installation"
    INSTALL_METHOD="cargo"
    return 0
  elif command -v curl &>/dev/null; then
    log_info "Will use direct download method"
    INSTALL_METHOD="download"
    return 0
  else
    log_error "No suitable installation method found. Please install Node.js/npm, Homebrew, Rust/Cargo, or ensure curl is available."
    exit 1
  fi
}

# ========================
#     Codex Installation
# ========================

install_codex_npm() {
  log_info "Installing Codex via npm..."
  npm install -g @openai/codex || {
    log_error "Failed to install Codex via npm"
    exit 1
  }
  log_success "Codex installed successfully via npm"
}

install_codex_brew() {
  log_info "Installing Codex via Homebrew..."
  brew install codex || {
    log_error "Failed to install Codex via Homebrew"
    exit 1
  }
  log_success "Codex installed successfully via Homebrew"
}

install_codex_cargo() {
  log_info "Installing Codex via Cargo..."
  cargo install codex || {
    log_error "Failed to install Codex via Cargo"
    exit 1
  }
  log_success "Codex installed successfully via Cargo"
}

install_codex_download() {
  log_info "Installing Codex via direct download..."
  local platform
  local arch
  platform=$(uname -s | tr '[:upper:]' '[:lower:]')
  arch=$(uname -m)

  # Normalize architecture names
  case "$arch" in
    x86_64) arch="x86_64" ;;
    arm64|aarch64) arch="aarch64" ;;
    *)
      log_error "Unsupported architecture: $arch"
      exit 1
      ;;
  esac

  local download_url="https://github.com/openai/codex/releases/latest/download/codex-${platform}-${arch}"
  local install_dir="$HOME/.local/bin"

  ensure_dir_exists "$install_dir"

  log_info "Downloading Codex for $platform-$arch..."
  curl -fsSL "$download_url" -o "$install_dir/codex" || {
    log_error "Failed to download Codex. Please check if the release exists for your platform."
    exit 1
  }

  chmod +x "$install_dir/codex" || {
    log_error "Failed to make Codex executable"
    exit 1
  }

  # Add to PATH if not already there
  if [[ ":$PATH:" != *":$install_dir:"* ]]; then
    echo "export PATH=\"\$PATH:$install_dir\"" >> "$HOME/.bashrc"
    echo "export PATH=\"\$PATH:$install_dir\"" >> "$HOME/.zshrc" 2>/dev/null || true
    export PATH="$PATH:$install_dir"
  fi

  log_success "Codex installed successfully via direct download"
}

install_codex() {
  if command -v codex &>/dev/null; then
    log_success "Codex is already installed"
    return 0
  fi

  case "${INSTALL_METHOD:-}" in
    npm) install_codex_npm ;;
    brew) install_codex_brew ;;
    cargo) install_codex_cargo ;;
    download) install_codex_download ;;
    *)
      log_error "Unknown installation method: ${INSTALL_METHOD:-}"
      exit 1
      ;;
  esac
}

# ========================
#     Configuration Management
# ========================

configure_nordlys_provider() {
  local config_file="$1"
  local model="$2"
  local model_provider="$3"

  # Check if config file exists and has content
  if [ -f "$config_file" ] && [ -s "$config_file" ]; then
    log_info "Existing Codex configuration found, adding Nordlys integration..."

    # Check if nordlys provider already exists
    if grep -q "\[model_providers\.nordlys\]" "$config_file" 2>/dev/null; then
      log_info "Nordlys integration already configured, updating..."
      # Remove existing nordlys provider section
      sed -i '/\[model_providers\.nordlys\]/,/^$/d' "$config_file"
      sed -i '/\[model_providers\.nordlys\]/,/^\[/{ /^\[/!d; }' "$config_file"
    fi

    # Add nordlys provider to existing config
    {
      echo ""
      echo "[model_providers.nordlys]"
      echo "name = \"Nordlys\""
      echo "base_url = \"$API_BASE_URL\""
      echo "env_key = \"NORDLYS_API_KEY\""
      echo "wire_api = \"chat\""
    } >> "$config_file"

    # Update model_provider to nordlys if not already set to a specific provider
    if ! grep -q "^model_provider" "$config_file" 2>/dev/null; then
      # Add model_provider if it doesn't exist
      sed -i "1i model_provider = \"$model_provider\"" "$config_file"
    elif [ "$model_provider" = "nordlys" ]; then
      # Only update to nordlys if that's what we want
      sed -i "s/^model_provider = .*/model_provider = \"$model_provider\"/" "$config_file"
    fi

    # Update model if specified and not already set
    if [ -n "$model" ] && ! grep -q "^model = " "$config_file" 2>/dev/null; then
      sed -i "1i model = \"$model\"" "$config_file"
    elif [ -n "$model" ] && [ "$model" != "$DEFAULT_MODEL" ]; then
      sed -i "s/^model = .*/model = \"$model\"/" "$config_file"
    fi

    log_success "Nordlys integration added to existing configuration"
  else
    log_info "Creating new Codex configuration with Nordlys integration..."
    # Create new config file
    cat > "$config_file" << EOF
# Nordlys Configuration
model = "$model"
model_provider = "$model_provider"
approval_policy = "untrusted"

[model_providers.nordlys]
name = "Nordlys"
base_url = "$API_BASE_URL"
env_key = "NORDLYS_API_KEY"
wire_api = "chat"
EOF
    log_success "New Codex configuration created with Nordlys integration"
  fi
}

# ========================
#     API Key Configuration
# ========================

validate_api_key() {
  local api_key="$1"

  # Basic validation - check if it looks like a valid API key format
  if [[ ! "$api_key" =~ ^[A-Za-z0-9_-]{20,}$ ]]; then
    log_error "API key format appears invalid. Please check your key."
    return 1
  fi
  return 0
}

validate_model_override() {
  local model="$1"

  # Empty values fall back to nordlys/nordlys-code for backward compatibility
  if [ -z "$model" ]; then
    return 0
  fi

  # Validate format: author/model_id
  if [[ ! "$model" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]; then
    log_error "Model format invalid. Use format: author/model_id (e.g., nordlys/nordlys-code)"
    return 1
  fi
  return 0
}

configure_codex() {
  log_info "Configuring Codex for Nordlys..."
  echo "   You can get your API key from: $API_KEY_URL"

  # Check for environment variable first
  local api_key="${NORDLYS_API_KEY:-}"

  # Check for model overrides
  local model="${NORDLYS_MODEL:-$DEFAULT_MODEL}"
  local model_provider="${NORDLYS_MODEL_PROVIDER:-$DEFAULT_MODEL_PROVIDER}"

  # Validate model override if provided
  if [ "$model" != "$DEFAULT_MODEL" ]; then
    log_info "Using custom model: $model"
    if ! validate_model_override "$model"; then
      log_error "Invalid model format in NORDLYS_MODEL"
      exit 1
    fi
  fi

  if [ -n "$api_key" ]; then
    log_info "Using API key from NORDLYS_API_KEY environment variable"
    if ! validate_api_key "$api_key"; then
      log_error "Invalid API key format in NORDLYS_API_KEY environment variable"
      exit 1
    fi
  # Check if running in non-interactive mode (e.g., piped from curl)
  elif [ ! -t 0 ]; then
    echo ""
    log_info "🎯 Interactive setup required for API key configuration"
    echo ""
    echo "📥 Option 1: Download and run interactively (Recommended)"
    echo "   curl -o codex.sh https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/codex.sh"
    echo "   chmod +x codex.sh"
    echo "   ./codex.sh"
    echo ""
    echo "🔑 Option 2: Set API key via environment variable"
    echo "   export NORDLYS_API_KEY='your-api-key-here'"
    echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/codex.sh | bash"
    echo ""
    echo "🎯 Option 3: Customize model (Advanced)"
    echo "   export NORDLYS_API_KEY='your-api-key-here'"
    echo "   export NORDLYS_MODEL='nordlys/nordlys-code'"
    echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/codex.sh | bash"
    echo ""
    echo "⚙️  Option 4: Manual configuration (Advanced users)"
    echo "   mkdir -p ~/.codex"
    echo "   cat > ~/.codex/config.toml << 'EOF'"
    echo "model = \"nordlys/nordlys-code\""
    echo "model_provider = \"nordlys\""
    echo ""
    echo "[model_providers.nordlys]"
    echo "name = \"Nordlys\""
    echo "base_url = \"https://www.llmadaptive.uk/api/v1\""
    echo "env_key = \"NORDLYS_API_KEY\""
    echo "wire_api = \"chat\""
    echo "EOF"
    echo ""
    echo "🔗 Get your API key: $API_KEY_URL"
    exit 1
  else
    # Interactive mode - prompt for API key
    local attempts=0
    local max_attempts=3

    while [ $attempts -lt $max_attempts ]; do
      echo -n "🔑 Please enter your Nordlys API key: "
      read -rs api_key
      echo

      if [ -z "$api_key" ]; then
        log_error "API key cannot be empty."
        ((attempts++))
        continue
      fi

      if validate_api_key "$api_key"; then
        break
      fi

      ((attempts++))
      if [ $attempts -lt $max_attempts ]; then
        log_info "Please try again ($((max_attempts - attempts)) attempts remaining)..."
      fi
    done

    if [ $attempts -eq $max_attempts ]; then
      log_error "Maximum attempts reached. Please run the script again."
      exit 1
    fi
  fi

  ensure_dir_exists "$CONFIG_DIR"

  # Configure Codex with Nordlys integration
  local config_file="$CONFIG_DIR/config.toml"
  create_config_backup "$config_file"
  configure_nordlys_provider "$config_file" "$model" "$model_provider"

  # Set environment variable for the session
  export NORDLYS_API_KEY="$api_key"

  # Add to shell profiles for persistence
  local env_line
  env_line="export NORDLYS_API_KEY='$api_key'"

  # Detect current shell and add to appropriate profile
  local current_shell
  local profile_updated=false
  current_shell=$(basename "${SHELL:-/bin/bash}")

  case "$current_shell" in
    zsh)
      if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "NORDLYS_API_KEY" "$HOME/.zshrc" 2>/dev/null; then
          {
            echo ""
            echo "# Nordlys API Key"
            echo "$env_line"
          } >> "$HOME/.zshrc"
          log_info "Added API key to ~/.zshrc"
          profile_updated=true
        fi
      fi
      ;;
    bash)
      if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "NORDLYS_API_KEY" "$HOME/.bashrc" 2>/dev/null; then
          {
            echo ""
            echo "# Nordlys API Key"
            echo "$env_line"
          } >> "$HOME/.bashrc"
          log_info "Added API key to ~/.bashrc"
          profile_updated=true
        fi
      fi
      ;;
    fish)
      # Fish shell uses different syntax
      local fish_config_dir="$HOME/.config/fish"
      local fish_config="$fish_config_dir/config.fish"
      ensure_dir_exists "$fish_config_dir"
      if [ -f "$fish_config" ]; then
        if ! grep -q "NORDLYS_API_KEY" "$fish_config" 2>/dev/null; then
          {
            echo ""
            echo "# Nordlys API Key"
            echo "set -gx NORDLYS_API_KEY '$api_key'"
          } >> "$fish_config"
          log_info "Added API key to ~/.config/fish/config.fish"
          profile_updated=true
        fi
      fi
      ;;
    *)
      log_info "Unknown shell: $current_shell, trying common profile files..."
      ;;
  esac

  # Fallback: try common profile files if shell-specific config didn't work
  if [ "$profile_updated" = false ]; then
    # Try .profile (POSIX-compliant, works with most shells)
    if [ -f "$HOME/.profile" ]; then
      if ! grep -q "NORDLYS_API_KEY" "$HOME/.profile" 2>/dev/null; then
        {
          echo ""
          echo "# Nordlys API Key"
          echo "$env_line"
        } >> "$HOME/.profile"
        log_info "Added API key to ~/.profile"
        profile_updated=true
      fi
    else
      # Create .profile if it doesn't exist
      {
        echo "# Nordlys API Key"
        echo "$env_line"
      } > "$HOME/.profile"
      log_info "Created ~/.profile and added API key"
      profile_updated=true
    fi

    # Also try shell-specific files as backup
    for profile in ".zshrc" ".bashrc"; do
      if [ -f "$HOME/$profile" ] && ! grep -q "NORDLYS_API_KEY" "$HOME/$profile" 2>/dev/null; then
        {
          echo ""
          echo "# Nordlys API Key"
          echo "$env_line"
        } >> "$HOME/$profile"
        log_info "Added API key to ~/$profile"
      fi
    done
  fi

  log_success "Codex configured for Nordlys successfully"
  log_info "Configuration saved to: $config_file"
  log_info "Environment variable added to shell profiles"
}

# ========================
#        Main Flow
# ========================

show_banner() {
  echo "=========================================="
  echo "  $SCRIPT_NAME v$SCRIPT_VERSION"
  echo "=========================================="
  echo "Configure OpenAI Codex to use Nordlys's"
  echo "Mixture of Models for 60-80% cost savings"
  echo ""
}

verify_installation() {
  log_info "Verifying installation..."

  # Check if Codex can be found
  if ! command -v codex &>/dev/null; then
    log_error "Codex installation verification failed"
    return 1
  fi

  # Check if configuration file exists
  if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    log_error "Configuration file not found"
    return 1
  fi

  # Check if environment variable is set
  if [ -z "${NORDLYS_API_KEY:-}" ]; then
    log_error "NORDLYS_API_KEY environment variable not set"
    return 1
  fi

  log_success "Installation verification passed"
  return 0
}

main() {
  show_banner

  detect_installation_method
  install_codex
  configure_codex

  if verify_installation; then
    echo ""
    echo "╭──────────────────────────────────────────╮"
    echo "│  🎉 Codex + Nordlys Setup Complete      │"
    echo "╰──────────────────────────────────────────╯"
    echo ""
    echo "🚀 Quick Start:"
    echo "   codex                     # Start Codex with Nordlys model"
    echo "   codex --model nordlys/nordlys-code  # Explicit Nordlys model"
    echo ""
    echo "🔍 Verify Setup:"
    echo "   codex --version           # Check Codex installation"
    echo "   cat ~/.codex/config.toml  # View configuration"
    echo "   echo \$NORDLYS_API_KEY    # Check API key"
    echo ""
    echo "💡 Usage Examples:"
    echo "   codex                     # Interactive mode"
    echo "   codex exec \"create a React component for user auth\""
    echo "   codex --model anthropic/claude-sonnet-4-5"
    echo "   codex --sandbox read-only # Secure sandbox mode"
    echo ""
    echo "📊 Monitor Usage:"
    echo "   Dashboard: $API_KEY_URL"
    echo "   Configuration: ~/.codex/config.toml"
    echo "   Environment: \$NORDLYS_API_KEY"
    echo ""
    echo "💡 Pro Tips:"
    echo "   • Nordlys model enabled by default"
    echo "   • Use --sandbox workspace-write for file editing tasks"
    echo "   • Configure MCP servers for extended capabilities"
    echo "   • Create AGENTS.md for project-specific instructions"
    echo ""
    echo "📖 Full Documentation: https://docs.llmadaptive.uk/developer-tools/codex"
    echo "🐛 Report Issues: https://github.com/Egham-7/nordlys/issues"
    echo ""
    local current_shell
    current_shell=$(basename "${SHELL:-/bin/bash}")
    case "$current_shell" in
      zsh)
        echo "⚠️  Important: Restart your terminal or run 'source ~/.zshrc' to load environment variables"
        ;;
      bash)
        echo "⚠️  Important: Restart your terminal or run 'source ~/.bashrc' to load environment variables"
        ;;
      fish)
        echo "⚠️  Important: Restart your terminal or start a new fish session to load environment variables"
        ;;
      *)
        echo "⚠️  Important: Restart your terminal or run 'source ~/.profile' to load environment variables"
        ;;
    esac
  else
    echo ""
    log_error "❌ Installation verification failed"
    echo ""
    echo "🔧 Manual Setup (if needed):"
    echo "   Configuration: ~/.codex/config.toml"
    echo "   Environment: export NORDLYS_API_KEY='your-key'"
    echo "   Expected config format:"
    echo '   model = "nordlys/nordlys-code"'
    echo '   model_provider = "nordlys"'
    echo '   [model_providers.nordlys]'
    echo '   name = "Nordlys"'
    echo '   base_url = "https://www.llmadaptive.uk/api/v1"'
    echo '   env_key = "NORDLYS_API_KEY"'
    echo ""
    echo "🆘 Get help: https://docs.llmadaptive.uk/troubleshooting"
    exit 1
  fi
}

main "$@"
