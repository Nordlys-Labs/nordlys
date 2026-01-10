#!/bin/bash

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="Gemini CLI Nordlys Installer"
SCRIPT_VERSION="1.0.0"
NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"
GEMINI_PACKAGE="@google/gemini-cli"
CONFIG_DIR="$HOME/.gemini"
API_BASE_URL="https://api.nordlyslabs.com"
API_KEY_URL="https://nordlyslabs.com/api-platform/orgs"

# Model override defaults (can be overridden by environment variables)
# Use nordlys-hypernova to enable Nordlys model for optimal cost/performance
DEFAULT_MODEL="nordlys-hypernova"

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
		local timestamp
		timestamp=$(date +"%Y%m%d_%H%M%S")
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

# Portable sed in-place edit (works on both macOS/BSD and Linux)
portable_sed() {
	local script="$1"
	local file="$2"
	local tmp
	tmp=$(mktemp) || {
		log_error "Failed to create temp file"
		exit 1
	}
	sed "$script" "$file" >"$tmp" || {
		rm -f "$tmp"
		log_error "sed failed"
		exit 1
	}
	mv "$tmp" "$file"
}

# ========================
#     Runtime Installation Functions
# ========================

install_nodejs() {
	local platform
	platform=$(uname -s)

	case "$platform" in
	Linux | Darwin)
		log_info "Installing Node.js on $platform..."

		# Install nvm
		log_info "Installing nvm ($NVM_VERSION)..."
		curl -s https://raw.githubusercontent.com/nvm-sh/nvm/"$NVM_VERSION"/install.sh | bash

		# Load nvm
		log_info "Loading nvm environment..."
		# shellcheck source=/dev/null
		\. "$HOME/.nvm/nvm.sh"

		# Install Node.js
		log_info "Installing Node.js $NODE_INSTALL_VERSION..."
		nvm install "$NODE_INSTALL_VERSION"

		# Verify installation
		node -v &>/dev/null || {
			log_error "Node.js installation failed"
			exit 1
		}
		log_success "Node.js installed: $(node -v)"
		log_success "npm version: $(npm -v)"
		;;
	*)
		log_error "Unsupported platform: $platform"
		exit 1
		;;
	esac
}

# ========================
#     Runtime Check Functions
# ========================

check_nodejs() {
	if command -v node &>/dev/null; then
		current_version=$(node -v | sed 's/v//')
		major_version=$(echo "$current_version" | cut -d. -f1)

		if [ "$major_version" -ge "$NODE_MIN_VERSION" ]; then
			log_success "Node.js is already installed: v$current_version"
			return 0
		else
			log_info "Node.js v$current_version is installed but version < $NODE_MIN_VERSION. Upgrading..."
			install_nodejs
		fi
	else
		log_info "Node.js not found. Installing..."
		install_nodejs
	fi
}

check_runtime() {
	# Check for Node.js and npm
	if command -v node &>/dev/null && command -v npm &>/dev/null; then
		current_version=$(node -v | sed 's/v//')
		major_version=$(echo "$current_version" | cut -d. -f1)

		if [ "$major_version" -ge "$NODE_MIN_VERSION" ]; then
			log_info "Using Node.js runtime"
			INSTALL_CMD="npm install -g"
			return 0
		fi
	fi

	# Install Node.js if not found or version is too old
	log_info "No suitable runtime found. Installing Node.js..."
	check_nodejs
	INSTALL_CMD="npm install -g"
}

# ========================
#     Gemini CLI Installation
# ========================

install_gemini_cli() {
	if command -v gemini &>/dev/null; then
		log_success "Gemini CLI is already installed: $(gemini --version 2>/dev/null || echo 'installed')"
	else
		log_info "Installing Gemini CLI..."
		$INSTALL_CMD "$GEMINI_PACKAGE" || {
			log_error "Failed to install gemini-cli"
			exit 1
		}
		log_success "Gemini CLI installed successfully"
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

detect_shell() {
	# Check SHELL environment variable first (most reliable for installer scripts)
	case "${SHELL:-}" in
	*/zsh) echo "zsh" ;;
	*/bash) echo "bash" ;;
	*/fish) echo "fish" ;;
	*)
		# Fallback to checking version variables if SHELL is not set
		if [ -n "${ZSH_VERSION:-}" ]; then
			echo "zsh"
		elif [ -n "${BASH_VERSION:-}" ]; then
			echo "bash"
		elif [ -n "${FISH_VERSION:-}" ]; then
			echo "fish"
		else
			echo "bash" # Default fallback
		fi
		;;
	esac
}

get_shell_config_file() {
	local shell_type="$1"

	case "$shell_type" in
	zsh)
		echo "$HOME/.zshrc"
		;;
	bash)
		if [ -f "$HOME/.bashrc" ]; then
			echo "$HOME/.bashrc"
		elif [ -f "$HOME/.bash_profile" ]; then
			echo "$HOME/.bash_profile"
		else
			echo "$HOME/.bashrc"
		fi
		;;
	fish)
		mkdir -p "$HOME/.config/fish"
		echo "$HOME/.config/fish/config.fish"
		;;
	*)
		echo "$HOME/.bashrc"
		;;
	esac
}

add_env_to_shell_config() {
	local api_key="$1"
	local base_url="$2"
	local shell_type
	local config_file

	shell_type=$(detect_shell)
	config_file=$(get_shell_config_file "$shell_type")

	log_info "Adding environment variables to $config_file"

	# Create config file if it doesn't exist
	touch "$config_file"

	# Check if GEMINI_API_KEY already exists in the config
	if grep -q "GEMINI_API_KEY" "$config_file" 2>/dev/null; then
		log_info "Gemini environment variables already exist in $config_file, updating..."

		if [ "$shell_type" = "fish" ]; then
			# Fish shell: update API key and base URL
			portable_sed "s|set -x GEMINI_API_KEY.*|set -x GEMINI_API_KEY \"$api_key\"|" "$config_file"
			portable_sed "s|set -x GOOGLE_GEMINI_BASE_URL.*|set -x GOOGLE_GEMINI_BASE_URL \"$base_url\"|" "$config_file"

			# Add GOOGLE_GEMINI_BASE_URL if it doesn't exist in Fish config
			if ! grep -q "GOOGLE_GEMINI_BASE_URL" "$config_file" 2>/dev/null; then
				echo "set -x GOOGLE_GEMINI_BASE_URL \"$base_url\"" >>"$config_file"
			fi
			# Remove GEMINI_MODEL from Fish config if it exists (now in settings.json)
			if grep -q "GEMINI_MODEL" "$config_file" 2>/dev/null; then
				portable_sed "/set -x GEMINI_MODEL/d" "$config_file"
				log_info "Removed GEMINI_MODEL from shell config (now in settings.json)"
			fi
		else
			# POSIX shells (bash/zsh): update API key and base URL
			portable_sed "s|export GEMINI_API_KEY=.*|export GEMINI_API_KEY=\"$api_key\"|" "$config_file"
			portable_sed "s|export GOOGLE_GEMINI_BASE_URL=.*|export GOOGLE_GEMINI_BASE_URL=\"$base_url\"|" "$config_file"

			# Add GOOGLE_GEMINI_BASE_URL if it doesn't exist in POSIX shell config
			if ! grep -q "GOOGLE_GEMINI_BASE_URL" "$config_file" 2>/dev/null; then
				echo "export GOOGLE_GEMINI_BASE_URL=\"$base_url\"" >>"$config_file"
			fi
			# Remove GEMINI_MODEL from POSIX shell config if it exists (now in settings.json)
			if grep -q "export GEMINI_MODEL" "$config_file" 2>/dev/null; then
				portable_sed "/export GEMINI_MODEL/d" "$config_file"
				log_info "Removed GEMINI_MODEL from shell config (now in settings.json)"
			fi
		fi
	else
		# Add new environment variables based on shell type (no GEMINI_MODEL - it's in settings.json)
		echo "" >>"$config_file"
		echo "# Gemini CLI with NordlysProxy Configuration (added by gemini-cli installer)" >>"$config_file"
		if [ "$shell_type" = "fish" ]; then
			echo "set -x GEMINI_API_KEY \"$api_key\"" >>"$config_file"
			echo "set -x GOOGLE_GEMINI_BASE_URL \"$base_url\"" >>"$config_file"
		else
			echo "export GEMINI_API_KEY=\"$api_key\"" >>"$config_file"
			echo "export GOOGLE_GEMINI_BASE_URL=\"$base_url\"" >>"$config_file"
		fi
	fi

	log_success "Environment variables added to $config_file"
	log_info "Model configuration is in ~/.gemini/settings.json"
	if [ "$shell_type" = "fish" ]; then
		log_info "Restart your terminal or run 'source $config_file' to apply changes"
	else
		log_info "Run 'source $config_file' or restart your terminal to apply changes"
	fi
}

create_settings_json() {
	local model="$1"
	local settings_file="$CONFIG_DIR/settings.json"

	log_info "Creating Gemini CLI settings at $settings_file"

	# Create config directory if needed
	ensure_dir_exists "$CONFIG_DIR"

	# Backup existing settings
	create_config_backup "$settings_file"

	# Try jq first for proper merging, fallback to simple creation
	if command -v jq &>/dev/null && [ -f "$settings_file" ]; then
		# Merge with existing settings using jq
		jq --arg model "$model" '
      .model.name = $model |
      .privacy.usageStatisticsEnabled = false |
      .modelConfigs.customAliases = {
        "summarizer-default": {
          "modelConfig": {
            "model": $model,
            "generateContentConfig": {
              "maxOutputTokens": 2000,
              "temperature": 0.2
            }
          }
        },
        "summarizer-shell": {
          "modelConfig": {
            "model": $model,
            "generateContentConfig": {
              "maxOutputTokens": 2000,
              "temperature": 0
            }
          }
        },
        "classifier": {
          "modelConfig": {
            "model": $model
          }
        },
        "prompt-completion": {
          "modelConfig": {
            "model": $model
          }
        },
        "edit-corrector": {
          "modelConfig": {
            "model": $model
          }
        },
        "web-search": {
          "modelConfig": {
            "model": $model
          }
        },
        "web-fetch": {
          "modelConfig": {
            "model": $model
          }
        }
      }
    ' "$settings_file" >"${settings_file}.tmp" && mv "${settings_file}.tmp" "$settings_file"

		log_success "Settings merged with existing configuration"
	else
		# Create new settings file (pure shell, no dependencies)
		cat >"$settings_file" <<EOF
{
  "model": {
    "name": "$model"
  },
  "privacy": {
    "usageStatisticsEnabled": false
  },
  "modelConfigs": {
    "customAliases": {
      "summarizer-default": {
        "modelConfig": {
          "model": "$model",
          "generateContentConfig": {
            "maxOutputTokens": 2000,
            "temperature": 0.2
          }
        }
      },
      "summarizer-shell": {
        "modelConfig": {
          "model": "$model",
          "generateContentConfig": {
            "maxOutputTokens": 2000,
            "temperature": 0
          }
        }
      },
      "classifier": {
        "modelConfig": {
          "model": "$model"
        }
      },
      "prompt-completion": {
        "modelConfig": {
          "model": "$model"
        }
      },
      "edit-corrector": {
        "modelConfig": {
          "model": "$model"
        }
      },
      "web-search": {
        "modelConfig": {
          "model": "$model"
        }
      },
      "web-fetch": {
        "modelConfig": {
          "model": "$model"
        }
      }
    }
  }
}
EOF
		log_success "Settings file created"
	fi

	log_success "Settings saved to: $settings_file"
}

validate_model_override() {
	local model="$1"

	# Empty values fall back to nordlys-hypernova for backward compatibility
	if [ -z "$model" ]; then
		return 0
	fi

	# Validate format: author/model_id
	if [[ ! "$model" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]; then
		log_error "Model format invalid. Use format: author/model_id (e.g., nordlys-hypernova)"
		return 1
	fi
	return 0
}

configure_gemini() {
	log_info "Configuring Gemini CLI for Nordlys..."
	echo "   You can get your API key from: $API_KEY_URL"

	# Check for environment variable first
	local api_key="${NORDLYS_API_KEY:-}"

	# Check for model overrides
	local model="${NORDLYS_MODEL:-$DEFAULT_MODEL}"

	# Validate model override if provided
	if [ "$model" != "$DEFAULT_MODEL" ]; then
		log_info "Using custom model: $model"
		if ! validate_model_override "$model"; then
			log_error "Invalid model format in NORDLYS_MODEL"
			exit 1
		fi
	fi

	# Use base URL as-is - let Gemini CLI construct the full path
	local base_url="$API_BASE_URL"

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
		echo "   curl -o gemini-cli.sh https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/gemini-cli.sh"
		echo "   chmod +x gemini-cli.sh"
		echo "   ./gemini-cli.sh"
		echo ""
		echo "🔑 Option 2: Set API key via environment variable"
		echo "   export NORDLYS_API_KEY='your-api-key-here'"
		echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/gemini-cli.sh | bash"
		echo "   # The installer will automatically add the API key to your shell config"
		echo ""
		echo "🎯 Option 3: Customize model (Advanced)"
		echo "   export NORDLYS_API_KEY='your-api-key-here'"
		echo "   export NORDLYS_MODEL='nordlys-hypernova'"
		echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/gemini-cli.sh | bash"
		echo ""
		echo "⚙️  Option 4: Manual configuration (Advanced users)"
		echo "   mkdir -p ~/.gemini"
		echo "   # Add to your shell config (~/.bashrc, ~/.zshrc, etc.):"
		echo "   echo 'export GEMINI_API_KEY=\"your-api-key-here\"' >> ~/.bashrc"
		echo "   echo 'export GOOGLE_GEMINI_BASE_URL=\"https://api.nordlyslabs.com\"' >> ~/.bashrc"
		echo "   # Create settings file with full model alias overrides:"
		echo "   cat > ~/.gemini/settings.json << 'EOF'"
		echo '{'
		echo '  "model": {"name": "nordlys-hypernova"},'
		echo '  "privacy": {"usageStatisticsEnabled": false},'
		echo '  "modelConfigs": {'
		echo '    "customAliases": {'
		echo '      "summarizer-default": {"modelConfig": {"model": "nordlys-hypernova"}},'
		echo '      "summarizer-shell": {"modelConfig": {"model": "nordlys-hypernova"}},'
		echo '      "classifier": {"modelConfig": {"model": "nordlys-hypernova"}},'
		echo '      "prompt-completion": {"modelConfig": {"model": "nordlys-hypernova"}},'
		echo '      "edit-corrector": {"modelConfig": {"model": "nordlys-hypernova"}},'
		echo '      "web-search": {"modelConfig": {"model": "nordlys-hypernova"}},'
		echo '      "web-fetch": {"modelConfig": {"model": "nordlys-hypernova"}}'
		echo '    }'
		echo '  }'
		echo '}'
		echo 'EOF'
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

	log_success "Gemini CLI configured for Nordlys successfully"
	log_info "Base URL: $base_url"

	# Add environment variables to shell configuration
	add_env_to_shell_config "$api_key" "$base_url"

	# Create settings.json for model and other preferences
	create_settings_json "$model"
}

# ========================
#        Main Flow
# ========================

show_banner() {
	echo "=========================================="
	echo "  $SCRIPT_NAME v$SCRIPT_VERSION"
	echo "=========================================="
	echo "Configure Gemini CLI to use Nordlys's"
	echo "Mixture of Models for 60-80% cost savings"
	echo ""
}

verify_installation() {
	log_info "Verifying installation..."

	# Check if Gemini CLI can be found
	if ! command -v gemini &>/dev/null; then
		log_error "Gemini CLI installation verification failed"
		return 1
	fi

	# Check if settings file exists
	if [ ! -f "$CONFIG_DIR/settings.json" ]; then
		log_error "Settings file not found at $CONFIG_DIR/settings.json"
		return 1
	fi

	log_success "Installation verification passed"
	return 0
}

main() {
	show_banner

	check_runtime
	install_gemini_cli
	configure_gemini

	if verify_installation; then
		echo ""
		echo "╭──────────────────────────────────────────╮"
		echo "│  🎉 Gemini CLI + Nordlys Setup Complete │"
		echo "╰──────────────────────────────────────────╯"
		echo ""
		echo "🚀 Quick Start:"
		echo "   gemini                    # Start Gemini CLI with Nordlys model"
		echo "   gemini \"help me code\"     # Interactive chat mode"
		echo ""
		echo "🔍 Verify Setup:"
		echo "   gemini --version          # Check Gemini CLI installation"
		echo "   echo \$GEMINI_API_KEY      # Check API key environment variable"
		echo "   echo \$GOOGLE_GEMINI_BASE_URL  # Check base URL configuration"
		echo "   cat ~/.gemini/settings.json    # Check model settings"
		echo ""
		echo "💡 Usage Examples:"
		echo "   gemini \"explain this code\""
		echo "   gemini \"create a React component for user authentication\""
		echo "   gemini \"debug my Python script\""
		echo ""
		echo "📊 Monitor Usage:"
		echo "   Dashboard: $API_KEY_URL"
		echo ""
		echo "💡 Pro Tips:"
		echo "   • API key saved to shell config (env var)"
		echo "   • Model configured in ~/.gemini/settings.json"
		echo "   • Model set to nordlys-hypernova for intelligent routing"
		echo "   • All background operations (summaries, search, etc.) use nordlys-hypernova"
		echo "   • Modify settings.json to customize model and preferences"
		echo ""
		echo "📖 Full Documentation: https://docs.nordlyslabs.com/developer-tools/gemini-cli"
		echo "🐛 Report Issues: https://github.com/Egham-7/nordlys/issues"
	else
		echo ""
		log_error "❌ Installation verification failed"
		echo ""
		echo "🔧 Manual Setup (if needed):"
		echo "   Configuration: Set environment variables and create settings file"
		echo ""
		echo "   # Add to your shell config (~/.bashrc, ~/.zshrc, etc.):"
		echo '   export GEMINI_API_KEY="your-nordlys-api-key"'
		echo '   export GOOGLE_GEMINI_BASE_URL="https://api.nordlyslabs.com"'
		echo ""
		echo "   # Create comprehensive settings file:"
		echo '   mkdir -p ~/.gemini'
		echo '   cat > ~/.gemini/settings.json << '"'"'EOF'"'"
		echo '{'
		echo '  "model": {"name": "nordlys-hypernova"},'
		echo '  "privacy": {"usageStatisticsEnabled": false},'
		echo '  "modelConfigs": {"customAliases": {'
		echo '    "summarizer-default": {"modelConfig": {"model": "nordlys-hypernova"}},'
		echo '    "summarizer-shell": {"modelConfig": {"model": "nordlys-hypernova"}}'
		echo '  }}'
		echo '}'
		echo 'EOF'
		echo ""
		echo "🆘 Get help: https://docs.nordlyslabs.com/troubleshooting"
		exit 1
	fi
}

main "$@"
