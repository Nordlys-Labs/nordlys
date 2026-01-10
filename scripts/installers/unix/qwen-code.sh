#!/bin/bash

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="Qwen Code Nordlys Installer"
SCRIPT_VERSION="1.0.0"
NODE_MIN_VERSION=20
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"
QWEN_PACKAGE="@qwen-code/qwen-code"
CONFIG_DIR="$HOME/.qwen"
API_BASE_URL="https://api.nordlyslabs.com/v1"
API_KEY_URL="https://nordlyslabs.com/api-platform/orgs"

# Model override defaults (can be overridden by environment variables)
# Use nordlys/hypernova to enable Nordlys model for optimal cost/performance
DEFAULT_MODEL="nordlys/hypernova"

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
#     Qwen Code Installation
# ========================

install_qwen_code() {
	if command -v qwen &>/dev/null; then
		log_success "Qwen Code is already installed: $(qwen --version 2>/dev/null || echo 'installed')"
	else
		log_info "Installing Qwen Code..."
		$INSTALL_CMD "$QWEN_PACKAGE" || {
			log_error "Failed to install Qwen Code"
			exit 1
		}
		log_success "Qwen Code installed successfully"
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
	local model="$2"
	local base_url="$3"
	local shell_type
	local config_file

	shell_type=$(detect_shell)
	config_file=$(get_shell_config_file "$shell_type")

	log_info "Adding environment variables to $config_file"

	# Create config file if it doesn't exist
	touch "$config_file"

	# Check if OPENAI_API_KEY already exists in the config (Qwen Code uses OpenAI-compatible API)
	if grep -q "OPENAI_API_KEY.*qwen-code" "$config_file" 2>/dev/null; then
		log_info "Qwen Code environment variables already exist in $config_file, updating..."

		if [ "$shell_type" = "fish" ]; then
			# Fish shell: update API key, base URL, and model
			portable_sed "s|set -x OPENAI_API_KEY.*# qwen-code|set -x OPENAI_API_KEY \"$api_key\"  # qwen-code|" "$config_file"
			portable_sed "s|set -x OPENAI_BASE_URL.*# qwen-code|set -x OPENAI_BASE_URL \"$base_url\"  # qwen-code|" "$config_file"
			portable_sed "s|set -x OPENAI_MODEL.*# qwen-code|set -x OPENAI_MODEL \"$model\"  # qwen-code|" "$config_file"
		else
			# POSIX shells (bash/zsh): update API key, base URL, and model
			portable_sed "s|export OPENAI_API_KEY=.*# qwen-code|export OPENAI_API_KEY=\"$api_key\"  # qwen-code|" "$config_file"
			portable_sed "s|export OPENAI_BASE_URL=.*# qwen-code|export OPENAI_BASE_URL=\"$base_url\"  # qwen-code|" "$config_file"
			portable_sed "s|export OPENAI_MODEL=.*# qwen-code|export OPENAI_MODEL=\"$model\"  # qwen-code|" "$config_file"
		fi
	else
		# Add new environment variables based on shell type
		if [ "$shell_type" = "fish" ]; then
			{
				echo ""
				echo "# Qwen Code with Nordlys Model API Configuration (added by qwen-code installer)"
				echo "set -x OPENAI_API_KEY \"$api_key\"  # qwen-code"
				echo "set -x OPENAI_BASE_URL \"$base_url\"  # qwen-code"
				echo "set -x OPENAI_MODEL \"$model\"  # qwen-code"
			} >>"$config_file"
		else
			{
				echo ""
				echo "# Qwen Code with Nordlys Model API Configuration (added by qwen-code installer)"
				echo "export OPENAI_API_KEY=\"$api_key\"  # qwen-code"
				echo "export OPENAI_BASE_URL=\"$base_url\"  # qwen-code"
				echo "export OPENAI_MODEL=\"$model\"  # qwen-code"
			} >>"$config_file"
		fi
	fi

	log_success "Environment variables added to $config_file"
	if [ "$model" = "$DEFAULT_MODEL" ]; then
		log_info "OPENAI_MODEL set to nordlys/hypernova for Nordlys model (default)"
	else
		log_info "OPENAI_MODEL set to: $model"
	fi
	if [ "$shell_type" = "fish" ]; then
		log_info "Restart your terminal or run 'source $config_file' to apply changes"
	else
		log_info "Run 'source $config_file' or restart your terminal to apply changes"
	fi
}

validate_model_override() {
	local model="$1"

	# Empty values fall back to nordlys/hypernova for backward compatibility
	if [ -z "$model" ]; then
		return 0
	fi

	# Validate format: author/model_id
	if [[ ! "$model" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]; then
		log_error "Model format invalid. Use format: author/model_id (e.g., nordlys/hypernova)"
		return 1
	fi
	return 0
}

configure_qwen() {
	log_info "Configuring Qwen Code for Nordlys..."
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

	# Use base URL as-is - let Qwen Code construct the full path
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
		echo "   curl -o qwen-code.sh https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/qwen-code.sh"
		echo "   chmod +x qwen-code.sh"
		echo "   ./qwen-code.sh"
		echo ""
		echo "🔑 Option 2: Set API key via environment variable"
		echo "   export NORDLYS_API_KEY='your-api-key-here'"
		echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/qwen-code.sh | bash"
		echo "   # The installer will automatically add the API key to your shell config"
		echo ""
		echo "🎯 Option 3: Customize model (Advanced)"
		echo "   export NORDLYS_API_KEY='your-api-key-here'"
		echo "   export NORDLYS_MODEL='nordlys/hypernova'"
		echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/qwen-code.sh | bash"
		echo ""
		echo "⚙️  Option 4: Manual configuration (Advanced users)"
		echo "   mkdir -p ~/.qwen"
		echo "   export NORDLYS_API_KEY='your-api-key-here'"
		echo "   # Add to your shell config (~/.bashrc, ~/.zshrc, etc.):"
		echo "   echo 'export OPENAI_API_KEY=\"your-api-key-here\"  # qwen-code' >> ~/.bashrc"
		echo "   echo 'export OPENAI_BASE_URL=\"https://api.nordlyslabs.com/v1\"  # qwen-code' >> ~/.bashrc"
		echo "   echo 'export OPENAI_MODEL=\"nordlys/hypernova\"  # qwen-code - Nordlys model' >> ~/.bashrc"
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

	log_success "Qwen Code configured for Nordlys successfully"
	log_info "Base URL: $base_url"

	# Add environment variables to shell configuration with the constructed base URL
	add_env_to_shell_config "$api_key" "$model" "$base_url"
}

# ========================
#        Main Flow
# ========================

show_banner() {
	echo "=========================================="
	echo "  $SCRIPT_NAME v$SCRIPT_VERSION"
	echo "=========================================="
	echo "Configure Qwen Code to use Nordlys's"
	echo "Mixture of Models for 60-80% cost savings"
	echo ""
}

verify_installation() {
	log_info "Verifying installation..."

	# Check if Qwen Code can be found
	if ! command -v qwen &>/dev/null; then
		log_error "Qwen Code installation verification failed"
		return 1
	fi

	log_success "Installation verification passed"
	return 0
}

main() {
	show_banner

	check_runtime
	install_qwen_code
	configure_qwen

	if verify_installation; then
		echo ""
		echo "╭──────────────────────────────────────────╮"
		echo "│  🎉 Qwen Code + Nordlys Setup Complete  │"
		echo "╰──────────────────────────────────────────╯"
		echo ""
		echo "🚀 Quick Start:"
		echo "   qwen                      # Start Qwen Code with Nordlys model"
		echo "   qwen \"help me code\"       # Interactive chat mode"
		echo ""
		echo "🔍 Verify Setup:"
		echo "   qwen --version            # Check Qwen Code installation"
		echo "   echo \$OPENAI_API_KEY      # Check API key environment variable"
		echo "   echo \$OPENAI_BASE_URL     # Check base URL configuration"
		echo ""
		echo "💡 Usage Examples:"
		echo "   qwen \"explain this code\""
		echo "   qwen \"create a React component for user authentication\""
		echo "   qwen \"debug my Python script\""
		echo ""
		echo "📊 Monitor Usage:"
		echo "   Dashboard: $API_KEY_URL"
		echo ""
		echo "💡 Pro Tips:"
		echo "   • Your API key is automatically saved to your shell config"
		echo "   • OPENAI_MODEL set to nordlys/hypernova for Nordlys model (recommended)"
		echo "   • Set OPENAI_MODEL to another model ID if your team uses additional Nordlys models"
		echo ""
		echo "📖 Full Documentation: https://docs.nordlyslabs.com/developer-tools/qwen-code"
		echo "🐛 Report Issues: https://github.com/Egham-7/nordlys/issues"
	else
		echo ""
		log_error "❌ Installation verification failed"
		echo ""
		echo "🔧 Manual Setup (if needed):"
		echo "   Configuration: Set environment variables in your shell config"
		echo "   Expected variables:"
		echo '   export OPENAI_API_KEY="your-nordlys-api-key"  # qwen-code'
		echo '   export OPENAI_BASE_URL="https://api.nordlyslabs.com/v1"  # qwen-code'
		echo '   export OPENAI_MODEL="nordlys/hypernova"  # qwen-code - Nordlys model'
		echo ""
		echo "🆘 Get help: https://docs.nordlyslabs.com/troubleshooting"
		exit 1
	fi
}

main "$@"
