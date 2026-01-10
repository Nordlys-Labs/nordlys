#!/bin/bash

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="Grok CLI Nordlys Installer"
SCRIPT_VERSION="1.0.0"
NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"
GROK_PACKAGE="@vibe-kit/grok-cli@0.0.16"
CONFIG_DIR="$HOME/.grok"
API_BASE_URL="https://api.nordlyslabs.com/v1"
API_KEY_URL="https://nordlyslabs.com/api-platform/orgs"

# Model override defaults (can be overridden by environment variables)
# Use nordlys/hypernova to enable Nordlys model for optimal cost/performance
DEFAULT_MODEL="nordlys/hypernova"
DEFAULT_MODELS='["nordlys/hypernova"]'

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

install_bun() {
	local platform
	platform=$(uname -s)

	case "$platform" in
	Linux | Darwin)
		log_info "Installing Bun on $platform..."
		curl -fsSL https://bun.sh/install | bash

		# Load Bun environment
		export BUN_INSTALL="$HOME/.bun"
		export PATH="$BUN_INSTALL/bin:$PATH"

		# Verify installation
		if command -v bun &>/dev/null; then
			log_success "Bun installed: $(bun --version)"
		else
			log_error "Bun installation failed"
			exit 1
		fi
		;;
	*)
		log_error "Unsupported platform: $platform"
		exit 1
		;;
	esac
}

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

check_bun() {
	if command -v bun &>/dev/null; then
		current_version=$(bun --version)
		log_success "Bun is already installed: v$current_version"
		return 0
	else
		log_info "Bun not found. Installing..."
		install_bun
	fi
}

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
	# Prefer Bun over Node.js for better performance
	if command -v bun &>/dev/null; then
		log_info "Using Bun runtime (recommended)"
		INSTALL_CMD="bun add -g"
		return 0
	elif command -v node &>/dev/null && command -v npm &>/dev/null; then
		current_version=$(node -v | sed 's/v//')
		major_version=$(echo "$current_version" | cut -d. -f1)

		if [ "$major_version" -ge "$NODE_MIN_VERSION" ]; then
			log_info "Using Node.js runtime (fallback)"
			INSTALL_CMD="npm install -g"
			return 0
		fi
	fi

	# Install preferred runtime
	log_info "No suitable runtime found. Installing Bun (recommended)..."
	check_bun
	INSTALL_CMD="bun add -g"
}

# ========================
#     Grok CLI Installation
# ========================

install_grok_cli() {
	if command -v grok &>/dev/null; then
		log_success "Grok CLI is already installed: $(grok --version 2>/dev/null || echo 'installed')"
	else
		log_info "Installing Grok CLI..."
		$INSTALL_CMD "$GROK_PACKAGE" || {
			log_error "Failed to install grok-cli"
			exit 1
		}
		log_success "Grok CLI installed successfully"
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
	if [ -n "${ZSH_VERSION:-}" ]; then
		echo "zsh"
	elif [ -n "${BASH_VERSION:-}" ]; then
		echo "bash"
	elif [ -n "${FISH_VERSION:-}" ]; then
		echo "fish"
	else
		# Fallback to checking SHELL environment variable
		case "$SHELL" in
		*/zsh) echo "zsh" ;;
		*/bash) echo "bash" ;;
		*/fish) echo "fish" ;;
		*) echo "bash" ;; # Default fallback
		esac
	fi
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
	local shell_type
	local config_file

	shell_type=$(detect_shell)
	config_file=$(get_shell_config_file "$shell_type")

	log_info "Adding environment variables to $config_file"

	# Create config file if it doesn't exist
	touch "$config_file"

	# Check if NORDLYS_API_KEY already exists in the config
	if grep -q "NORDLYS_API_KEY" "$config_file" 2>/dev/null; then
		log_info "NORDLYS environment variables already exist in $config_file, updating..."

		if [ "$shell_type" = "fish" ]; then
			# Fish shell: update both API key and base URL
			portable_sed "s|set -x NORDLYS_API_KEY.*|set -x NORDLYS_API_KEY \"$api_key\"|" "$config_file"
			portable_sed "s|set -x NORDLYS_BASE_URL.*|set -x NORDLYS_BASE_URL \"$API_BASE_URL\"|" "$config_file"

			# Add NORDLYS_BASE_URL if it doesn't exist in Fish config
			if ! grep -q "NORDLYS_BASE_URL" "$config_file" 2>/dev/null; then
				echo "set -x NORDLYS_BASE_URL \"$API_BASE_URL\"" >>"$config_file"
			fi
		else
			# POSIX shells (bash/zsh): update both API key and base URL
			portable_sed "s|export NORDLYS_API_KEY=.*|export NORDLYS_API_KEY=\"$api_key\"|" "$config_file"
			portable_sed "s|export NORDLYS_BASE_URL=.*|export NORDLYS_BASE_URL=\"$API_BASE_URL\"|" "$config_file"

			# Add NORDLYS_BASE_URL if it doesn't exist in POSIX shell config
			if ! grep -q "NORDLYS_BASE_URL" "$config_file" 2>/dev/null; then
				echo "export NORDLYS_BASE_URL=\"$API_BASE_URL\"" >>"$config_file"
			fi
		fi
	else
		# Add new environment variables based on shell type
		echo "" >>"$config_file"
		echo "# Nordlys Model API Configuration (added by grok-cli installer)" >>"$config_file"
		if [ "$shell_type" = "fish" ]; then
			echo "set -x NORDLYS_API_KEY \"$api_key\"" >>"$config_file"
			echo "set -x NORDLYS_BASE_URL \"$API_BASE_URL\"" >>"$config_file"
			echo "set -x GROK_MODEL \"$DEFAULT_MODEL\"" >>"$config_file"
			echo "set -x GROK_BASE_URL \"$API_BASE_URL\"" >>"$config_file"
		else
			echo "export NORDLYS_API_KEY=\"$api_key\"" >>"$config_file"
			echo "export NORDLYS_BASE_URL=\"$API_BASE_URL\"" >>"$config_file"
			echo "export GROK_MODEL=\"$DEFAULT_MODEL\"" >>"$config_file"
			echo "export GROK_BASE_URL=\"$API_BASE_URL\"" >>"$config_file"
		fi
	fi

	log_success "Environment variables added to $config_file"
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

configure_grok() {
	log_info "Configuring Grok CLI for Nordlys..."
	echo "   You can get your API key from: $API_KEY_URL"

	# Check for environment variable first
	local api_key="${NORDLYS_API_KEY:-}"

	# Check for model overrides
	local model="${NORDLYS_MODEL:-$DEFAULT_MODEL}"
	local models="${NORDLYS_MODELS:-$DEFAULT_MODELS}"

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
		echo "   curl -o grok-cli.sh https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/grok-cli.sh"
		echo "   chmod +x grok-cli.sh"
		echo "   ./grok-cli.sh"
		echo ""
		echo "🔑 Option 2: Set API key via environment variable"
		echo "   export NORDLYS_API_KEY='your-api-key-here'"
		echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/grok-cli.sh | bash"
		echo "   # The installer will automatically add the API key to your shell config"
		echo ""
		echo "🎯 Option 3: Customize model (Advanced)"
		echo "   export NORDLYS_API_KEY='your-api-key-here'"
		echo "   export NORDLYS_MODEL='nordlys/hypernova'"
		echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/nordlys/main/scripts/installers/unix/grok-cli.sh | bash"
		echo ""
		echo "⚙️  Option 4: Manual configuration (Advanced users)"
		echo "   mkdir -p ~/.grok"
		echo "   export NORDLYS_API_KEY='your-api-key-here'"
		echo "   # Add to your shell config (~/.bashrc, ~/.zshrc, etc.):"
		echo "   echo 'export NORDLYS_API_KEY=\"your-api-key-here\"' >> ~/.bashrc"
		echo "   cat > ~/.grok/user-settings.json << 'EOF'"
		echo "{"
		echo '  "apiKey": "your_api_key_here",'
		echo '  "baseURL": "https://api.nordlyslabs.com/v1",'
		echo '  "defaultModel": "nordlys/hypernova",'
		echo '  "models": ["nordlys/hypernova"]'
		echo "}"
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

	# Create user-settings.json
	local settings_file="$CONFIG_DIR/user-settings.json"
	create_config_backup "$settings_file"
	cat >"$settings_file" <<EOF
{
  "apiKey": "$api_key",
  "baseURL": "$API_BASE_URL",
  "defaultModel": "$model",
  "models": $models
}
EOF

	# Verify the JSON is valid
	if command -v node &>/dev/null; then
		node -e "JSON.parse(require('fs').readFileSync('$settings_file', 'utf8'))" || {
			log_error "Failed to create valid settings.json"
			exit 1
		}
	elif command -v python3 &>/dev/null; then
		python3 -c "import json; json.load(open('$settings_file'))" || {
			log_error "Failed to create valid settings.json"
			exit 1
		}
	fi

	log_success "Grok CLI configured for Nordlys successfully"
	log_info "Configuration saved to: $settings_file"

	# Add environment variables to shell configuration
	add_env_to_shell_config "$api_key"
}

# ========================
#        Main Flow
# ========================

show_banner() {
	echo "=========================================="
	echo "  $SCRIPT_NAME v$SCRIPT_VERSION"
	echo "=========================================="
	echo "Configure Grok CLI to use Nordlys's"
	echo "Mixture of Models for 60-80% cost savings"
	echo ""
}

verify_installation() {
	log_info "Verifying installation..."

	# Check if Grok CLI can be found
	if ! command -v grok &>/dev/null; then
		log_error "Grok CLI installation verification failed"
		return 1
	fi

	# Check if configuration file exists
	if [ ! -f "$CONFIG_DIR/user-settings.json" ]; then
		log_error "Configuration file not found"
		return 1
	fi

	log_success "Installation verification passed"
	return 0
}

main() {
	show_banner

	check_runtime
	install_grok_cli
	configure_grok

	if verify_installation; then
		echo ""
		echo "╭──────────────────────────────────────────╮"
		echo "│  🎉 Grok CLI + Nordlys Setup Complete   │"
		echo "╰──────────────────────────────────────────╯"
		echo ""
		echo "🚀 Quick Start:"
		echo "   grok                      # Start Grok CLI with Nordlys model"
		echo "   grok -p \"help me code\"     # Headless mode for quick tasks"
		echo ""
		echo "🔍 Verify Setup:"
		echo "   grok --version            # Check Grok CLI installation"
		echo "   cat ~/.grok/user-settings.json  # View configuration"
		echo ""
		echo "💡 Usage Examples:"
		echo "   grok -p \"show me the package.json file\""
		echo "   grok -p \"create a React component for user authentication\""
		echo "   grok -d /path/to/project  # Set working directory"
		echo "   grok --model nordlys/hypernova  # Override model"
		echo ""
		echo "📊 Monitor Usage:"
		echo "   Dashboard: $API_KEY_URL"
		echo "   Configuration: ~/.grok/user-settings.json"
		echo ""
		echo "💡 Pro Tips:"
		echo "   • Your API key is automatically saved to your shell config"
		echo "   • Nordlys model enabled by default"
		echo "   • Use --max-tool-rounds to control execution complexity"
		echo "   • Create .grok/GROK.md for custom project instructions"
		echo "   • Add MCP servers with: grok mcp add server-name"
		echo ""
		echo "📖 Full Documentation: https://docs.nordlyslabs.com/developer-tools/grok-cli"
		echo "🐛 Report Issues: https://github.com/Egham-7/nordlys/issues"
	else
		echo ""
		log_error "❌ Installation verification failed"
		echo ""
		echo "🔧 Manual Setup (if needed):"
		echo "   Configuration: ~/.grok/user-settings.json"
		echo "   Expected format:"
		echo '   {"apiKey":"your_key","baseURL":"https://api.nordlyslabs.com/v1","defaultModel":"nordlys/hypernova","models":["nordlys/hypernova"]}'
		echo ""
		echo "🆘 Get help: https://docs.nordlyslabs.com/troubleshooting"
		exit 1
	fi
}

main "$@"
