#!/bin/bash

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="Zed Editor Nordlys Installer"
SCRIPT_VERSION="1.0.0"
CONFIG_FILE="$HOME/.config/zed/settings.json"
API_BASE_URL="https://api.nordlyslabs.com/v1"
API_KEY_URL="https://nordlyslabs.com/api-platform/orgs"

# Model override defaults
DEFAULT_MODEL="nordlys/hypernova"
DEFAULT_PROVIDER="openai_compatible"

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
#  Zed Installation Check
# ========================

check_zed_installed() {
	# Check if Zed config directory exists (more reliable than checking PATH)
	local config_dir="$HOME/.config/zed"
	
	if [ -d "$config_dir" ]; then
		log_success "Zed editor config directory found"
		return 0
	fi
	
	# Also check if zed is in PATH as fallback
	if command -v zed &>/dev/null; then
		log_success "Zed editor is installed"
		return 0
	fi

	log_error "Zed editor is not installed"
	log_info "Please install Zed from: https://zed.dev and run it at least once"
	exit 1
}

# ========================
#  API Key Management
# ========================

get_api_key() {
	local api_key="$NORDLYS_API_KEY"

	if [ -z "$api_key" ]; then
		log_info "API key not found in environment"
		log_info "Get your API key from: $API_KEY_URL"
		read -rp "Enter your Nordlys API key: " api_key
	fi

	if [ -z "$api_key" ]; then
		log_error "API key is required"
		exit 1
	fi

	echo "$api_key"
}

# ========================
#  JSON Configuration
# ========================

update_zed_config() {
	local api_key="$1"
	local config_dir
	config_dir=$(dirname "$CONFIG_FILE")

	ensure_dir_exists "$config_dir"
	create_config_backup "$CONFIG_FILE"

	# Use Python to safely update JSON config
	python3 <<PYTHON_EOF
import json
import sys
from pathlib import Path

config_file = Path("$CONFIG_FILE")
api_key = "$api_key"
api_url = "$API_BASE_URL"
model = "$DEFAULT_MODEL"
provider = "$DEFAULT_PROVIDER"

# Read existing config or create new one
if config_file.exists():
    with open(config_file, 'r') as f:
        content = f.read().strip()
        # Remove comments for JSON parsing
        lines = [line for line in content.split('\\n') if not line.strip().startswith('//')]
        clean_content = '\\n'.join(lines)
        if clean_content:
            try:
                config = json.loads(clean_content)
            except json.JSONDecodeError:
                print("Warning: Existing config has invalid JSON, creating new config", file=sys.stderr)
                config = {}
        else:
            config = {}
else:
    config = {}

# Ensure language_models section exists
if 'language_models' not in config:
    config['language_models'] = {}

# Add Nordlys configuration under openai_compatible
if 'openai_compatible' not in config['language_models']:
    config['language_models']['openai_compatible'] = {}

# Set up Nordlys provider
config['language_models']['openai_compatible']['Nordlys'] = {
    'api_url': api_url,
    'available_models': [
        {
            'name': model,
            'display_name': 'Hypernova (Nordlys MoM)',
            'max_tokens': 200000,
            'capabilities': {
                'tools': True,
                'images': True,
                'parallel_tool_calls': True,
                'prompt_cache_key': False
            }
        }
    ]
}

# Update default agent model if not set or using a different provider
if 'agent' not in config:
    config['agent'] = {}

if 'default_model' not in config['agent']:
    config['agent']['default_model'] = {
        'provider': provider,
        'model': model,
        'name': 'Nordlys'
    }

# Write updated config
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
    f.write('\\n')

print(f"Configuration updated successfully")
PYTHON_EOF

	if [ $? -eq 0 ]; then
		log_success "Zed configuration updated"
	else
		log_error "Failed to update configuration"
		exit 1
	fi
}

# ========================
#  Environment Setup
# ========================

setup_environment() {
	local api_key="$1"
	local shell_rc=""

	# Detect shell configuration file
	if [ -n "${BASH_VERSION:-}" ]; then
		shell_rc="$HOME/.bashrc"
	elif [ -n "${ZSH_VERSION:-}" ]; then
		shell_rc="$HOME/.zshrc"
	else
		shell_rc="$HOME/.profile"
	fi

	# Check if API key already exists in shell config
	if grep -q "NORDLYS_API_KEY" "$shell_rc" 2>/dev/null; then
		log_info "NORDLYS_API_KEY already configured in $shell_rc"
		return
	fi

	# Add API key to shell configuration
	cat >>"$shell_rc" <<-EOF

	# Nordlys API Configuration (added by Zed installer)
	export NORDLYS_API_KEY="$api_key"
	EOF

	log_success "Added NORDLYS_API_KEY to $shell_rc"
	log_info "Run 'source $shell_rc' or restart your terminal to apply changes"
}

# ========================
#       Main Script
# ========================

main() {
	echo ""
	echo "═══════════════════════════════════════════════"
	echo "  $SCRIPT_NAME v$SCRIPT_VERSION"
	echo "═══════════════════════════════════════════════"
	echo ""

	# Check if Zed is installed
	check_zed_installed

	# Get API key
	local api_key
	api_key=$(get_api_key)

	# Update Zed configuration
	update_zed_config "$api_key"

	# Setup environment variables
	setup_environment "$api_key"

	echo ""
	log_success "Installation complete!"
	echo ""
	log_info "Next steps:"
	echo "  1. Restart Zed editor"
	echo "  2. Open Agent Panel (Cmd/Ctrl + Shift + A)"
	echo "  3. Select 'Hypernova (Nordlys MoM)' from model dropdown"
	echo "  4. Start using Nordlys Mixture of Models!"
	echo ""
	log_info "Documentation: https://docs.nordlyslabs.com/developer-tools/zed"
	echo ""
}

main "$@"
