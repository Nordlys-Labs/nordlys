# Nordlys Installation Scripts

This directory contains installation and configuration scripts for integrating various developer tools with Nordlys models.

## Prerequisites

Before running these scripts, ensure you have:

- **Operating System**: Linux or macOS (Windows via WSL2)
- **Shell**: Bash 4.0+ or compatible shell
- **curl**: For downloading scripts and making API requests
- **Node.js** (optional): Some tools may require Node.js 16+ and npm
- **Nordlys API Key**: Required for authentication with Nordlys services

## Supported Platforms

| Platform | Support | Notes |
|----------|---------|-------|
| Linux (Ubuntu, Debian) | ✅ Full | Recommended for production |
| macOS (Intel/Apple Silicon) | ✅ Full | Tested on macOS 12+ |
| Windows WSL2 | ✅ Full | Use Ubuntu 20.04+ in WSL2 |
| Windows Native | ⚠️ Limited | Requires Git Bash or similar |

## Available Scripts

### Developer Tools

| Tool | Script | Description |
|------|--------|-------------|
| Claude Code | `installers/claude-code.sh` | Configure Claude Code to use Nordlys API |
| OpenAI Codex | `installers/codex.sh` | Configure OpenAI Codex to use Nordlys models |
| Grok CLI | `installers/grok-cli.sh` | Configure Grok CLI to use Nordlys models |
| OpenCode | `installers/opencode.sh` | Configure OpenCode to use Nordlys integration |

### Usage

#### Claude Code Setup
```bash
# Download and run
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/claude-code.sh | bash

# Or download first
curl -O https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/claude-code.sh
chmod +x claude-code.sh
./claude-code.sh
```

#### OpenAI Codex Setup
```bash
# Download and run
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/codex.sh | bash

# Or download first
curl -O https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/codex.sh
chmod +x codex.sh
./codex.sh
```

#### Grok CLI Setup
```bash
# Download and run
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/grok-cli.sh | bash

# Or download first
curl -O https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/grok-cli.sh
chmod +x grok-cli.sh
./grok-cli.sh
```

#### OpenCode Setup
```bash
# Download and run
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/opencode.sh | bash

# Or download first
curl -O https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/opencode.sh
chmod +x opencode.sh
./opencode.sh
```

## Script Structure

Each installer script follows this pattern:
- **Prerequisites check**: Verify required dependencies (Node.js, etc.)
- **Tool installation**: Install the developer tool if not present
- **Configuration**: Set up Nordlys API integration
- **Verification**: Test the connection and configuration

## Adding New Tools

To add support for a new developer tool:

1. Create `installers/{tool-name}.sh`
2. Follow the existing script structure
3. Update this README with the new tool
4. Add documentation in `adaptive-docs/developer-tools/{tool-name}.mdx`

## Common Configuration

All scripts configure tools to use:
- **API Base URL**: `https://www.llmadaptive.uk/api/v1`
- **Authentication**: User's Nordlys API key
- **Timeout**: 3000000ms for long-running requests

## Troubleshooting

### Script Execution Errors

**Permission denied**:
```bash
chmod +x installers/script-name.sh
./installers/script-name.sh
```

**curl command not found**:
```bash
# Ubuntu/Debian
sudo apt-get install curl

# macOS
brew install curl
```

**Node.js not found** (for tools requiring Node.js):
```bash
# Ubuntu/Debian
sudo apt-get install nodejs npm

# macOS
brew install node

# Verify installation
node --version  # Should show v16.0.0 or higher
```

### Configuration Issues

**API key not working**:
- Verify your API key at [llmadaptive.uk](https://llmadaptive.uk)
- Ensure key is properly copied (no extra spaces)
- Check if key has necessary permissions

**Connection timeout**:
- Check internet connectivity
- Verify firewall isn't blocking connections to `llmadaptive.uk`
- Try increasing timeout in tool configuration

### Verification Steps

After running a script, verify installation:

```bash
# Check if tool is installed
which <tool-name>

# Test API connection
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://www.llmadaptive.uk/api/v1/health

# Check tool configuration
cat ~/.config/<tool-name>/config.json  # Path varies by tool
```

## Support

For issues with installation scripts:
- Check the tool-specific documentation in `adaptive-docs/developer-tools/`
- Visit [docs.llmadaptive.uk](https://docs.llmadaptive.uk)
- Contact support at [support@llmadaptive.uk](mailto:support@llmadaptive.uk)
