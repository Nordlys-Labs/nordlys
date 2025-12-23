# Nordlys Installation Scripts

This directory contains installation and configuration scripts for integrating various developer tools with Nordlys models.

## Prerequisites

Before running these scripts, ensure you have:

- **Operating System**: Linux, macOS, or Windows
- **Shell**: Bash for Unix scripts, PowerShell or cmd for Windows scripts
- **curl**: For downloading scripts and making API requests
- **Node.js** (optional): Some tools require Node.js and npm
- **Nordlys API Key**: Required for authentication with Nordlys services

## Supported Platforms

| Platform | Support | Notes |
|----------|---------|-------|
| Linux (Ubuntu, Debian) | ✅ Full | Use Unix scripts |
| macOS (Intel/Apple Silicon) | ✅ Full | Use Unix scripts |
| Windows (PowerShell) | ✅ Full | Use PowerShell scripts |
| Windows (cmd) | ✅ Full | Use cmd scripts |

## Available Scripts

| Tool | Unix | Windows PowerShell | Windows cmd |
|------|------|-------------------|-------------|
| Claude Code | `installers/unix/claude-code.sh` | `installers/windows/claude-code.ps1` | `installers/windows/claude-code.cmd` |
| OpenAI Codex | `installers/unix/codex.sh` | `installers/windows/codex.ps1` | `installers/windows/codex.cmd` |
| Grok CLI | `installers/unix/grok-cli.sh` | `installers/windows/grok-cli.ps1` | `installers/windows/grok-cli.cmd` |
| Gemini CLI | `installers/unix/gemini-cli.sh` | `installers/windows/gemini-cli.ps1` | `installers/windows/gemini-cli.cmd` |
| Qwen Code | `installers/unix/qwen-code.sh` | `installers/windows/qwen-code.ps1` | `installers/windows/qwen-code.cmd` |
| OpenCode | `installers/unix/opencode.sh` | `installers/windows/opencode.ps1` | `installers/windows/opencode.cmd` |

## Usage

### Claude Code Setup
```bash
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/unix/claude-code.sh | bash
```

```powershell
iwr -useb https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/claude-code.ps1 | iex
```

```bat
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/claude-code.cmd -o claude-code.cmd
claude-code.cmd
```

### OpenAI Codex Setup
```bash
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/unix/codex.sh | bash
```

```powershell
iwr -useb https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/codex.ps1 | iex
```

```bat
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/codex.cmd -o codex.cmd
codex.cmd
```

### Grok CLI Setup
```bash
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/unix/grok-cli.sh | bash
```

```powershell
iwr -useb https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/grok-cli.ps1 | iex
```

```bat
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/grok-cli.cmd -o grok-cli.cmd
grok-cli.cmd
```

### Gemini CLI Setup
```bash
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/unix/gemini-cli.sh | bash
```

```powershell
iwr -useb https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/gemini-cli.ps1 | iex
```

```bat
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/gemini-cli.cmd -o gemini-cli.cmd
gemini-cli.cmd
```

### Qwen Code Setup
```bash
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/unix/qwen-code.sh | bash
```

```powershell
iwr -useb https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/qwen-code.ps1 | iex
```

```bat
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/qwen-code.cmd -o qwen-code.cmd
qwen-code.cmd
```

### OpenCode Setup
```bash
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/unix/opencode.sh | bash
```

```powershell
iwr -useb https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/opencode.ps1 | iex
```

```bat
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/windows/opencode.cmd -o opencode.cmd
opencode.cmd
```

## Script Structure

Each installer script follows this pattern:
- **Prerequisites check**: Verify required dependencies (Node.js, etc.)
- **Tool installation**: Install the developer tool if not present
- **Configuration**: Set up Nordlys API integration
- **Verification**: Test the connection and configuration

## Adding New Tools

To add support for a new developer tool:

1. Create `installers/unix/{tool-name}.sh`
2. Create `installers/windows/{tool-name}.ps1` and `installers/windows/{tool-name}.cmd`
3. Update this README with the new tool
4. Add documentation in `nordlys-docs/developer-tools/{tool-name}.mdx`

## Common Configuration

All scripts configure tools to use:
- **API Base URL**: `https://api.llmadaptive.uk`
- **Authentication**: User's Nordlys API key
- **Timeout**: 3000000ms for long-running requests (where applicable)

## Troubleshooting

### Script Execution Errors

**Permission denied (Unix)**:
```bash
chmod +x installers/unix/script-name.sh
./installers/unix/script-name.sh
```

**PowerShell execution policy (Windows)**:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
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
node --version
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
  https://api.llmadaptive.uk/v1/health

# Check tool configuration
cat ~/.config/<tool-name>/config.json
```

## Support

For issues with installation scripts:
- Check the tool-specific documentation in `nordlys-docs/developer-tools/`
- Visit [docs.llmadaptive.uk](https://docs.llmadaptive.uk)
- Contact support at [support@llmadaptive.uk](mailto:support@llmadaptive.uk)
