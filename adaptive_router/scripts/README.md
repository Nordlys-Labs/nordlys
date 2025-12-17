# Adaptive Router Profile Conversion Script

Convert router profiles between JSON and MessagePack formats for optimal storage and performance.

## Overview

The `convert_profile.py` script provides a unified interface for converting router profiles between human-readable JSON format and compact MessagePack binary format.

## Quick Start

```bash
# Convert JSON to MessagePack (recommended for production)
uv run python -m adaptive_router.scripts.convert_profile profile.json

# Convert MessagePack back to JSON (for inspection/debugging)
uv run python -m adaptive_router.scripts.convert_profile profile.msgpack
```

## Usage

### Command Line

```bash
uv run python -m adaptive_router.scripts.convert_profile [OPTIONS] INPUT [OUTPUT]
```

### Arguments

- `INPUT`: Input profile file (`.json` or `.msgpack`)
- `OUTPUT`: Output file path (optional - auto-generated if omitted)

### Options

- `--compact`: Generate compact JSON without indentation
- `-f, --force`: Overwrite existing output files
- `-q, --quiet`: Suppress conversion summary
- `-h, --help`: Show help message

### Examples

```bash
# Basic conversion (JSON → MessagePack)
uv run python -m adaptive_router.scripts.convert_profile profile.json

# Explicit output path
uv run python -m adaptive_router.scripts.convert_profile profile.json ./output.msgpack

# MessagePack → JSON with pretty printing
uv run python -m adaptive_router.scripts.convert_profile profile.msgpack readable.json

# Compact JSON output
uv run python -m adaptive_router.scripts.convert_profile profile.msgpack compact.json --compact

# Force overwrite existing file
uv run python -m adaptive_router.scripts.convert_profile profile.json existing.msgpack --force

# Quiet mode (no output)
uv run python -m adaptive_router.scripts.convert_profile profile.json --quiet
```

## Features

### ✨ Auto-Detection
Automatically determines input/output formats based on file extensions:
- `.json` → JSON format
- `.msgpack` → MessagePack format

### 🔍 Validation
Validates profiles after loading to ensure data integrity.

### 📊 Rich Summary
Shows detailed conversion statistics:
- Number of clusters and models
- Data type (float32/float64)
- Embedding model used
- File sizes and compression ratio

### 🎯 Smart Output
- JSON output includes proper indentation by default
- Auto-generates output filenames when not specified
- Prevents accidental overwrites without `--force`

### ⚡ Performance
Leverages C++ core for fast, efficient conversions.

## Requirements

- **Environment**: Must be run with `uv run` in the adaptive_router workspace
- **Dependencies**: Requires `adaptive_core_ext` C++ extension (automatically available in proper environment)
- **Python**: Compatible with Python 3.11+

## Format Comparison

| Aspect | JSON | MessagePack |
|--------|------|-------------|
| **Readability** | Human-readable | Binary format |
| **Size** | Larger files | 50-70% smaller |
| **Load Speed** | Slower | Faster loading |
| **Use Case** | Development/debugging | Production deployment |

## Output Example

```
Converted: profile.json -> profile.msgpack
  Clusters: 10
  Models: 5
  Dtype: float32
  Embedding model: all-MiniLM-L6-v2
  Input size: 2,145,678 bytes
  Output size: 1,234,567 bytes (57.6%)
```

## Troubleshooting

### Common Issues

**"adaptive_core_ext not available"**
- Ensure you're using `uv run` in the adaptive_router workspace
- The script requires the C++ extension which is only available in the proper environment

**"Input file not found"**
- Check the file path and ensure the file exists
- Use absolute paths if working from different directories

**"Output exists"**
- Use `--force` to overwrite existing files
- Or specify a different output path

**"Input and output have same format"**
- Ensure input and output files have different extensions (.json vs .msgpack)
- Or specify an explicit output path with the desired extension

### Getting Help

```bash
uv run python -m adaptive_router.scripts.convert_profile --help
```