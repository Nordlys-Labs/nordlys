# Nordlys Model Engine - Python Bindings

Python bindings for the Nordlys C++ core. Internal use only.

## Package Structure

The Python bindings use a standard `src/` layout:
- `src/nordlys_core/` - Python package directory
- `src/nordlys_core/__init__.py` - Package initialization
- `src/nordlys_core/_core.pyi` - Type stubs for IDE support
- `src/nordlys_core/module.cpp` - Main module definition
- `src/nordlys_core/helpers.cpp` - Helper functions
- `src/nordlys_core/types.cpp` - Configuration type bindings
- `src/nordlys_core/results.cpp` - Result type bindings
- `src/nordlys_core/checkpoint.cpp` - NordlysCheckpoint bindings
- `src/nordlys_core/nordlys.cpp` - Nordlys32/64 bindings

The package can be imported as:
```python
import nordlys_core
from nordlys_core import Nordlys32, Nordlys64, NordlysCheckpoint
```

## Build

```bash
pip install -e .
```

## Links

- Docs: https://docs.nordlyslabs.com
- Issues: https://github.com/Egham-7/nordlys/issues
