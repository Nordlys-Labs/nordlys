"""
Python ctypes bindings for Cactus C library.

This module provides a Python interface to the Cactus FFI (Foreign Function Interface).
It wraps the C functions defined in cactus/cactus/ffi/cactus_ffi.h

Usage:
    from bindings.cactus_bindings import CactusModel

    # Initialize model
    model = CactusModel("path/to/model.gguf", context_size=2048)

    # Generate embeddings
    embedding = model.embed("Hello world")
    print(f"Embedding shape: {embedding.shape}")  # (768,) or (1024,)

    # Clean up
    model.destroy()
"""

import ctypes
import numpy as np
import platform
import os
from pathlib import Path
from typing import Optional, Tuple


class CactusError(Exception):
    """Exception raised when Cactus operations fail."""
    pass


class CactusNotAvailableError(Exception):
    """Exception raised when Cactus library is not available (e.g., on x86)."""
    pass


class CactusModel:
    """
    Python wrapper for Cactus model operations.

    This class provides a high-level interface to load models and generate embeddings
    using the Cactus C library via ctypes.
    """

    def __init__(self, model_path: str, context_size: int = 2048, lib_path: Optional[str] = None):
        """
        Initialize a Cactus model.

        Args:
            model_path: Path to the GGUF model file
            context_size: Context window size (default: 2048)
            lib_path: Optional path to libcactus shared library
                     If not provided, will search in default locations

        Raises:
            CactusNotAvailableError: If Cactus library cannot be loaded (e.g., on x86)
            CactusError: If model initialization fails
        """
        self.model_path = model_path
        self.context_size = context_size
        self.lib_path = lib_path
        self._model_ptr = None
        self._lib = None

        # Load library
        self._load_library()

        # Initialize model
        self._init_model()

    def _load_library(self):
        """Load the Cactus shared library."""

        # Check architecture
        machine = platform.machine().lower()
        if machine not in ['arm64', 'aarch64']:
            raise CactusNotAvailableError(
                f"Cactus library requires ARM architecture. Current architecture: {machine}\n"
                f"You are running on x86. Please use --mock-embeddings flag for testing, "
                f"or run this on your Mac (ARM) machine."
            )

        # Determine library name based on platform
        system = platform.system()
        if system == "Darwin":  # macOS
            lib_name = "libcactus.dylib"
        elif system == "Linux":
            lib_name = "libcactus.so"
        else:
            raise CactusNotAvailableError(f"Unsupported platform: {system}")

        # Search paths for library
        search_paths = []

        if self.lib_path:
            search_paths.append(self.lib_path)

        # Add default search paths (repo root is three levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent
        cactus_repo = project_root / "cactus"

        # macOS xcframework build output (contains a Mach-O shared library named "cactus")
        xcframework_lib = (
            cactus_repo
            / "apple"
            / "cactus-macos.xcframework"
            / "macos-arm64"
            / "cactus.framework"
            / "Versions"
            / "A"
            / "cactus"
        )

        search_paths.extend([
            cactus_repo / "build" / "cactus" / lib_name,
            cactus_repo / "build" / lib_name,
            xcframework_lib,
            Path.cwd() / lib_name,
            Path(f"./{lib_name}"),
        ])

        # Try to load library
        for path in search_paths:
            if Path(path).exists():
                try:
                    self._lib = ctypes.CDLL(str(path))
                    print(f"✅ Loaded Cactus library from: {path}")
                    break
                except OSError as e:
                    print(f"⚠️  Failed to load {path}: {e}")
                    continue

        if self._lib is None:
            raise CactusNotAvailableError(
                f"Could not find Cactus library '{lib_name}' in any of these locations:\n" +
                "\n".join(f"  - {p}" for p in search_paths) +
                "\n\nPlease build Cactus first:\n"
                "  cd cactus\n"
                "  ./apple/build.sh    # on Mac\n"
                "  ./android/build.sh  # on Linux ARM"
            )

        # Define function signatures
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Define ctypes function signatures for Cactus FFI."""

        # cactus_model_t is void*
        cactus_model_t = ctypes.c_void_p

        # cactus_model_t cactus_init(const char* model_path, size_t context_size, const char* corpus_dir)
        self._lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
        self._lib.cactus_init.restype = cactus_model_t

        # int cactus_embed(cactus_model_t model, const char* text,
        #                  float* embeddings_buffer, size_t buffer_size, size_t* embedding_dim)
        self._lib.cactus_embed.argtypes = [
            cactus_model_t,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self._lib.cactus_embed.restype = ctypes.c_int

        # void cactus_destroy(cactus_model_t model)
        self._lib.cactus_destroy.argtypes = [cactus_model_t]
        self._lib.cactus_destroy.restype = None

    def _init_model(self):
        """Initialize the Cactus model."""

        if not Path(self.model_path).exists():
            raise CactusError(f"Model file not found: {self.model_path}")

        model_path_bytes = self.model_path.encode('utf-8')

        print(f"🔄 Initializing Cactus model: {self.model_path}")
        self._model_ptr = self._lib.cactus_init(
            model_path_bytes,
            self.context_size,
            None  # corpus_dir (for RAG, not needed for embeddings)
        )

        if not self._model_ptr:
            raise CactusError(f"Failed to initialize model: {self.model_path}")

        print(f"✅ Model initialized successfully")

    def embed(self, text: str, max_dim: int = 4096) -> np.ndarray:
        """
        Generate embeddings for the given text.

        Args:
            text: Input text to embed
            max_dim: Maximum embedding dimension (default: 4096)

        Returns:
            numpy array of shape (embedding_dim,) containing the embeddings

        Raises:
            CactusError: If embedding generation fails
        """

        if not self._model_ptr:
            raise CactusError("Model not initialized")

        # Prepare buffers
        text_bytes = text.encode('utf-8')
        embeddings_buffer = (ctypes.c_float * max_dim)()
        embedding_dim = ctypes.c_size_t(0)
        buffer_size = ctypes.sizeof(ctypes.c_float) * max_dim

        # Call cactus_embed
        result = self._lib.cactus_embed(
            self._model_ptr,
            text_bytes,
            embeddings_buffer,
            buffer_size,
            ctypes.byref(embedding_dim)
        )

        if result < 0:
            raise CactusError(f"Failed to generate embeddings. Error code: {result}")

        # If the C side didn't populate embedding_dim for some reason, fall back to result
        dim = embedding_dim.value or result
        if dim == 0 or dim > max_dim:
            raise CactusError(f"Invalid embedding dimension: {dim}")

        embeddings = np.array([embeddings_buffer[i] for i in range(dim)], dtype=np.float32)

        return embeddings

    def embed_batch(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """

        embeddings_list = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Generating embeddings")
            except ImportError:
                print(f"Generating embeddings for {len(texts)} texts...")
                iterator = texts
        else:
            iterator = texts

        for text in iterator:
            emb = self.embed(text)
            embeddings_list.append(emb)

        return np.array(embeddings_list)

    def get_embedding_dim(self, sample_text: str = "test") -> int:
        """
        Get the embedding dimension by generating a sample embedding.

        Args:
            sample_text: Sample text to use for dimension detection

        Returns:
            Embedding dimension
        """
        emb = self.embed(sample_text)
        return len(emb)

    def destroy(self):
        """Free the model and release resources."""

        if self._model_ptr and self._lib:
            self._lib.cactus_destroy(self._model_ptr)
            self._model_ptr = None
            print("✅ Model destroyed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.destroy()

    def __del__(self):
        """Destructor."""
        if self._model_ptr:
            self.destroy()


def is_cactus_available() -> Tuple[bool, str]:
    """
    Check if Cactus library is available on this system.

    Returns:
        Tuple of (is_available, message)
    """

    machine = platform.machine().lower()
    if machine not in ['arm64', 'aarch64']:
        return False, f"ARM architecture required. Current: {machine}"

    # Try to find library
    system = platform.system()
    if system == "Darwin":
        lib_name = "libcactus.dylib"
    elif system == "Linux":
        lib_name = "libcactus.so"
    else:
        return False, f"Unsupported platform: {system}"

    project_root = Path(__file__).parent.parent.parent.parent
    search_paths = [
        project_root / "cactus" / "build" / "cactus" / lib_name,
        project_root / "cactus" / "build" / lib_name,
    ]

    for path in search_paths:
        if path.exists():
            return True, f"Found at {path}"

    return False, f"Library '{lib_name}' not found. Please build Cactus first."


if __name__ == "__main__":
    # Test if Cactus is available
    available, message = is_cactus_available()
    print(f"Cactus available: {available}")
    print(f"Message: {message}")
