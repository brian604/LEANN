# packages/leann-core/src/leann/__init__.py
import os
import platform

# Fix OpenMP threading issues on macOS ARM64
if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["KMP_BLOCKTIME"] = "0"
    # Additional fixes for PyTorch/sentence-transformers on macOS ARM64 only in CI
    if os.environ.get("CI") == "true":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .api import LeannBuilder, LeannChat, LeannSearcher
from .registry import BACKEND_REGISTRY, autodiscover_backends

autodiscover_backends()

# Conditional import for extraction (requires contextgem)
_extraction_available = False
try:
    from .extraction import LeannExtractor

    _extraction_available = True
except ImportError:
    # ContextGem not installed - extraction features unavailable
    LeannExtractor = None  # type: ignore

__all__ = ["BACKEND_REGISTRY", "LeannBuilder", "LeannChat", "LeannSearcher"]

if _extraction_available:
    __all__.append("LeannExtractor")
