"""
Runtime compatibility setup for audio dependencies.
"""

import os
import tempfile
from pathlib import Path


def configure_numba_cache() -> None:
    """Ensure numba-backed libraries can write cache files in restricted environments."""
    cache_dir = Path(tempfile.gettempdir()) / "aiaa_numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
