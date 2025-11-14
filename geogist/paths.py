"""
Path utilities for the geogist package.

This module provides utility functions for getting package-relative paths
that work correctly regardless of the current working directory.
"""

import tempfile
from pathlib import Path
from typing import Union

def get_package_root() -> Path:
    """Get the root directory of the geogist package."""
    return Path(__file__).parent.resolve()

def get_project_root() -> Path:
    """Get the root directory of the project (parent of geogist package)."""
    return get_package_root().parent

def get_weights_path() -> Path:
    """Get the path to the weights directory."""
    return get_project_root() / "weights"

def get_cache_path(cache_dir: Union[str, Path] = None) -> Path:
    """
    Get the cache directory path.
    
    Args:
        cache_dir: Optional cache directory. If relative path or None,
                  will default to a directory inside the system temp folder.
    
    Returns:
        Absolute path to cache directory
    """
    if cache_dir is None:
        cache_path = Path(tempfile.gettempdir()) / "geogist-cache"
    else:
        cache_path = Path(cache_dir)
        # If it's a relative path, make it relative to project root
        if not cache_path.is_absolute():
            cache_path = get_project_root() / cache_path

    return cache_path.resolve()

def get_config_path() -> Path:
    """Get the path to the config directory."""
    return get_project_root() / "config"

def ensure_path_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a path exists, creating directories if necessary.
    
    Args:
        path: Path to ensure exists
        
    Returns:
        Path object for the ensured path
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
