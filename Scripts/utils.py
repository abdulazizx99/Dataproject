"""utils.py

Small helper utilities for file-system operations and model persistence.
"""

import os
import joblib


def ensure_dir(directory_path: str) -> None:
    """Create a directory (and parents) if it does not already exist."""
    if directory_path:
        os.makedirs(directory_path, exist_ok=True)


def save_model(model, path: str) -> None:
    """Save a trained model to disk using joblib."""
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")


def load_model(path: str):
    """Load a model that was previously saved with :func:`save_model`."""
    return joblib.load(path)
