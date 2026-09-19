#!/usr/bin/env python3
"""Seed utilities for reproducibility (Plan Section 1).
Sets random seeds for Python, NumPy, and PyTorch."""
import os
import random

import numpy as np

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed=42):
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_seed():
    """Get current random seed (from environment or time)."""
    return int(os.environ.get('PYTHONHASHSEED', random.randint(0, 2**32 - 1)))