"""Fixed-seed reproducibility for Phase 1 dataset synthesis."""
import random, numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
