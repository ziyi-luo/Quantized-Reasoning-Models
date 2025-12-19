"""
Logging utilities for HH quantization.
"""
import random
import numpy as np
import torch

def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_section(logger, title):
    """Log a section separator."""
    logger.info("=" * 60)
    logger.info(f"STARTING: {title}")
    logger.info("=" * 60)
