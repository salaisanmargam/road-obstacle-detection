import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across common libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch may not be installed yet.
        pass
