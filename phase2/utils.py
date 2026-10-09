# phase2/utils.py
"""
Shared utilities for Phase 2 pipeline.
"""

import os
import sys
import json
import csv
import time
import random
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

# ─── Constants ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data" / "phase2"
RDD_ROOT     = Path("D:/road obstacle detection/rdd2022/RDD_SPLIT")
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase2"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "phase2"
EXP_ROOT     = PROJECT_ROOT / "experiments" / "phase2_adaptive_enhancement"

PHASE1_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "phase1_final_baseline"
    / "final"
    / "weights"
    / "best.pt"
)

DATASET_YAML = PROJECT_ROOT / "data" / "dataset_final.yaml"

CLASS_NAMES  = ["D00", "D10", "D20", "D40", "class_4"]
NUM_CLASSES  = 5

SEED = 42

CONDITIONS = ["low_light", "low_contrast", "blur", "noise", "haze", "rain_like"]
SEVERITIES = ["mild", "moderate", "severe"]

# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str = "phase2") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = get_logger()

def logmsg(msg: str) -> None:
    log.info(msg)

# ─── Seeding ─────────────────────────────────────────────────────────────────

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ─── File I/O ────────────────────────────────────────────────────────────────

def save_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def append_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ─── Image I/O ───────────────────────────────────────────────────────────────

def read_image(path: str) -> np.ndarray:
    """Read BGR image. Raises on failure."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def write_image(path: str, img: np.ndarray) -> None:
    """Write BGR image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

# ─── Test image list ─────────────────────────────────────────────────────────

def get_test_image_paths(limit: Optional[int] = None) -> List[Path]:
    """Return sorted list of test image paths from RDD2022."""
    img_dir = RDD_ROOT / "test" / "images"
    paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.png"))
    if limit:
        paths = paths[:limit]
    return paths

def get_label_path(img_path: Path, split: str = "test") -> Path:
    """Given an image path, return the corresponding label file path."""
    stem = img_path.stem
    return RDD_ROOT / split / "labels" / f"{stem}.txt"

# ─── Metrics helpers ─────────────────────────────────────────────────────────

def compute_f1(precision: float, recall: float) -> float:
    if precision + recall < 1e-9:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def robustness_drop(clean_map: float, degraded_map: float) -> float:
    return round(clean_map - degraded_map, 4)

def robustness_retention(clean_map: float, degraded_map: float) -> float:
    if clean_map < 1e-9:
        return 0.0
    return round(degraded_map / clean_map * 100, 2)

# ─── Timer ───────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.perf_counter()
        return self

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self._elapsed = self.elapsed()

    @property
    def seconds(self) -> float:
        return self._elapsed

# ─── Degradation path helper ─────────────────────────────────────────────────

def degraded_image_path(
    orig_path: Path, condition: str, severity: str
) -> Path:
    """
    Given original image path, return the path where the degraded version lives.
    """
    return DATA_ROOT / condition / severity / orig_path.name

def clean_copy_path(orig_path: Path) -> Path:
    """Path for clean copy of test image."""
    return DATA_ROOT / "clean" / "original" / orig_path.name
