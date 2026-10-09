# phase2/degradation_pipeline.py
"""
Controlled image degradation pipeline for Phase 2.

Creates synthetic visual degradation to simulate difficult road-imaging conditions.
Implements 7 degradation types at 3 severity levels each.

IMPORTANT LIMITATIONS (documented for scientific integrity):
- These are SYNTHETIC degradations, NOT real weather conditions.
- They are designed to be representative, NOT identical to real-world scenarios.
- Severity levels were chosen conservatively to avoid artificially low baselines.
- Bounding-box annotations are PRESERVED unchanged (pixel-level ops only).
- Results from these synthetic conditions cannot directly predict real-world
  performance without additional real-world validation.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .utils import (
    PROJECT_ROOT, DATA_ROOT, RDD_ROOT,
    get_test_image_paths, read_image, write_image,
    degraded_image_path, clean_copy_path,
    logmsg, SEED, CONDITIONS, SEVERITIES,
)

# ─── Degradation parameter table ─────────────────────────────────────────────
# Parameters were chosen conservatively to simulate realistic conditions.
# Reference: standard degradation benchmarks in robustness literature.

DEGRADATION_PARAMS = {
    "low_light": {
        # Gamma > 1 darkens; we apply gamma < 1 to compress intensity downward.
        # gamma=0.6 → mild darkening; gamma=0.15 → near-night conditions
        "mild":     {"gamma": 0.55},
        "moderate": {"gamma": 0.35},
        "severe":   {"gamma": 0.15},
    },
    "low_contrast": {
        # alpha < 1 compresses contrast around mean gray level
        # alpha=0.6 → mild fade; alpha=0.2 → very washed-out
        "mild":     {"alpha": 0.60, "beta": 60},
        "moderate": {"alpha": 0.40, "beta": 80},
        "severe":   {"alpha": 0.22, "beta": 100},
    },
    "noise": {
        # Gaussian noise standard deviation (0–255 scale)
        "mild":     {"sigma": 10},
        "moderate": {"sigma": 25},
        "severe":   {"sigma": 50},
    },
    "blur": {
        # Gaussian blur kernel size (must be odd)
        "mild":     {"ksize": 5},
        "moderate": {"ksize": 11},
        "severe":   {"ksize": 21},
    },
    "haze": {
        # Atmospheric scattering model: I = I*t + A*(1-t)
        # transmission t = exp(-beta*depth), simplified to uniform depth
        # density is effectively (1 - transmission)
        "mild":     {"density": 0.30, "airlight": 200},
        "moderate": {"density": 0.52, "airlight": 210},
        "severe":   {"density": 0.72, "airlight": 220},
    },
    "rain_like": {
        # Simulates rain streaks as semi-transparent bright lines
        # n_lines: number of streaks; length, angle, thickness control appearance
        "mild":     {"n_lines": 300,  "length": 30, "angle": 80, "alpha": 0.35},
        "moderate": {"n_lines": 600,  "length": 45, "angle": 75, "alpha": 0.45},
        "severe":   {"n_lines": 1000, "length": 60, "angle": 70, "alpha": 0.55},
    },
}

# ─── Individual degradation functions ────────────────────────────────────────

def degrade_low_light(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply power-law (gamma) darkening.
    gamma < 1 darkens the image (simulates low-light capture).
    Uses lookup table for speed.
    """
    inv_gamma = gamma  # we apply directly (gamma < 1 = darken)
    lut = np.array([
        (i / 255.0) ** (1.0 / inv_gamma) * 255
        if inv_gamma != 0 else 0
        for i in range(256)
    ], dtype=np.uint8)
    # Actually: for darkening, apply gamma > 1 in the transform
    # I_out = I_in^gamma where gamma < 1 brightens, gamma > 1 darkens
    # We WANT darkening, so apply gamma as exponent directly:
    lut = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
    # Correction: standard gamma correction where gamma_param > 1 → dark output
    # We want the image to appear darker, so use gamma_out = I^(1/gamma) with gamma < 1
    # i.e., gamma=0.5 → I^2 → darkens. Let's be explicit:
    lut = np.array([int(((i / 255.0) ** (1.0 / gamma)) * 255 + 0.5) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def degrade_low_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Reduce contrast: out = alpha * in + beta.
    alpha < 1 compresses dynamic range; beta shifts to mid-gray.
    Simulates washed-out or overcast conditions.
    """
    result = img.astype(np.float32) * alpha + beta
    return np.clip(result, 0, 255).astype(np.uint8)


def degrade_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add zero-mean Gaussian noise with standard deviation sigma.
    Simulates sensor noise in low-light or poor-quality cameras.
    """
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def degrade_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Apply Gaussian blur with kernel size ksize.
    Simulates camera defocus or motion at slow shutter speeds.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def degrade_haze(
    img: np.ndarray, density: float, airlight: int
) -> np.ndarray:
    """
    Atmospheric scattering model (simplified):
        I_hazy = I * t + A * (1 - t)
    where t = 1 - density (uniform transmission map).
    Simulates fog/haze reducing visibility and contrast.

    Limitation: Real haze has depth-dependent transmission; this is uniform.
    """
    t = 1.0 - density
    A = airlight
    result = img.astype(np.float32) * t + A * (1.0 - t)
    return np.clip(result, 0, 255).astype(np.uint8)


def degrade_rain_like(
    img: np.ndarray,
    n_lines: int,
    length: int,
    angle: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate rain-like visual degradation by drawing semi-transparent
    bright diagonal streaks on the image.

    Limitation: Real rain streaks are complex optical phenomena. These are
    simplified synthetic streaks that simulate visual clutter, not photorealistic
    rain. Claims are limited to performance under this specific visual degradation.
    """
    h, w = img.shape[:2]
    overlay = img.copy()
    angle_rad = np.deg2rad(angle)
    dx = int(length * np.cos(angle_rad))
    dy = int(length * np.sin(angle_rad))

    for _ in range(n_lines):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        x1 = np.clip(x0 + dx, 0, w - 1)
        y1 = np.clip(y0 - dy, 0, h - 1)
        brightness = int(rng.integers(180, 255))
        cv2.line(overlay, (x0, y0), (x1, y1), (brightness, brightness, brightness), 1)

    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


# ─── Dispatcher ──────────────────────────────────────────────────────────────

def apply_degradation(
    img: np.ndarray,
    condition: str,
    severity: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply a named degradation at a specified severity level.

    Parameters
    ----------
    img       : BGR image (H, W, 3)
    condition : one of CONDITIONS
    severity  : 'mild', 'moderate', or 'severe'
    rng       : seeded numpy random generator (for reproducibility)

    Returns
    -------
    Degraded BGR image, same dtype and shape as input.
    Bounding-box coordinates are NOT affected (pixel-level ops only).
    """
    p = DEGRADATION_PARAMS[condition][severity]

    if condition == "low_light":
        return degrade_low_light(img, p["gamma"])
    elif condition == "low_contrast":
        return degrade_low_contrast(img, p["alpha"], p["beta"])
    elif condition == "noise":
        return degrade_noise(img, p["sigma"], rng)
    elif condition == "blur":
        return degrade_blur(img, p["ksize"])
    elif condition == "haze":
        return degrade_haze(img, p["density"], p["airlight"])
    elif condition == "rain_like":
        return degrade_rain_like(img, p["n_lines"], p["length"], p["angle"], p["alpha"], rng)
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run_degradation_pipeline(
    sample_limit: int = 0,
    save_samples: int = 5,
    overwrite: bool = False,
) -> dict:
    """
    Generate all degraded image sets from the RDD2022 test split.

    For each (condition, severity) pair, creates degraded versions of all
    test images and saves them to data/phase2/<condition>/<severity>/.
    Also copies the originals to data/phase2/clean/original/.

    Parameters
    ----------
    sample_limit : If > 0, only process this many images (for quick testing)
    save_samples : Number of side-by-side sample pairs to save for each condition
    overwrite    : If False, skip images that already exist on disk

    Notes
    -----
    - Original images are NOT modified.
    - Labels (bounding boxes) are NOT copied or modified — evaluator uses originals.
    - Random seed is fixed per-image using (SEED + image_index) for full reproducibility.
    """
    logmsg("STEP 1 (Phase 2): Generating degraded image sets...")
    logmsg(f"  Source: {RDD_ROOT / 'test' / 'images'}")

    img_paths = get_test_image_paths(limit=sample_limit if sample_limit > 0 else None)
    n_images = len(img_paths)
    logmsg(f"  Total test images: {n_images}")

    samples_dir = PROJECT_ROOT / "results" / "phase2" / "degradation_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    stats = {"n_images": n_images, "conditions": {}}

    # Step 1a: Copy clean images
    logmsg("  Copying clean images...")
    clean_dir = DATA_ROOT / "clean" / "original"
    clean_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for p in img_paths:
        dest = clean_dir / p.name
        if not dest.exists() or overwrite:
            shutil.copy2(p, dest)
            n_copied += 1
    logmsg(f"  Clean images ready: {n_copied} copied, {n_images - n_copied} already existed.")

    # Step 1b: Generate degraded images
    for condition in CONDITIONS:
        logmsg(f"  Condition: {condition}")
        stats["conditions"][condition] = {}
        for severity in SEVERITIES:
            out_dir = DATA_ROOT / condition / severity
            out_dir.mkdir(parents=True, exist_ok=True)

            n_done = 0
            n_skipped = 0
            sample_saved = 0

            for idx, img_path in enumerate(img_paths):
                out_path = out_dir / img_path.name

                if out_path.exists() and not overwrite:
                    n_skipped += 1
                    continue

                # Per-image seeded RNG for full reproducibility
                rng = np.random.default_rng(SEED + idx)

                img = read_image(str(img_path))
                deg = apply_degradation(img, condition, severity, rng)
                write_image(str(out_path), deg)
                n_done += 1

                # Save a side-by-side comparison sample
                if sample_saved < save_samples:
                    pair = np.hstack([img, deg])
                    sample_path = samples_dir / f"{condition}_{severity}_sample{sample_saved:02d}.jpg"
                    cv2.imwrite(str(sample_path), pair)
                    sample_saved += 1

            total = n_done + n_skipped
            logmsg(f"    [{severity}]: {n_done} generated, {n_skipped} skipped ({total} total)")
            stats["conditions"][condition][severity] = {
                "n_generated": n_done,
                "n_skipped": n_skipped,
                "n_total": total,
            }

    logmsg("  Degradation pipeline complete.")
    return stats


if __name__ == "__main__":
    run_degradation_pipeline(save_samples=5)
