# phase2/condition_analyzer.py
"""
Lightweight image condition analyzer for Phase 2.

Extracts classical computer-vision features from an image and classifies it
into one of several condition categories. Uses NO neural networks — purely
classical signal processing metrics.

Condition labels (in priority order when multiple are detected):
    NORMAL       - image appears visually normal
    LOW_LIGHT    - image is significantly darker than expected
    LOW_CONTRAST - dynamic range is compressed / image is flat
    NOISY        - high-frequency noise is above threshold
    BLUR         - image sharpness (Laplacian variance) is low
    HAZY         - brightness is high and contrast is low simultaneously

Design notes:
    - All thresholds were tuned on the RDD2022 VALIDATION set (not test set).
    - A single dominant condition is returned per image.
    - When multiple conditions co-occur, a priority order is enforced.
    - Thresholds are intentionally conservative to avoid over-classification.

Limitations:
    - Classical features are heuristic approximations.
    - Performance on edge cases (e.g., night + fog simultaneously) may be
      incorrect. Multi-condition co-occurrence is documented as a limitation.
    - Thresholds calibrated for RDD2022 image characteristics; may not
      generalize to different datasets without re-calibration.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


# ─── Data structure ──────────────────────────────────────────────────────────

@dataclass
class ImageCondition:
    """
    Holds all computed condition metrics and the final classification.

    Attributes
    ----------
    label             : Primary condition label (e.g., 'LOW_LIGHT')
    mean_brightness   : Mean pixel brightness (0–255, grayscale)
    brightness_std    : Std dev of pixel brightness
    contrast_rms      : RMS contrast (std of grayscale / mean * 100)
    laplacian_var     : Variance of Laplacian (sharpness estimate)
    noise_estimate    : Estimated noise level (high-frequency energy)
    mean_saturation   : Mean HSV saturation (0–255)
    haze_score        : Heuristic haze score (high brightness + low contrast)
    confidence        : Rough confidence in the classification (0–1)
    all_scores        : Dict of all feature values
    """
    label: str
    mean_brightness: float
    brightness_std: float
    contrast_rms: float
    laplacian_var: float
    noise_estimate: float
    mean_saturation: float
    haze_score: float
    confidence: float
    all_scores: Dict[str, float]


# ─── Thresholds ──────────────────────────────────────────────────────────────
# These thresholds were set by analysing the distribution of feature values
# across the RDD2022 validation set (clean images) and the synthetically
# degraded validation images.
# They are FROZEN after initial calibration — the test set is NOT used for tuning.

THRESHOLDS = {
    # Mean grayscale brightness thresholds
    "low_light_max_brightness":  75.0,   # Mean < 75 → dark image
    "hazy_min_brightness":      165.0,   # Mean > 165 AND low contrast → hazy

    # RMS contrast thresholds
    "low_contrast_max_rms":      28.0,   # RMS contrast < 28 → flat/washed out
    "hazy_max_rms":              35.0,   # Used in haze detection

    # Sharpness (Laplacian variance)
    "blur_max_laplacian":       120.0,   # Lap var < 120 → likely blurred

    # Noise estimate (high-frequency energy via diff of blurred versions)
    "noise_min_estimate":        12.0,   # Noise level > 12 → noisy
}


# ─── Feature extraction ──────────────────────────────────────────────────────

def extract_features(img: np.ndarray) -> Dict[str, float]:
    """
    Extract classical image quality features.

    Parameters
    ----------
    img : BGR image (H, W, 3), uint8

    Returns
    -------
    Dictionary of feature_name → float value
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── Brightness ──────────────────────────────────────────────────────────
    mean_brightness = float(np.mean(gray))
    brightness_std  = float(np.std(gray))

    # ── Contrast (RMS contrast) ──────────────────────────────────────────────
    # RMS contrast = std / mean * 100 (coefficient of variation style)
    # Avoids division by zero for near-black images
    mean_safe = max(mean_brightness, 1.0)
    contrast_rms = float(np.std(gray) / mean_safe * 100)

    # ── Sharpness (Laplacian variance) ───────────────────────────────────────
    # High variance → sharp image; low variance → blurred
    laplacian    = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    laplacian_var = float(laplacian.var())

    # ── Noise estimate ───────────────────────────────────────────────────────
    # Noise ≈ difference between image and its Gaussian-smoothed version
    # Captures high-frequency content that isn't due to edges.
    smoothed   = cv2.GaussianBlur(gray, (5, 5), 0)
    diff       = np.abs(gray - smoothed)
    noise_estimate = float(np.mean(diff))

    # ── Saturation (HSV) ─────────────────────────────────────────────────────
    hsv             = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    # ── Haze score ───────────────────────────────────────────────────────────
    # Haze tends to: increase brightness, decrease contrast, reduce saturation
    # Haze score = normalized combination
    haze_score = float(
        (mean_brightness / 255.0) * 0.5
        + (1.0 - contrast_rms / 100.0) * 0.3
        + (1.0 - mean_saturation / 255.0) * 0.2
    )

    return {
        "mean_brightness": mean_brightness,
        "brightness_std":  brightness_std,
        "contrast_rms":    contrast_rms,
        "laplacian_var":   laplacian_var,
        "noise_estimate":  noise_estimate,
        "mean_saturation": mean_saturation,
        "haze_score":      haze_score,
    }


# ─── Classification logic ────────────────────────────────────────────────────

def classify_condition(features: Dict[str, float]) -> tuple:
    """
    Classify image condition from extracted features.

    Priority order (to resolve multi-condition co-occurrence):
        1. LOW_LIGHT   — darkness takes priority over other conditions
        2. HAZY        — haze check before low-contrast (haze ⊃ low-contrast)
        3. LOW_CONTRAST— flat appearance but not dark or hazy
        4. NOISY       — excessive high-frequency energy
        5. BLUR        — low sharpness (after noise, to avoid confusion)
        6. NORMAL      — none of the above

    Returns
    -------
    (label: str, confidence: float)
    """
    t = THRESHOLDS
    mb  = features["mean_brightness"]
    rms = features["contrast_rms"]
    lv  = features["laplacian_var"]
    ne  = features["noise_estimate"]
    hs  = features["haze_score"]

    # 1. LOW_LIGHT
    if mb < t["low_light_max_brightness"]:
        conf = 1.0 - mb / t["low_light_max_brightness"]
        return "LOW_LIGHT", round(min(conf, 1.0), 3)

    # 2. HAZY (high brightness + low contrast)
    if mb > t["hazy_min_brightness"] and rms < t["hazy_max_rms"]:
        conf = hs
        return "HAZY", round(min(conf, 1.0), 3)

    # 3. LOW_CONTRAST
    if rms < t["low_contrast_max_rms"]:
        conf = 1.0 - rms / t["low_contrast_max_rms"]
        return "LOW_CONTRAST", round(min(conf, 1.0), 3)

    # 4. NOISY
    if ne > t["noise_min_estimate"]:
        conf = min((ne - t["noise_min_estimate"]) / t["noise_min_estimate"], 1.0)
        return "NOISY", round(conf, 3)

    # 5. BLUR
    if lv < t["blur_max_laplacian"]:
        conf = 1.0 - lv / t["blur_max_laplacian"]
        return "BLUR", round(min(conf, 1.0), 3)

    # 6. NORMAL
    return "NORMAL", 1.0


# ─── Public API ──────────────────────────────────────────────────────────────

def analyze_image(img: np.ndarray) -> ImageCondition:
    """
    Full pipeline: extract features → classify condition.

    Parameters
    ----------
    img : BGR image (H, W, 3)

    Returns
    -------
    ImageCondition dataclass with all metrics and final label
    """
    features = extract_features(img)
    label, confidence = classify_condition(features)

    return ImageCondition(
        label=label,
        mean_brightness=features["mean_brightness"],
        brightness_std=features["brightness_std"],
        contrast_rms=features["contrast_rms"],
        laplacian_var=features["laplacian_var"],
        noise_estimate=features["noise_estimate"],
        mean_saturation=features["mean_saturation"],
        haze_score=features["haze_score"],
        confidence=confidence,
        all_scores=features,
    )


def analyze_image_from_path(path: str) -> ImageCondition:
    """Convenience wrapper that reads image from path."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return analyze_image(img)


# ─── Calibration helper (run on validation set, NOT test set) ────────────────

def calibrate_thresholds(val_image_dir: str, n_samples: int = 500) -> Dict[str, Dict]:
    """
    Compute feature statistics on clean validation images.
    Used to understand the feature distribution of 'NORMAL' images.
    THIS IS ONLY FOR DEVELOPMENT — thresholds are frozen for final evaluation.

    Parameters
    ----------
    val_image_dir : path to clean validation images
    n_samples     : number of images to sample

    Returns
    -------
    Dict of feature_name → {mean, std, p5, p95}
    """
    import os, random
    paths = [
        os.path.join(val_image_dir, f)
        for f in os.listdir(val_image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.seed(42)
    random.shuffle(paths)
    paths = paths[:n_samples]

    all_feats = {k: [] for k in [
        "mean_brightness", "brightness_std", "contrast_rms",
        "laplacian_var", "noise_estimate", "mean_saturation", "haze_score"
    ]}

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        feats = extract_features(img)
        for k, v in feats.items():
            all_feats[k].append(v)

    stats = {}
    for k, vals in all_feats.items():
        arr = np.array(vals)
        stats[k] = {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "p5":   float(np.percentile(arr, 5)),
            "p25":  float(np.percentile(arr, 25)),
            "p75":  float(np.percentile(arr, 75)),
            "p95":  float(np.percentile(arr, 95)),
        }
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cond = analyze_image_from_path(sys.argv[1])
        print(f"Condition: {cond.label} (confidence={cond.confidence:.3f})")
        print(f"  Brightness: {cond.mean_brightness:.1f}")
        print(f"  Contrast:   {cond.contrast_rms:.1f}")
        print(f"  Sharpness:  {cond.laplacian_var:.1f}")
        print(f"  Noise:      {cond.noise_estimate:.2f}")
        print(f"  Haze score: {cond.haze_score:.3f}")
