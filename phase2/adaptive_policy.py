# phase2/adaptive_policy.py
"""
Adaptive Enhancement Policy for Phase 2.

The policy maps detected image conditions to the most appropriate enhancement
method. This is the central component of Phase 2.

Design principles:
    - ADAPTIVE: a different enhancement is selected per condition.
    - CONSERVATIVE: avoid stacking multiple methods or over-enhancing.
    - DOCUMENTED: every mapping decision is explained below.
    - FROZEN: the policy is finalized using the validation set and must NOT
      be changed based on test-set results.

Policy Mapping (condition → method):
────────────────────────────────────────────────────────────────
 NORMAL        → none              (don't alter normal images)
 LOW_LIGHT     → gamma             (inverse gamma to brighten)
 LOW_CONTRAST  → clahe             (CLAHE to restore local contrast)
 NOISY         → bilateral_denoise (edge-preserving denoising)
 BLUR          → unsharp_mask      (frequency-domain sharpening)
 HAZY          → dehaze_clahe      (CLAHE + gamma for scattering)
 UNKNOWN       → none              (safe fallback)
────────────────────────────────────────────────────────────────

Fixed Enhancement (for Experiment C — comparison baseline):
    The "fixed enhancement" baseline applies CLAHE to ALL images regardless
    of condition. CLAHE is chosen as the fixed method because:
    - It is the most commonly used road-image enhancement in the literature.
    - It does not require condition detection.
    - It provides a fair representative single-method baseline.

Scientific note:
    This policy does NOT guarantee improvement in all cases. The hypothesis
    is that adaptive selection outperforms fixed enhancement ON AVERAGE across
    diverse conditions. Individual condition-method pairs may still reduce mAP.
"""

from typing import Dict, Optional

from .condition_analyzer import ImageCondition, analyze_image
from .enhancement_methods import apply_enhancement, ENHANCEMENT_REGISTRY

import cv2
import numpy as np


# ─── Policy table ────────────────────────────────────────────────────────────

# Maps condition label → enhancement method name
# Method names must be keys in ENHANCEMENT_REGISTRY
ADAPTIVE_POLICY: Dict[str, str] = {
    "NORMAL":       "none",
    "LOW_LIGHT":    "gamma",
    "LOW_CONTRAST": "clahe",
    "NOISY":        "bilateral_denoise",
    "BLUR":         "unsharp_mask",
    "HAZY":         "dehaze_clahe",
    "UNKNOWN":      "none",   # safe fallback
}

# Fixed enhancement used in Experiment C (applied to ALL images)
FIXED_ENHANCEMENT_METHOD = "clahe"

# Enhancement parameters for each condition-specific call
# (overrides DEFAULT_PARAMS where needed for condition-specific tuning)
CONDITION_SPECIFIC_PARAMS: Dict[str, Dict] = {
    "NORMAL":       {},
    "LOW_LIGHT":    {"gamma": 1.9},        # Stronger correction for dark images
    "LOW_CONTRAST": {"clip_limit": 2.5},   # Slightly stronger CLAHE for flat images
    "NOISY":        {"d": 9, "sigma_color": 80.0, "sigma_space": 80.0},
    "BLUR":         {"sigma": 1.5, "strength": 1.8},
    "HAZY":         {"clip_limit": 3.5, "gamma": 1.3},
    "UNKNOWN":      {},
}


# ─── Adaptive pipeline ───────────────────────────────────────────────────────

def adaptive_enhance(img: np.ndarray) -> tuple:
    """
    Full adaptive enhancement pipeline:
        1. Analyze image condition
        2. Select appropriate enhancement
        3. Apply enhancement

    Parameters
    ----------
    img : BGR image (H, W, 3), uint8

    Returns
    -------
    (enhanced_img, condition, method_name)
        enhanced_img  : BGR image (H, W, 3), uint8
        condition     : ImageCondition dataclass
        method_name   : Name of enhancement applied
    """
    # Step 1: Analyze
    condition = analyze_image(img)

    # Step 2: Select
    method_name = ADAPTIVE_POLICY.get(condition.label, "none")

    # Step 3: Apply with condition-specific parameters
    params = CONDITION_SPECIFIC_PARAMS.get(condition.label, {})
    enhanced = apply_enhancement(img, method_name, **params)

    return enhanced, condition, method_name


def fixed_enhance(img: np.ndarray) -> np.ndarray:
    """
    Fixed enhancement (Experiment C baseline):
    Apply CLAHE regardless of detected image condition.
    """
    return apply_enhancement(img, FIXED_ENHANCEMENT_METHOD)


def no_enhance(img: np.ndarray) -> np.ndarray:
    """No enhancement — pass-through (used in Experiments A and B)."""
    return img.copy()


# ─── Batch processing ────────────────────────────────────────────────────────

def process_image_adaptive(img_path: str) -> tuple:
    """
    Read image from path, apply adaptive enhancement, return result.

    Returns
    -------
    (original, enhanced, condition, method_name)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    enhanced, condition, method = adaptive_enhance(img)
    return img, enhanced, condition, method


def process_image_fixed(img_path: str) -> tuple:
    """
    Read image from path, apply fixed CLAHE enhancement.

    Returns
    -------
    (original, enhanced)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    enhanced = fixed_enhance(img)
    return img, enhanced


# ─── Policy description ──────────────────────────────────────────────────────

def get_policy_description() -> str:
    """Return human-readable policy description for reports."""
    lines = [
        "Adaptive Enhancement Policy (Phase 2)",
        "=" * 50,
        "",
        "Condition → Enhancement Method:",
    ]
    for cond, method in ADAPTIVE_POLICY.items():
        params = CONDITION_SPECIFIC_PARAMS.get(cond, {})
        param_str = f" | params: {params}" if params else ""
        lines.append(f"  {cond:<15} → {method}{param_str}")
    lines += [
        "",
        f"Fixed Enhancement (Exp C): {FIXED_ENHANCEMENT_METHOD} (all images)",
        "",
        "Priority order for condition classification:",
        "  1. LOW_LIGHT  (brightness < 75)",
        "  2. HAZY       (brightness > 165 AND contrast < 35)",
        "  3. LOW_CONTRAST (contrast < 28)",
        "  4. NOISY      (noise estimate > 12)",
        "  5. BLUR       (Laplacian variance < 120)",
        "  6. NORMAL     (none of the above)",
    ]
    return "\n".join(lines)
