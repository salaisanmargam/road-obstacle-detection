# phase2/enhancement_methods.py
"""
Classical image enhancement methods for Phase 2.

Each method is independently callable and operates on BGR images.
Methods are selected by the adaptive policy based on detected image condition.

Enhancement methods implemented:
    1. no_enhancement      - pass-through (returns image unchanged)
    2. clahe               - CLAHE on L-channel in LAB color space
    3. gamma_correction    - power-law gamma brightening
    4. contrast_stretch    - histogram stretching to [0,255]
    5. bilateral_denoise   - edge-preserving Gaussian noise reduction
    6. nlm_denoise         - Non-Local Means denoising (higher quality, slower)
    7. unsharp_mask        - sharpening via unsharp masking
    8. laplacian_sharpen   - Laplacian-based edge sharpening
    9. dehaze_clahe        - dark channel prior + CLAHE for haze removal

Design principles:
    - Each method is a pure function: (img: ndarray, **params) → ndarray
    - All methods preserve spatial dimensions.
    - All methods accept only BGR uint8 images.
    - Methods are tuned to be conservative — avoid over-enhancement.
    - No method applies multiple stacked enhancements internally.
"""

import cv2
import numpy as np


# ─── 1. No enhancement ───────────────────────────────────────────────────────

def no_enhancement(img: np.ndarray, **kwargs) -> np.ndarray:
    """Pass-through: return image unchanged."""
    return img.copy()


# ─── 2. CLAHE ────────────────────────────────────────────────────────────────

def clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
    **kwargs,
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) applied to the
    L-channel in LAB color space.

    Works well for low-contrast images and haze. Preserves color information
    better than applying CLAHE to each BGR channel independently.

    Parameters
    ----------
    clip_limit     : Contrast clipping limit (higher = more contrast boost)
    tile_grid_size : Grid size for local histogram computation
    """
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe_obj.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ─── 3. Gamma correction ─────────────────────────────────────────────────────

def gamma_correction(
    img: np.ndarray,
    gamma: float = 1.8,
    **kwargs,
) -> np.ndarray:
    """
    Power-law gamma correction: out = in^(1/gamma).

    gamma > 1 → brightens the image (corrects low-light darkening).
    gamma < 1 → darkens the image.
    gamma = 1 → identity.

    Parameters
    ----------
    gamma : Correction factor. Typical brightening: 1.5–2.5.
    """
    if gamma <= 0:
        return img.copy()
    lut = np.array(
        [int(((i / 255.0) ** (1.0 / gamma)) * 255 + 0.5) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, lut)


# ─── 4. Contrast stretching ──────────────────────────────────────────────────

def contrast_stretch(
    img: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    **kwargs,
) -> np.ndarray:
    """
    Stretch image contrast using percentile-based histogram clipping.
    Maps [p_low, p_high] → [0, 255] to avoid extreme outlier pixels
    washing out the stretch.

    Parameters
    ----------
    percentile_low  : Lower clip percentile (typically 1–5)
    percentile_high : Upper clip percentile (typically 95–99)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    p_lo = np.percentile(gray, percentile_low)
    p_hi = np.percentile(gray, percentile_high)

    if p_hi - p_lo < 1.0:
        return img.copy()  # Already at full contrast

    result = img.astype(np.float32)
    result = (result - p_lo) / (p_hi - p_lo) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── 5. Bilateral denoising ──────────────────────────────────────────────────

def bilateral_denoise(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    **kwargs,
) -> np.ndarray:
    """
    Edge-preserving denoising using bilateral filter.
    Reduces Gaussian noise while preserving edges (useful for road cracks).

    Parameters
    ----------
    d           : Diameter of pixel neighborhood (higher = slower)
    sigma_color : Color range sigma (higher = more color blending)
    sigma_space : Spatial sigma (higher = farther pixel influence)
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ─── 6. Non-Local Means denoising ────────────────────────────────────────────

def nlm_denoise(
    img: np.ndarray,
    h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
    **kwargs,
) -> np.ndarray:
    """
    Non-Local Means (NLM) denoising.
    Higher quality than bilateral for strong noise, but significantly slower.

    Parameters
    ----------
    h                   : Filter strength (higher = more denoising, more blur)
    template_window_size: Patch size for similarity computation
    search_window_size  : Search area size
    """
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h,
        h,
        template_window_size,
        search_window_size,
    )


# ─── 7. Unsharp masking ──────────────────────────────────────────────────────

def unsharp_mask(
    img: np.ndarray,
    sigma: float = 1.5,
    strength: float = 1.5,
    **kwargs,
) -> np.ndarray:
    """
    Sharpen image using unsharp masking:
        sharpened = img + strength * (img - blurred)

    Effective for recovering some sharpness in mildly blurred images.
    Conservative strength to avoid halos or artifacts.

    Parameters
    ----------
    sigma    : Gaussian blur sigma for the mask
    strength : Sharpening intensity multiplier (0.5–2.0 is typical)
    """
    blurred   = cv2.GaussianBlur(img, (0, 0), sigma)
    mask      = img.astype(np.float32) - blurred.astype(np.float32)
    sharpened = img.astype(np.float32) + strength * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ─── 8. Laplacian sharpening ─────────────────────────────────────────────────

def laplacian_sharpen(
    img: np.ndarray,
    alpha: float = 0.7,
    **kwargs,
) -> np.ndarray:
    """
    Sharpen using Laplacian edge enhancement:
        sharpened = img + alpha * Laplacian(img)

    Parameters
    ----------
    alpha : Edge boost strength (0.3–1.0 recommended)
    """
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap      = cv2.Laplacian(gray, cv2.CV_64F)
    result   = img.astype(np.float32)
    for c in range(3):
        result[:, :, c] += alpha * lap
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── 9. CLAHE-based dehaze ───────────────────────────────────────────────────

def dehaze_clahe(
    img: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: tuple = (8, 8),
    gamma: float = 1.2,
    **kwargs,
) -> np.ndarray:
    """
    Haze removal using a two-step approach:
        Step 1: CLAHE on L-channel to restore contrast lost to scattering
        Step 2: Mild gamma correction to recover darker regions

    This is a simplified approximation — not a full dark-channel-prior
    implementation. It works well for uniform/mild haze.

    Limitation: For severe non-uniform haze, a full DCP implementation
    would be more effective but is computationally heavier.

    Parameters
    ----------
    clip_limit     : CLAHE clip limit (higher for heavy haze)
    tile_grid_size : CLAHE tile grid
    gamma          : Post-CLAHE gamma boost
    """
    # Step 1: CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enh = clahe_obj.apply(l)
    lab_enh = cv2.merge([l_enh, a, b])
    dehazed = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

    # Step 2: Gamma correction
    lut = np.array(
        [int(((i / 255.0) ** (1.0 / gamma)) * 255 + 0.5) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(dehazed, lut)


# ─── Method registry ─────────────────────────────────────────────────────────

# Maps method name → callable
ENHANCEMENT_REGISTRY = {
    "none":              no_enhancement,
    "clahe":             clahe,
    "gamma":             gamma_correction,
    "contrast_stretch":  contrast_stretch,
    "bilateral_denoise": bilateral_denoise,
    "nlm_denoise":       nlm_denoise,
    "unsharp_mask":      unsharp_mask,
    "laplacian_sharpen": laplacian_sharpen,
    "dehaze_clahe":      dehaze_clahe,
}

# Default parameters for each method when used in the adaptive pipeline
DEFAULT_PARAMS = {
    "none":              {},
    "clahe":             {"clip_limit": 2.0, "tile_grid_size": (8, 8)},
    "gamma":             {"gamma": 1.8},
    "contrast_stretch":  {"percentile_low": 2.0, "percentile_high": 98.0},
    "bilateral_denoise": {"d": 9, "sigma_color": 75.0, "sigma_space": 75.0},
    "nlm_denoise":       {"h": 10.0, "template_window_size": 7, "search_window_size": 21},
    "unsharp_mask":      {"sigma": 1.5, "strength": 1.5},
    "laplacian_sharpen": {"alpha": 0.7},
    "dehaze_clahe":      {"clip_limit": 3.0, "tile_grid_size": (8, 8), "gamma": 1.2},
}


def apply_enhancement(img: np.ndarray, method: str, **override_params) -> np.ndarray:
    """
    Apply a named enhancement method to an image using default parameters,
    with optional overrides.

    Parameters
    ----------
    img            : BGR image (H, W, 3), uint8
    method         : Name from ENHANCEMENT_REGISTRY
    **override_params : Optional parameter overrides

    Returns
    -------
    Enhanced BGR image, same shape and dtype as input.
    """
    if method not in ENHANCEMENT_REGISTRY:
        raise ValueError(
            f"Unknown enhancement method: '{method}'. "
            f"Available: {list(ENHANCEMENT_REGISTRY.keys())}"
        )
    params = {**DEFAULT_PARAMS.get(method, {}), **override_params}
    fn     = ENHANCEMENT_REGISTRY[method]
    return fn(img, **params)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        img = cv2.imread(sys.argv[1])
        method = sys.argv[2]
        enhanced = apply_enhancement(img, method)
        out_path = f"enhanced_{method}.jpg"
        cv2.imwrite(out_path, enhanced)
        print(f"Saved: {out_path}")
    else:
        print("Usage: python enhancement_methods.py <image_path> <method>")
        print(f"Methods: {list(ENHANCEMENT_REGISTRY.keys())}")
