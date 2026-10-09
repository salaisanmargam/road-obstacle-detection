# phase3/integration/pipeline.py
"""
End-to-End Secure Vision Pipeline for Phase 3.

Implements four system configurations:
    System A: YOLO only (Phase 1 reference)
    System B: Adaptive Enhancement + YOLO (Phase 2 reference)
    System C: Security + YOLO (Phase 3, no enhancement)
    System D: Security + Adaptive Enhancement + YOLO (Phase 3 proposed)

Phase 2 optimization (Part G):
    In System D, enhancement is SKIPPED when the image condition is NORMAL.
    This reduces average latency for real-world images that are already clean.

DISCLAIMER: The security gateway in this pipeline uses CIC-IDS2017 flow
features for demonstration. In a real deployment, these features would come
from actual network monitoring. This is a controlled research simulation.
"""

import time
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import sys

# Phase 2 imports
sys.path.insert(0, str(Path("D:/road obstacle detection/road_damage_research")))
from phase2.condition_analyzer import analyze_image
from phase2.adaptive_policy import adaptive_enhance, no_enhance

YOLO_CHECKPOINT = Path(
    "D:/road obstacle detection/road_damage_research"
    "/experiments/phase1_final_baseline/final/weights/best.pt"
)


def _load_yolo():
    from ultralytics import YOLO
    model = YOLO(str(YOLO_CHECKPOINT))
    return model


class LatencyTracker:
    def __init__(self):
        self.security_ms   = []
        self.condition_ms  = []
        self.enhance_ms    = []
        self.yolo_ms       = []
        self.total_ms      = []
        self.n_enhanced    = 0
        self.n_bypassed    = 0
        self.n_suspicious  = 0
        self.n_processed   = 0

    def summary(self) -> Dict:
        def _ms(lst):
            return round(float(np.mean(lst)), 2) if lst else 0.0
        total = self.n_enhanced + self.n_bypassed + self.n_suspicious
        return {
            "avg_security_ms":    _ms(self.security_ms),
            "avg_condition_ms":   _ms(self.condition_ms),
            "avg_enhance_ms":     _ms(self.enhance_ms),
            "avg_yolo_ms":        _ms(self.yolo_ms),
            "avg_total_ms":       _ms(self.total_ms),
            "fps":                round(1000.0 / max(_ms(self.total_ms), 0.001), 1),
            "n_enhanced":         self.n_enhanced,
            "n_bypassed":         self.n_bypassed,
            "n_suspicious":       self.n_suspicious,
            "n_total":            total,
            "enhance_rate_pct":   round(100.0 * self.n_enhanced / max(self.n_processed, 1), 1),
        }


class SecureVisionPipeline:
    """
    Security-aware road damage detection pipeline.

    Parameters
    ----------
    gateway : SecurityGateway or None
        If None, security check is skipped.
    use_enhancement : bool
        If True, apply Phase 2 adaptive enhancement.
    bypass_normal : bool
        If True (and use_enhancement=True), skip enhancement when
        condition is NORMAL (Phase 2 optimization).
    """

    def __init__(self, gateway=None, use_enhancement: bool = True,
                 bypass_normal: bool = True):
        print("  Loading YOLO model...")
        self.yolo   = _load_yolo()
        self.gateway = gateway
        self.use_enhancement = use_enhancement
        self.bypass_normal   = bypass_normal
        self.tracker = LatencyTracker()
        print(f"  Pipeline config: security={gateway is not None} | "
              f"enhancement={use_enhancement} | bypass_normal={bypass_normal}")

    def process(self, img_path: str,
                flow_features: Optional[np.ndarray] = None,
                conf: float = 0.001, iou: float = 0.6):
        """
        Process a single image through the full pipeline.

        Parameters
        ----------
        img_path       : Path to image file
        flow_features  : Optional network flow features for security check.
                         If None and gateway is not None, traffic is assumed NORMAL.
        conf, iou      : YOLO detection thresholds

        Returns
        -------
        dict with keys: 'result', 'label', 'method', 'latency_breakdown'
        """
        t_start = time.perf_counter()

        # ── Step 1: Security check ────────────────────────────────────────────
        security_decision = None
        t_sec_ms = 0.0
        if self.gateway is not None and flow_features is not None:
            t0 = time.perf_counter()
            security_decision = self.gateway.classify(flow_features)
            t_sec_ms = (time.perf_counter() - t0) * 1000
            self.tracker.security_ms.append(t_sec_ms)

            if security_decision.is_suspicious:
                self.tracker.n_suspicious += 1
                total_ms = (time.perf_counter() - t_start) * 1000
                self.tracker.total_ms.append(total_ms)
                return {
                    "result": None,
                    "label": "SUSPICIOUS",
                    "method": "blocked",
                    "latency_breakdown": {
                        "security_ms": t_sec_ms, "total_ms": total_ms
                    },
                }

        # ── Step 2: Load image ────────────────────────────────────────────────
        img = cv2.imread(str(img_path))
        if img is None:
            return {"result": None, "label": "ERROR", "method": "error",
                    "latency_breakdown": {}}

        # ── Step 3: Condition analysis + optional enhancement ─────────────────
        t_cond_ms = 0.0
        t_enh_ms  = 0.0
        method_name = "none"
        condition_label = "UNKNOWN"

        if self.use_enhancement:
            t0 = time.perf_counter()
            condition = analyze_image(img)
            t_cond_ms = (time.perf_counter() - t0) * 1000
            condition_label = condition.label
            self.tracker.condition_ms.append(t_cond_ms)

            if self.bypass_normal and condition.label == "NORMAL":
                # Bypass: image is fine, skip enhancement
                enhanced = img
                method_name = "bypass"
                self.tracker.n_bypassed += 1
            else:
                t0 = time.perf_counter()
                enhanced, _, method_name = adaptive_enhance(img)
                t_enh_ms = (time.perf_counter() - t0) * 1000
                self.tracker.enhance_ms.append(t_enh_ms)
                self.tracker.n_enhanced += 1
        else:
            enhanced = img

        self.tracker.n_processed += 1

        # ── Step 4: YOLO inference ────────────────────────────────────────────
        t0 = time.perf_counter()
        results = self.yolo(enhanced, conf=conf, iou=iou, verbose=False)
        t_yolo_ms = (time.perf_counter() - t0) * 1000
        self.tracker.yolo_ms.append(t_yolo_ms)

        total_ms = (time.perf_counter() - t_start) * 1000
        self.tracker.total_ms.append(total_ms)

        return {
            "result":    results,
            "label":     "NORMAL",
            "condition": condition_label,
            "method":    method_name,
            "latency_breakdown": {
                "security_ms":  round(t_sec_ms,  2),
                "condition_ms": round(t_cond_ms, 2),
                "enhance_ms":   round(t_enh_ms,  2),
                "yolo_ms":      round(t_yolo_ms, 2),
                "total_ms":     round(total_ms,  2),
            },
        }

    def get_latency_summary(self) -> Dict:
        return self.tracker.summary()


def measure_system_latency(img_paths: list, system_name: str,
                            gateway=None, use_enhancement: bool = True,
                            bypass_normal: bool = True,
                            n_warmup: int = 5) -> Dict:
    """
    Measure average inference latency for a given system configuration.

    Parameters
    ----------
    img_paths      : list of image file paths
    system_name    : label for the system (A/B/C/D)
    gateway        : SecurityGateway or None
    use_enhancement: whether to apply Phase 2 enhancement
    bypass_normal  : whether to bypass enhancement for normal images
    n_warmup       : number of warm-up images before measuring

    Returns
    -------
    dict with latency statistics
    """
    print(f"\n--- Measuring latency: {system_name} ---")
    pipe = SecureVisionPipeline(
        gateway=gateway,
        use_enhancement=use_enhancement,
        bypass_normal=bypass_normal,
    )

    # Warm-up
    for p in img_paths[:n_warmup]:
        pipe.process(str(p))

    # Reset tracker after warm-up
    pipe.tracker = LatencyTracker()

    # Measure
    for p in img_paths:
        pipe.process(str(p))

    summary = pipe.get_latency_summary()
    summary["system"] = system_name
    print(f"  Avg total latency: {summary['avg_total_ms']} ms | FPS: {summary['fps']}")
    return summary
