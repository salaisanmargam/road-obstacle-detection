import time
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path("D:/road obstacle detection/road_damage_research")))
from phase2.condition_analyzer import analyze_image
from phase2.adaptive_policy import adaptive_enhance
from phase3.security.security_gateway import SecurityGateway
from ultralytics import YOLO

class OptimizedVideoPipeline:
    """
    Optimized continuous video pipeline for near-real-time operation.
    Implements:
    - Temporal Condition Reuse (avoid analyzing condition every frame)
    - Selective Enhancement Bypass (skip enhancement if NORMAL)
    - Temporal Security Reuse (avoid checking network flow every single frame)
    - Half-precision YOLO inference (FP16)
    """
    def __init__(self, yolo_ckpt, gateway=None, condition_interval=5, security_interval=30):
        self.yolo = YOLO(yolo_ckpt)
        self.gateway = gateway
        
        self.condition_interval = condition_interval
        self.security_interval = security_interval
        
        self.frame_count = 0
        self.last_condition_label = "UNKNOWN"
        self.last_condition_obj = None
        self.last_security_decision = None
        
        self.tracker = {
            "security_ms": [], "condition_ms": [], 
            "enhance_ms": [], "yolo_ms": [], "total_ms": []
        }

    def process_frame(self, img_bgr, flow_features=None, conf=0.001, iou=0.6):
        t_start = time.perf_counter()
        
        # 1. Temporal Security Check
        t_sec_ms = 0.0
        if self.gateway is not None and flow_features is not None:
            if self.frame_count % self.security_interval == 0 or self.last_security_decision is None:
                t0 = time.perf_counter()
                self.last_security_decision = self.gateway.classify(flow_features)
                t_sec_ms = (time.perf_counter() - t0) * 1000
            
            self.tracker["security_ms"].append(t_sec_ms)
            if self.last_security_decision.is_suspicious:
                total_ms = (time.perf_counter() - t_start) * 1000
                self.tracker["total_ms"].append(total_ms)
                self.frame_count += 1
                return {"label": "SUSPICIOUS", "blocked": True, "latency": total_ms}

        # 2. Temporal Condition Check
        t_cond_ms = 0.0
        if self.frame_count % self.condition_interval == 0 or self.last_condition_obj is None:
            t0 = time.perf_counter()
            self.last_condition_obj = analyze_image(img_bgr)
            self.last_condition_label = self.last_condition_obj.label
            t_cond_ms = (time.perf_counter() - t0) * 1000
        self.tracker["condition_ms"].append(t_cond_ms)

        # 3. Selective Enhancement (Bypass if NORMAL)
        t_enh_ms = 0.0
        method_name = "bypass"
        if self.last_condition_label == "NORMAL":
            enhanced = img_bgr
        else:
            t0 = time.perf_counter()
            enhanced, _, method_name = adaptive_enhance(img_bgr)
            t_enh_ms = (time.perf_counter() - t0) * 1000
        self.tracker["enhance_ms"].append(t_enh_ms)

        # 4. YOLO Inference (half precision)
        t0 = time.perf_counter()
        results = self.yolo(enhanced, conf=conf, iou=iou, verbose=False, half=True)
        t_yolo_ms = (time.perf_counter() - t0) * 1000
        self.tracker["yolo_ms"].append(t_yolo_ms)

        total_ms = (time.perf_counter() - t_start) * 1000
        self.tracker["total_ms"].append(total_ms)
        self.frame_count += 1

        return {
            "blocked": False,
            "result": results,
            "condition": self.last_condition_label,
            "method": method_name,
            "latency": total_ms
        }
        
    def summary(self):
        def _ms(lst): return round(float(np.mean(lst)), 2) if lst else 0.0
        def _p95(lst): return round(float(np.percentile(lst, 95)), 2) if lst else 0.0
        avg_total = _ms(self.tracker["total_ms"])
        return {
            "avg_security_ms": _ms(self.tracker["security_ms"]),
            "avg_condition_ms": _ms(self.tracker["condition_ms"]),
            "avg_enhance_ms": _ms(self.tracker["enhance_ms"]),
            "avg_yolo_ms": _ms(self.tracker["yolo_ms"]),
            "avg_total_ms": avg_total,
            "p95_total_ms": _p95(self.tracker["total_ms"]),
            "fps": round(1000.0 / avg_total if avg_total > 0 else 0.0, 1)
        }
