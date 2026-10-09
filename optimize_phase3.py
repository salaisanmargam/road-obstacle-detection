import time
import json
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path("D:/road obstacle detection/road_damage_research")))
from phase3.integration.optimized_pipeline import OptimizedVideoPipeline
from phase3.security.security_gateway import SecurityGateway

# Paths
RDD_TEST = Path("D:/road obstacle detection/rdd2022/RDD_SPLIT/test/images")
PHASE1_CKPT = Path("D:/road obstacle detection/road_damage_research/experiments/phase1_final_baseline/final/weights/best.pt")
RESULTS_DIR = Path("D:/road obstacle detection/road_damage_research/results/phase3")

def benchmark_system(system_name, img_paths, gateway=None, use_enhancement=False, condition_interval=1, security_interval=1, always_enhance=False):
    print(f"\n--- Benchmarking: {system_name} ---")
    
    # Initialize pipeline
    pipe = OptimizedVideoPipeline(
        yolo_ckpt=str(PHASE1_CKPT), 
        gateway=gateway, 
        condition_interval=condition_interval,
        security_interval=security_interval
    )
    
    # Hack for System B (Enhancement on EVERY frame regardless of condition)
    # If always_enhance is True, we force it in the processing loop by mocking condition
    
    # Dummy flow for security
    n_feat = 77
    if gateway:
        n_feat = len(gateway.feature_names)
    dummy_flow = np.zeros(n_feat, dtype=np.float32)

    # Warmup
    for p in img_paths[:10]:
        img = cv2.imread(str(p))
        if always_enhance: pipe.last_condition_label = "DEGRADED" # force enhancement
        pipe.process_frame(img, dummy_flow if gateway else None)
    
    pipe.tracker = {k: [] for k in pipe.tracker}
    pipe.frame_count = 0
    pipe.last_condition_obj = None
    
    n_enhanced = 0
    t_start = time.perf_counter()
    
    for i, p in enumerate(img_paths):
        img = cv2.imread(str(p))
        if img is None: continue
        
        # Override for System B
        if always_enhance:
            pipe.last_condition_label = "LOW_LIGHT"  # Force a condition that triggers enhancement
            
        # Disable enhancement for System A and C(no enh) if needed.
        # Wait, the pipeline class applies enhancement if use_enhancement is True?
        # The OptimizedVideoPipeline uses enhancement UNLESS label is NORMAL.
        if not use_enhancement:
            pipe.last_condition_label = "NORMAL" # forces bypass
            
        res = pipe.process_frame(img, dummy_flow if gateway else None)
        if res.get("method", "bypass") != "bypass":
            n_enhanced += 1

    total_time = time.perf_counter() - t_start
    summ = pipe.summary()
    summ["system"] = system_name
    summ["n_enhanced"] = n_enhanced
    summ["activation_rate"] = round(n_enhanced / len(img_paths) * 100, 1)
    
    print(f"  FPS: {summ['fps']} | Avg Latency: {summ['avg_total_ms']} ms | P95: {summ['p95_total_ms']} ms | Enh Rate: {summ['activation_rate']}%")
    return summ

def run_benchmarks():
    img_paths = sorted(RDD_TEST.glob("*.jpg"))[:300] # Use 300 for stable timing
    
    print(f"Benchmarking on {len(img_paths)} images...")
    
    try:
        gateway = SecurityGateway.load()
    except:
        gateway = None
        print("Warning: SecurityGateway could not be loaded. Running without security.")

    results = []
    
    # SYSTEM A: YOLO only
    rA = benchmark_system("System A: YOLO only", img_paths, gateway=None, use_enhancement=False)
    results.append(rA)
    
    # SYSTEM B: YOLO + enhancement on every frame (condition=1, force enhance)
    rB = benchmark_system("System B: YOLO + Enh (Always)", img_paths, gateway=None, use_enhancement=True, condition_interval=1, always_enhance=True)
    results.append(rB)
    
    # SYSTEM C: YOLO + adaptive enhancement (condition=1)
    rC = benchmark_system("System C: YOLO + Adaptive Enh (Every frame)", img_paths, gateway=None, use_enhancement=True, condition_interval=1)
    results.append(rC)
    
    # SYSTEM D: YOLO + temporal condition reuse (condition=5) + selective enh
    rD = benchmark_system("System D: YOLO + Temporal Enh (No Sec)", img_paths, gateway=None, use_enhancement=True, condition_interval=5)
    results.append(rD)
    
    # SYSTEM E: SECURITY + temporal adaptive CV pipeline (condition=5, security=30)
    rE = benchmark_system("System E: Security + Temporal Pipeline (Final)", img_paths, gateway=gateway, use_enhancement=True, condition_interval=5, security_interval=30)
    results.append(rE)

    out_path = RESULTS_DIR / "optimized_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nOptimization benchmark saved to {out_path}")

if __name__ == "__main__":
    run_benchmarks()
