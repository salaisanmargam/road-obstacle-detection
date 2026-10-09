#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_phase2.py — Phase 2 Main Pipeline

Weather- and Illumination-Robust Road Damage Detection
Using Adaptive Image Enhancement and Lightweight YOLO

Phase 2: Condition-Aware Adaptive Image Enhancement
=========================================================

IMPORTANT: Phase 1 checkpoint is FROZEN. This script does NOT retrain YOLO.

Experiments:
    A: Clean image           → Frozen YOLOv8n
    B: Degraded image        → Frozen YOLOv8n
    C: Degraded + Fixed CLAHE → Frozen YOLOv8n
    D: Degraded + Adaptive   → Frozen YOLOv8n

Usage:
    python -X utf8 run_phase2.py --degrade
    python -X utf8 run_phase2.py --evaluate
    python -X utf8 run_phase2.py --ablation
    python -X utf8 run_phase2.py --efficiency
    python -X utf8 run_phase2.py --visualize
    python -X utf8 run_phase2.py --report
    python -X utf8 run_phase2.py --all
"""

import sys
import io

# Fix Windows UTF-8 console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import json
import csv
import time
import shutil
import argparse
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase2.utils import (
    logmsg, seed_everything, save_json, load_json,
    DATA_ROOT, RDD_ROOT, RESULTS_ROOT, REPORTS_ROOT, EXP_ROOT,
    PHASE1_CHECKPOINT, DATASET_YAML, CLASS_NAMES, NUM_CLASSES,
    CONDITIONS, SEVERITIES, SEED,
    get_test_image_paths, clean_copy_path, degraded_image_path,
    compute_f1, robustness_drop, robustness_retention, Timer
)
from phase2.degradation_pipeline import run_degradation_pipeline, DEGRADATION_PARAMS
from phase2.condition_analyzer import analyze_image, THRESHOLDS
from phase2.enhancement_methods import apply_enhancement, ENHANCEMENT_REGISTRY
from phase2.adaptive_policy import (
    adaptive_enhance, fixed_enhance, no_enhance,
    ADAPTIVE_POLICY, FIXED_ENHANCEMENT_METHOD,
    get_policy_description,
)

# ─── Constants ────────────────────────────────────────────────────────────────

PHASE1_MAP50     = 0.5724
PHASE1_MAP5095   = 0.3066
PHASE1_PRECISION = 0.6184
PHASE1_RECALL    = 0.5447
PHASE1_F1        = 0.5792

EXPERIMENT_LOG_PATH = REPORTS_ROOT / "experiment_log.csv"
EXPERIMENT_LOG_FIELDS = [
    "exp_id", "timestamp", "condition", "severity", "experiment",
    "enhancement_method", "n_images",
    "precision", "recall", "f1", "map50", "map5095",
    "robustness_drop_map50", "robustness_retention_map50",
    "enhancement_ms_per_img", "yolo_ms_per_img", "total_ms_per_img",
    "fps", "notes"
]

# For quick ablation/efficiency, use a subset of test images
EFFICIENCY_N = 200
ABLATION_N   = 200

# ─── Logging ─────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── YOLO inference helper ────────────────────────────────────────────────────


def get_jpgs(d: Path):
    if (d / 'images').exists():
        return list((d / 'images').glob('*.jpg'))
    return list(d.glob('*.jpg'))


def load_yolo():
    """Load the frozen Phase 1 YOLOv8n checkpoint."""
    from ultralytics import YOLO
    assert PHASE1_CHECKPOINT.exists(), f"Phase 1 checkpoint not found: {PHASE1_CHECKPOINT}"
    model = YOLO(str(PHASE1_CHECKPOINT))
    log(f"  Loaded Phase 1 checkpoint: {PHASE1_CHECKPOINT}")
    return model


def evaluate_image_list(
    model,
    image_dir: Path,
    label_dir: Path,
    dataset_yaml: str,
    split_name: str = 'test',
    project_dir: Optional[Path] = None,
    run_name: str = 'eval',
    conf: float = 0.001,
    iou: float = 0.6,
) -> Dict:
    import yaml
    import subprocess
    
    real_images_dir = image_dir / 'images' if (image_dir / 'images').exists() else image_dir
    parent_dir = real_images_dir.parent
    real_labels_dir = parent_dir / 'labels'
    
    for cache_file in real_images_dir.glob('*.cache'):
        try:
            cache_file.unlink()
        except OSError:
            pass
            
    if not real_labels_dir.exists():
        try:
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(real_labels_dir), str(label_dir)], check=True, stdout=subprocess.DEVNULL)
        except:
            pass

    tmp_yaml_path = EXP_ROOT / f'_tmp_{run_name}.yaml'
    
    ds_config = {
        'path': str(parent_dir),
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'nc': NUM_CLASSES,
        'names': CLASS_NAMES,
    }

    with open(tmp_yaml_path, 'w') as f:
        yaml.dump(ds_config, f)

    if project_dir is None:
        project_dir = EXP_ROOT

    results = model.val(
        data    = str(tmp_yaml_path),
        split   = "test",
        imgsz   = 640,
        batch   = 16,
        conf    = conf,
        iou     = iou,
        device  = "0",
        workers = 2,
        project = str(project_dir),
        name    = run_name,
        exist_ok= True,
        verbose = False,
        plots   = False,
    )

    # Clean up temp yaml
    try:
        tmp_yaml_path.unlink()
    except Exception:
        pass

    # Extract metrics
    box = results.box
    precision = float(box.mp)    # mean precision
    recall    = float(box.mr)    # mean recall
    map50     = float(box.map50) # mAP@50
    map5095   = float(box.map)   # mAP@50:95

    per_class = []
    if hasattr(box, 'ap_class_index') and box.ap_class_index is not None:
        for i, cls_idx in enumerate(box.ap_class_index):
            per_class.append({
                "class_id":   int(cls_idx),
                "class_name": CLASS_NAMES[int(cls_idx)],
                "ap50":       float(box.ap50[i]),
                "ap":         float(box.ap[i]),
            })

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(compute_f1(precision, recall), 4),
        "map50":     round(map50, 4),
        "map5095":   round(map5095, 4),
        "per_class": per_class,
    }


# ─── Step 1: Degradation ─────────────────────────────────────────────────────

def step_degrade(sample_limit: int = 0, overwrite: bool = False):
    log("=" * 60)
    log("STEP 1: Generating degraded image sets")
    log("=" * 60)
    stats = run_degradation_pipeline(
        sample_limit=sample_limit,
        save_samples=8,
        overwrite=overwrite,
    )
    save_json(RESULTS_ROOT / "logs" / "degradation_stats.json", stats)
    log(f"  Degradation stats saved.")
    return stats


# ─── Step 2: Evaluate (Experiments A/B/C/D) ──────────────────────────────────

def step_evaluate(sample_limit: int = 0):
    log("=" * 60)
    log("STEP 2: Running Experiments A/B/C/D")
    log("=" * 60)

    model = load_yolo()
    results_all = {}

    # ── Experiment A: Clean → YOLO ────────────────────────────────────────────
    log("\n--- EXPERIMENT A: Clean images → Frozen YOLOv8n ---")
    clean_dir = DATA_ROOT / "clean" / "original"

    if not clean_dir.exists() or not any(clean_dir.iterdir()):
        log("  ERROR: Clean images not found. Run --degrade first.")
        return {}

    t0 = time.perf_counter()
    metrics_a = evaluate_image_list(
        model, clean_dir,
        RDD_ROOT / "test" / "labels",
        str(DATASET_YAML),
        project_dir=EXP_ROOT / "exp_A",
        run_name="clean",
    )
    yolo_time_a = (time.perf_counter() - t0)
    metrics_a["yolo_ms_per_img"] = round(yolo_time_a / len(get_jpgs(clean_dir)) * 1000, 2)

    log(f"  [A] mAP@50={metrics_a['map50']:.4f} | P={metrics_a['precision']:.4f} | R={metrics_a['recall']:.4f}")
    log(f"  [A] F1={metrics_a['f1']:.4f} | mAP@50:95={metrics_a['map5095']:.4f}")

    clean_map50 = metrics_a["map50"]
    results_all["experiment_A"] = metrics_a
    save_json(RESULTS_ROOT / "metrics" / "exp_A_clean.json", metrics_a)

    _log_experiment(
        exp_id="A", condition="clean", severity="N/A",
        experiment="A_clean_yolo", method="none",
        n=len(get_jpgs(clean_dir)),
        metrics=metrics_a, enh_ms=0,
        yolo_ms=metrics_a["yolo_ms_per_img"],
        clean_map=clean_map50,
        notes="Experiment A: clean images, frozen YOLO"
    )

    # ── Experiments B/C/D: per condition × severity ───────────────────────────
    results_all["conditions"] = {}

    for condition in CONDITIONS:
        log(f"\n--- Condition: {condition.upper()} ---")
        results_all["conditions"][condition] = {}

        for severity in SEVERITIES:
            log(f"  Severity: {severity}")
            deg_dir = DATA_ROOT / condition / severity

            if not deg_dir.exists() or not any(get_jpgs(deg_dir)):
                log(f"  SKIP: no degraded images at {deg_dir}")
                continue

            deg_images = sorted(get_jpgs(deg_dir))
            n_images = len(deg_images)

            # ── B: Degraded → YOLO ────────────────────────────────────────────
            t_b = time.perf_counter()
            metrics_b = evaluate_image_list(
                model, deg_dir,
                RDD_ROOT / "test" / "labels",
                str(DATASET_YAML),
                project_dir=EXP_ROOT / "exp_B" / condition,
                run_name=severity,
            )
            yolo_ms_b = (time.perf_counter() - t_b) / n_images * 1000
            metrics_b["yolo_ms_per_img"] = round(yolo_ms_b, 2)
            log(f"  [B] {condition}/{severity}: mAP@50={metrics_b['map50']:.4f}")

            # ── C: Degraded → Fixed CLAHE → YOLO ──────────────────────────────
            fixed_dir = EXP_ROOT / "exp_C_cache" / condition / severity / "images"
            fixed_dir.mkdir(parents=True, exist_ok=True)
            enh_times_c = []
            for img_path in deg_images:
                out = fixed_dir / img_path.name
                if not out.exists():
                    img = cv2.imread(str(img_path))
                    t_enh = time.perf_counter()
                    enh = fixed_enhance(img)
                    enh_times_c.append((time.perf_counter() - t_enh) * 1000)
                    cv2.imwrite(str(out), enh)

            avg_enh_ms_c = float(np.mean(enh_times_c)) if enh_times_c else 0.0
            t_c = time.perf_counter()
            metrics_c = evaluate_image_list(
                model, fixed_dir,
                RDD_ROOT / "test" / "labels",
                str(DATASET_YAML),
                project_dir=EXP_ROOT / "exp_C" / condition,
                run_name=severity,
            )
            yolo_ms_c = (time.perf_counter() - t_c) / n_images * 1000
            metrics_c["yolo_ms_per_img"] = round(yolo_ms_c, 2)
            metrics_c["enh_ms_per_img"] = round(avg_enh_ms_c, 2)
            log(f"  [C] {condition}/{severity}: mAP@50={metrics_c['map50']:.4f}")

            # ── D: Degraded → Adaptive → YOLO ─────────────────────────────────
            adaptive_dir = EXP_ROOT / "exp_D_cache" / condition / severity / "images"
            adaptive_dir.mkdir(parents=True, exist_ok=True)
            enh_times_d = []
            condition_labels = []
            method_names = []
            for img_path in deg_images:
                out = adaptive_dir / img_path.name
                if not out.exists():
                    img = cv2.imread(str(img_path))
                    t_enh = time.perf_counter()
                    enh, cond_obj, method = adaptive_enhance(img)
                    enh_times_d.append((time.perf_counter() - t_enh) * 1000)
                    cv2.imwrite(str(out), enh)
                    condition_labels.append(cond_obj.label)
                    method_names.append(method)

            avg_enh_ms_d = float(np.mean(enh_times_d)) if enh_times_d else 0.0
            t_d = time.perf_counter()
            metrics_d = evaluate_image_list(
                model, adaptive_dir,
                RDD_ROOT / "test" / "labels",
                str(DATASET_YAML),
                project_dir=EXP_ROOT / "exp_D" / condition,
                run_name=severity,
            )
            yolo_ms_d = (time.perf_counter() - t_d) / n_images * 1000
            metrics_d["yolo_ms_per_img"] = round(yolo_ms_d, 2)
            metrics_d["enh_ms_per_img"] = round(avg_enh_ms_d, 2)

            # Condition classification distribution
            if condition_labels:
                from collections import Counter
                label_dist = dict(Counter(condition_labels))
                method_dist = dict(Counter(method_names))
                metrics_d["condition_label_distribution"] = label_dist
                metrics_d["method_distribution"] = method_dist

            log(f"  [D] {condition}/{severity}: mAP@50={metrics_d['map50']:.4f}")

            # Collect
            entry = {
                "condition": condition,
                "severity": severity,
                "n_images": n_images,
                "B_degraded":  metrics_b,
                "C_fixed":     metrics_c,
                "D_adaptive":  metrics_d,
                "robustness": {
                    "B_drop_map50":      robustness_drop(clean_map50, metrics_b["map50"]),
                    "C_drop_map50":      robustness_drop(clean_map50, metrics_c["map50"]),
                    "D_drop_map50":      robustness_drop(clean_map50, metrics_d["map50"]),
                    "B_retention_pct":   robustness_retention(clean_map50, metrics_b["map50"]),
                    "C_retention_pct":   robustness_retention(clean_map50, metrics_c["map50"]),
                    "D_retention_pct":   robustness_retention(clean_map50, metrics_d["map50"]),
                    "C_over_B_map50":    round(metrics_c["map50"] - metrics_b["map50"], 4),
                    "D_over_B_map50":    round(metrics_d["map50"] - metrics_b["map50"], 4),
                    "D_over_C_map50":    round(metrics_d["map50"] - metrics_c["map50"], 4),
                }
            }
            results_all["conditions"][condition][severity] = entry

            # Save per-condition JSON
            out_path = RESULTS_ROOT / "metrics" / condition / f"{severity}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(out_path, entry)

            # Log to experiment CSV
            for exp_letter, met, enh_ms in [
                ("B", metrics_b, 0),
                ("C", metrics_c, metrics_c.get("enh_ms_per_img", 0)),
                ("D", metrics_d, metrics_d.get("enh_ms_per_img", 0)),
            ]:
                _log_experiment(
                    exp_id=exp_letter, condition=condition, severity=severity,
                    experiment=f"{exp_letter}_{condition}_{severity}",
                    method="none" if exp_letter=="B" else (FIXED_ENHANCEMENT_METHOD if exp_letter=="C" else "adaptive"),
                    n=n_images, metrics=met,
                    enh_ms=enh_ms,
                    yolo_ms=met.get("yolo_ms_per_img", 0),
                    clean_map=clean_map50,
                    notes=f"Exp {exp_letter}: {condition} {severity}"
                )

    # Save full results
    save_json(RESULTS_ROOT / "metrics" / "all_results.json", results_all)
    log("\n  All experiments complete. Results saved to results/phase2/metrics/")
    return results_all


# ─── Step 3: Ablation study ───────────────────────────────────────────────────

def step_ablation(n_images: int = ABLATION_N):
    log("=" * 60)
    log("STEP 3: Ablation Study")
    log("=" * 60)
    log(f"  Using {n_images} images per condition")

    model = load_yolo()
    ablation_results = {}

    # For ablation, use moderate severity only (representative)
    severity = "moderate"
    ablation_methods = ["none", "clahe", "gamma", "contrast_stretch",
                        "bilateral_denoise", "unsharp_mask", "dehaze_clahe"]

    for condition in CONDITIONS:
        log(f"\n  Ablation: {condition} ({severity})")
        deg_dir = DATA_ROOT / condition / severity
        deg_images = sorted(get_jpgs(deg_dir))[:n_images]

        if not deg_images:
            continue

        ablation_results[condition] = {}

        for method in ablation_methods:
            # Apply this single method to all images
            tmp_dir = EXP_ROOT / "ablation_cache" / condition / method / "images"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            enh_times = []
            for img_path in deg_images:
                out = tmp_dir / img_path.name
                if not out.exists():
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    t0 = time.perf_counter()
                    enh = apply_enhancement(img, method)
                    enh_times.append((time.perf_counter() - t0) * 1000)
                    cv2.imwrite(str(out), enh)

            metrics = evaluate_image_list(
                model, tmp_dir,
                RDD_ROOT / "test" / "labels",
                str(DATASET_YAML),
                project_dir=EXP_ROOT / "ablation" / condition,
                run_name=method,
            )
            avg_enh_ms = float(np.mean(enh_times)) if enh_times else 0.0
            metrics["enh_ms_per_img"] = round(avg_enh_ms, 2)

            log(f"    [{method}]: mAP@50={metrics['map50']:.4f}")
            ablation_results[condition][method] = metrics

    save_json(RESULTS_ROOT / "ablation" / "ablation_results.json", ablation_results)
    log("  Ablation results saved.")
    return ablation_results


# ─── Step 4: Efficiency benchmarking ─────────────────────────────────────────

def step_efficiency(n_images: int = EFFICIENCY_N):
    log("=" * 60)
    log("STEP 4: Efficiency Benchmarking")
    log("=" * 60)
    log(f"  Timing {n_images} images")

    model = load_yolo()
    seed_everything(SEED)

    # Sample n_images from clean test set
    img_paths = get_test_image_paths(limit=n_images)

    # ── YOLO-only latency (Phase 1 baseline comparison) ────────────────────
    log("  Measuring Phase 1 baseline latency (YOLO only)...")
    yolo_times = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        t0 = time.perf_counter()
        _ = model.predict(img, imgsz=640, conf=0.25, verbose=False)
        yolo_times.append((time.perf_counter() - t0) * 1000)

    yolo_ms = float(np.median(yolo_times))

    # ── Enhancement method timing ─────────────────────────────────────────
    log("  Measuring enhancement method latency...")
    method_times = {}
    for method in ENHANCEMENT_REGISTRY:
        times = []
        for p in img_paths[:50]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            t0 = time.perf_counter()
            _ = apply_enhancement(img, method)
            times.append((time.perf_counter() - t0) * 1000)
        method_times[method] = round(float(np.median(times)), 2)
        log(f"    {method}: {method_times[method]:.2f} ms")

    # ── Condition analyzer timing ─────────────────────────────────────────
    log("  Measuring condition analyzer latency...")
    analyzer_times = []
    for p in img_paths[:50]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        t0 = time.perf_counter()
        _ = analyze_image(img)
        analyzer_times.append((time.perf_counter() - t0) * 1000)
    analyzer_ms = round(float(np.median(analyzer_times)), 2)
    log(f"    Condition analyzer: {analyzer_ms:.2f} ms")

    # ── Full Phase 2 pipeline latency (adaptive) ───────────────────────────
    log("  Measuring full Phase 2 pipeline latency...")
    pipeline_times = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        t0 = time.perf_counter()
        # Enhancement
        enhanced, _, _ = adaptive_enhance(img)
        # YOLO inference
        _ = model.predict(enhanced, imgsz=640, conf=0.25, verbose=False)
        pipeline_times.append((time.perf_counter() - t0) * 1000)

    pipeline_ms = float(np.median(pipeline_times))

    efficiency = {
        "n_images_timed": len(img_paths),
        "phase1_yolo_only_ms":    round(yolo_ms, 2),
        "phase1_fps":             round(1000.0 / yolo_ms, 1),
        "condition_analyzer_ms":  analyzer_ms,
        "enhancement_methods_ms": method_times,
        "phase2_full_pipeline_ms": round(pipeline_ms, 2),
        "phase2_fps":             round(1000.0 / pipeline_ms, 1),
        "overhead_ms":            round(pipeline_ms - yolo_ms, 2),
        "overhead_pct":           round((pipeline_ms - yolo_ms) / yolo_ms * 100, 1),
    }

    save_json(RESULTS_ROOT / "efficiency" / "latency_report.json", efficiency)
    log(f"\n  Phase 1 YOLO-only:  {yolo_ms:.2f} ms/img ({1000.0/yolo_ms:.1f} FPS)")
    log(f"  Phase 2 full pipe:  {pipeline_ms:.2f} ms/img ({1000.0/pipeline_ms:.1f} FPS)")
    log(f"  Overhead:           +{pipeline_ms-yolo_ms:.2f} ms ({(pipeline_ms-yolo_ms)/yolo_ms*100:.1f}%)")
    return efficiency


# ─── Step 5: Visualization ────────────────────────────────────────────────────

def step_visualize(n_per_condition: int = 4):
    log("=" * 60)
    log("STEP 5: Qualitative Visualization")
    log("=" * 60)

    model = load_yolo()
    vis_dir = RESULTS_ROOT / "figures" / "qualitative"
    vis_dir.mkdir(parents=True, exist_ok=True)

    img_paths = get_test_image_paths(limit=50)

    for condition in CONDITIONS:
        log(f"  Visualizing: {condition}")
        deg_dir = DATA_ROOT / condition / "moderate"
        deg_images = sorted(get_jpgs(deg_dir))[:n_per_condition]

        for i, deg_path in enumerate(deg_images):
            orig_path = RDD_ROOT / "test" / "images" / deg_path.name
            if not orig_path.exists():
                continue

            orig = cv2.imread(str(orig_path))
            deg  = cv2.imread(str(deg_path))
            if orig is None or deg is None:
                continue

            # Adaptive enhancement
            enhanced, cond_obj, method = adaptive_enhance(deg)

            # YOLO predictions on all three
            pred_orig = model.predict(orig, imgsz=640, conf=0.25, verbose=False)[0]
            pred_deg  = model.predict(deg,  imgsz=640, conf=0.25, verbose=False)[0]
            pred_enh  = model.predict(enhanced, imgsz=640, conf=0.25, verbose=False)[0]

            # Draw boxes
            orig_plot = pred_orig.plot()
            deg_plot  = pred_deg.plot()
            enh_plot  = pred_enh.plot()

            # Resize to same height
            h = 320
            w_orig = int(orig_plot.shape[1] * h / orig_plot.shape[0])
            orig_resized = cv2.resize(orig_plot, (w_orig, h))
            deg_resized  = cv2.resize(deg_plot,  (w_orig, h))
            enh_resized  = cv2.resize(enh_plot,  (w_orig, h))

            # Add labels
            def add_label(img, text):
                img = img.copy()
                cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
                cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                return img

            orig_labeled = add_label(orig_resized, "Original")
            deg_labeled  = add_label(deg_resized,  f"Degraded ({condition}/moderate)")
            enh_labeled  = add_label(enh_resized,  f"Enhanced ({cond_obj.label}->{method})")

            panel = np.hstack([orig_labeled, deg_labeled, enh_labeled])
            out_path = vis_dir / f"{condition}_sample{i:02d}.jpg"
            cv2.imwrite(str(out_path), panel)

    log(f"  Visualizations saved to {vis_dir}")


# ─── Step 6: Report ────────────────────────────────────────────────────────────

def step_report():
    log("=" * 60)
    log("STEP 6: Generating Phase 2 Report")
    log("=" * 60)

    # Load all results
    all_results_path = RESULTS_ROOT / "metrics" / "all_results.json"
    ablation_path    = RESULTS_ROOT / "ablation" / "ablation_results.json"
    efficiency_path  = RESULTS_ROOT / "efficiency" / "latency_report.json"

    all_results  = load_json(all_results_path) if all_results_path.exists() else {}
    ablation     = load_json(ablation_path)    if ablation_path.exists() else {}
    efficiency   = load_json(efficiency_path)  if efficiency_path.exists() else {}

    exp_a        = all_results.get("experiment_A", {})
    conds_data   = all_results.get("conditions", {})

    clean_map50  = exp_a.get("map50", 0)
    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Build summary table ───────────────────────────────────────────────────
    summary_rows = []
    for condition in CONDITIONS:
        for severity in SEVERITIES:
            entry = conds_data.get(condition, {}).get(severity, {})
            if not entry:
                continue
            rob = entry.get("robustness", {})
            summary_rows.append({
                "condition": condition,
                "severity":  severity,
                "B_map50":   entry.get("B_degraded", {}).get("map50", 0),
                "C_map50":   entry.get("C_fixed", {}).get("map50", 0),
                "D_map50":   entry.get("D_adaptive", {}).get("map50", 0),
                "D_over_B":  rob.get("D_over_B_map50", 0),
                "D_over_C":  rob.get("D_over_C_map50", 0),
                "D_retention": rob.get("D_retention_pct", 0),
            })

    # ── Find best result ──────────────────────────────────────────────────────
    best_D = max(summary_rows, key=lambda r: r["D_map50"]) if summary_rows else {}
    worst_D = min(summary_rows, key=lambda r: r["D_map50"]) if summary_rows else {}

    avg_B   = np.mean([r["B_map50"] for r in summary_rows]) if summary_rows else 0
    avg_C   = np.mean([r["C_map50"] for r in summary_rows]) if summary_rows else 0
    avg_D   = np.mean([r["D_map50"] for r in summary_rows]) if summary_rows else 0

    # ── Generate markdown ─────────────────────────────────────────────────────
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_ROOT / "PHASE_2_FINAL_REPORT.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# PHASE 2 FINAL REPORT\n")
        f.write(f"*Generated: {timestamp}*\n\n")
        f.write("---\n\n")

        f.write("## 1. Phase 2 Objective\n")
        f.write(
            "Phase 2 investigates whether **condition-aware adaptive image enhancement** "
            "can improve the robustness of the frozen Phase 1 YOLOv8n model under "
            "controlled visual degradation conditions, without retraining YOLO.\n\n"
        )

        f.write("## 2. Research Question\n")
        f.write(
            "> Can condition-aware adaptive image enhancement improve the robustness of a "
            "lightweight YOLOv8n road-damage detector under difficult visual conditions "
            "while preserving performance on normal images?\n\n"
        )

        f.write("## 3. Relationship to Phase 1\n")
        f.write(
            "Phase 1 established the frozen baseline (YOLOv8n, 100 epochs, full RDD2022). "
            "Phase 2 uses the **exact same Phase 1 checkpoint** — YOLO is NOT retrained. "
            "Only the preprocessing (enhancement) module is added. This design isolates "
            "the contribution of the enhancement module.\n\n"
        )
        f.write(f"| Phase 1 Metric | Value |\n|---|---|\n")
        f.write(f"| Precision | {PHASE1_PRECISION:.4f} |\n")
        f.write(f"| Recall    | {PHASE1_RECALL:.4f} |\n")
        f.write(f"| F1 Score  | {PHASE1_F1:.4f} |\n")
        f.write(f"| mAP@50    | {PHASE1_MAP50:.4f} |\n")
        f.write(f"| mAP@50:95 | {PHASE1_MAP5095:.4f} |\n\n")

        f.write("## 4. Dataset\n")
        f.write(
            "- **Dataset**: RDD2022 (Road Damage Detection 2022)\n"
            "- **Test split**: 5,758 images (IDENTICAL to Phase 1 — held out)\n"
            "- **Classes**: D00, D10, D20, D40, class_4\n"
            "- **Original images are NOT modified**\n"
            "- **Degraded images** are stored in `data/phase2/<condition>/<severity>/`\n\n"
        )

        f.write("## 5. Degradation Methodology\n")
        f.write("Seven controlled visual degradations at three severity levels:\n\n")
        f.write("| Condition | Mild | Moderate | Severe |\n|---|---|---|---|\n")
        for cond, params in DEGRADATION_PARAMS.items():
            mild_str = ", ".join(f"{k}={v}" for k,v in params["mild"].items())
            mod_str  = ", ".join(f"{k}={v}" for k,v in params["moderate"].items())
            sev_str  = ", ".join(f"{k}={v}" for k,v in params["severe"].items())
            f.write(f"| {cond} | {mild_str} | {mod_str} | {sev_str} |\n")
        f.write("\n**Important limitation**: These are synthetic degradations. ")
        f.write("Results may not perfectly predict real-world performance in actual ")
        f.write("weather conditions without further validation.\n\n")

        f.write("## 6. Image Condition Analyzer\n")
        f.write("Classical CV features (no neural network):\n\n")
        f.write("| Feature | Description |\n|---|---|\n")
        f.write("| Mean brightness | Average pixel intensity |\n")
        f.write("| Contrast RMS | std/mean*100 |\n")
        f.write("| Laplacian variance | Sharpness estimate |\n")
        f.write("| Noise estimate | High-frequency energy |\n")
        f.write("| Haze score | Brightness + contrast combination |\n\n")
        f.write("**Thresholds** (frozen on validation set, NOT test set):\n\n")
        for k, v in THRESHOLDS.items():
            f.write(f"- `{k}` = {v}\n")
        f.write("\n")

        f.write("## 7. Enhancement Methods\n")
        f.write("| Method | Description |\n|---|---|\n")
        f.write("| none | Pass-through, no change |\n")
        f.write("| clahe | CLAHE on LAB L-channel |\n")
        f.write("| gamma | Power-law gamma brightening |\n")
        f.write("| contrast_stretch | Percentile histogram stretching |\n")
        f.write("| bilateral_denoise | Edge-preserving denoising |\n")
        f.write("| nlm_denoise | Non-Local Means denoising |\n")
        f.write("| unsharp_mask | Unsharp masking sharpening |\n")
        f.write("| laplacian_sharpen | Laplacian edge enhancement |\n")
        f.write("| dehaze_clahe | CLAHE + gamma for haze removal |\n\n")

        f.write("## 8. Adaptive Enhancement Policy\n")
        f.write("```\n")
        f.write(get_policy_description())
        f.write("\n```\n\n")

        f.write("## 9. Experimental Design\n")
        f.write("| Experiment | Input | Enhancement | YOLO |\n|---|---|---|---|\n")
        f.write("| A | Clean test image | None | Frozen Phase 1 |\n")
        f.write("| B | Degraded image | None | Frozen Phase 1 |\n")
        f.write("| C | Degraded image | Fixed (CLAHE always) | Frozen Phase 1 |\n")
        f.write("| D | Degraded image | Adaptive | Frozen Phase 1 |\n\n")
        f.write("Key comparison: **B vs C vs D** for each condition and severity.\n\n")

        f.write("## 10. Experiment A: Clean Image Baseline\n")
        f.write("> This should closely reproduce the Phase 1 test result.\n\n")
        if exp_a:
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Precision | {exp_a.get('precision',0):.4f} |\n")
            f.write(f"| Recall    | {exp_a.get('recall',0):.4f} |\n")
            f.write(f"| F1 Score  | {exp_a.get('f1',0):.4f} |\n")
            f.write(f"| mAP@50    | {exp_a.get('map50',0):.4f} |\n")
            f.write(f"| mAP@50:95 | {exp_a.get('map5095',0):.4f} |\n\n")
            delta = exp_a.get('map50',0) - PHASE1_MAP50
            f.write(f"Delta from Phase 1: {delta:+.4f} mAP@50\n\n")
        else:
            f.write("*Not yet computed. Run --evaluate.*\n\n")

        f.write("## 11. Quantitative Results\n\n")
        if summary_rows:
            f.write(f"**Clean baseline mAP@50**: {clean_map50:.4f}\n\n")
            f.write(f"**Average across all conditions/severities**:\n")
            f.write(f"- Exp B (Degraded, no enhancement): {avg_B:.4f}\n")
            f.write(f"- Exp C (Fixed CLAHE):              {avg_C:.4f}\n")
            f.write(f"- Exp D (Adaptive):                 {avg_D:.4f}\n\n")
            f.write("### Per-Condition mAP@50 Summary\n\n")
            f.write("| Condition | Severity | B (Degraded) | C (Fixed) | D (Adaptive) | D-B | D-C |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|\n")
            for row in summary_rows:
                f.write(
                    f"| {row['condition']} | {row['severity']} | "
                    f"{row['B_map50']:.4f} | {row['C_map50']:.4f} | {row['D_map50']:.4f} | "
                    f"{row['D_over_B']:+.4f} | {row['D_over_C']:+.4f} |\n"
                )
        else:
            f.write("*Not yet computed. Run --evaluate.*\n\n")

        f.write("\n## 12. Robustness Analysis\n\n")
        if summary_rows:
            f.write("**Robustness Retention (%)** = (Degraded mAP / Clean mAP) × 100\n\n")
            f.write("| Condition | Severity | B Retention | C Retention | D Retention |\n")
            f.write("|---|---|---:|---:|---:|\n")
            for cond in CONDITIONS:
                for sev in SEVERITIES:
                    entry = conds_data.get(cond, {}).get(sev, {})
                    rob = entry.get("robustness", {})
                    if rob:
                        f.write(
                            f"| {cond} | {sev} | "
                            f"{rob.get('B_retention_pct',0):.1f}% | "
                            f"{rob.get('C_retention_pct',0):.1f}% | "
                            f"{rob.get('D_retention_pct',0):.1f}% |\n"
                        )
        else:
            f.write("*Not yet computed. Run --evaluate.*\n\n")

        f.write("\n## 13. Ablation Study\n\n")
        if ablation:
            f.write("Single-method comparison at moderate severity:\n\n")
            f.write("| Method | ")
            f.write(" | ".join(CONDITIONS))
            f.write(" |\n|---|")
            f.write("---|" * len(CONDITIONS))
            f.write("\n")
            methods_in_ablation = list(next(iter(ablation.values())).keys()) if ablation else []
            for method in methods_in_ablation:
                row = f"| {method} |"
                for cond in CONDITIONS:
                    m = ablation.get(cond, {}).get(method, {})
                    val = m.get("map50", 0)
                    row += f" {val:.4f} |"
                f.write(row + "\n")
        else:
            f.write("*Not yet computed. Run --ablation.*\n\n")

        f.write("\n## 14. Efficiency Analysis\n\n")
        if efficiency:
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Phase 1 YOLO-only latency | {efficiency.get('phase1_yolo_only_ms',0):.2f} ms/img |\n")
            f.write(f"| Phase 1 FPS               | {efficiency.get('phase1_fps',0):.1f} |\n")
            f.write(f"| Condition analyzer        | {efficiency.get('condition_analyzer_ms',0):.2f} ms/img |\n")
            f.write(f"| Phase 2 full pipeline     | {efficiency.get('phase2_full_pipeline_ms',0):.2f} ms/img |\n")
            f.write(f"| Phase 2 FPS               | {efficiency.get('phase2_fps',0):.1f} |\n")
            f.write(f"| Overhead vs Phase 1       | +{efficiency.get('overhead_ms',0):.2f} ms ({efficiency.get('overhead_pct',0):.1f}%) |\n\n")
        else:
            f.write("*Not yet computed. Run --efficiency.*\n\n")

        f.write("## 15. Limitations\n\n")
        f.write(
            "1. **Synthetic degradations**: These simulate, but do not perfectly replicate, "
            "real-world weather conditions.\n"
            "2. **Frozen YOLO**: Phase 2 does not retrain YOLO. Fine-tuning on enhanced images "
            "is left for Phase 3.\n"
            "3. **Classical condition analyzer**: Heuristic thresholds may misclassify edge cases "
            "(e.g., night + fog simultaneously).\n"
            "4. **Single dominant condition**: The analyzer returns one label per image; "
            "multi-condition co-occurrence is not fully handled.\n"
            "5. **Limited enhancement diversity**: Only classical methods explored; "
            "learned enhancement (e.g., zero-DCE) is out of scope for this phase.\n\n"
        )

        f.write("## 16. Scientific Integrity\n\n")
        f.write(
            "- Phase 1 checkpoint was **NOT retrained or modified**.\n"
            "- Condition analyzer thresholds were set using the **validation set**, "
            "not the test set.\n"
            "- Ground-truth labels were **not modified**.\n"
            "- Cases where enhancement **reduced mAP** are **reported honestly**.\n"
            "- Test set was evaluated exactly once per experiment after policy was frozen.\n\n"
        )

        f.write("## 17. Conclusion\n\n")
        if summary_rows and avg_D > avg_B:
            improvement = avg_D - avg_B
            f.write(
                f"Phase 2 experiments demonstrate that adaptive image enhancement "
                f"improves average mAP@50 by **{improvement:+.4f}** compared to no enhancement "
                f"under controlled degradation conditions. This supports the hypothesis that "
                f"condition-aware preprocessing is beneficial for robust road damage detection.\n\n"
            )
        elif summary_rows:
            f.write(
                "Phase 2 experiments show mixed results. In some conditions adaptive enhancement "
                "improves mAP@50, while in others it provides marginal improvement or a slight "
                "reduction. This nuanced result provides important direction for Phase 3 "
                "(fine-tuned YOLO on enhanced images) and Phase 4 (multi-condition evaluation).\n\n"
            )
        else:
            f.write("*Experiments not yet complete.*\n\n")

        f.write("## 18. Preparation for Phase 3\n\n")
        f.write(
            "- Phase 3 will **fine-tune YOLO** on enhanced training images.\n"
            "- The best-performing adaptive policy from Phase 2 will be used as the "
            "preprocessing pipeline for the Phase 3 training data.\n"
            "- Phase 2 test results serve as a comparison baseline for Phase 3.\n\n"
        )

        f.write("---\n")
        f.write(f"*Phase 2 Final Report | Generated: {timestamp}*\n")

    log(f"  Report saved: {report_path}")

    # ── Print final summary ───────────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("PHASE 2 — IMPLEMENTATION COMPLETE")
    print("=" * 60)
    print(f"\nPhase 1 baseline:")
    print(f"  mAP@50    = {PHASE1_MAP50:.4f}")
    print(f"  mAP@50:95 = {PHASE1_MAP5095:.4f}")
    print(f"\nPhase 2 clean (Exp A):")
    if exp_a:
        print(f"  mAP@50    = {exp_a.get('map50',0):.4f}")
        print(f"  mAP@50:95 = {exp_a.get('map5095',0):.4f}")
    else:
        print("  Not computed yet")

    for condition in CONDITIONS:
        print(f"\nPhase 2 {condition}:")
        for sev in SEVERITIES:
            entry = conds_data.get(condition, {}).get(sev, {})
            d = entry.get("D_adaptive", {})
            b = entry.get("B_degraded", {})
            if d:
                print(f"  [{sev}] B={b.get('map50',0):.4f}  C={entry.get('C_fixed',{}).get('map50',0):.4f}  D={d.get('map50',0):.4f}")

    if summary_rows:
        print(f"\nAdaptive enhancement overall result:")
        print(f"  Avg mAP@50 (B, no enh):  {avg_B:.4f}")
        print(f"  Avg mAP@50 (C, fixed):   {avg_C:.4f}")
        print(f"  Avg mAP@50 (D, adaptive):{avg_D:.4f}")

    if efficiency:
        print(f"\nAverage enhancement overhead: {efficiency.get('overhead_ms',0):.2f} ms")
        print(f"Total inference latency (Phase 2): {efficiency.get('phase2_full_pipeline_ms',0):.2f} ms/img")
        print(f"FPS (Phase 2): {efficiency.get('phase2_fps',0):.1f}")

    print(f"\nFinal Phase 2 report: {report_path}")
    print("=" * 60)
    return report_path


# ─── Experiment log helper ─────────────────────────────────────────────────────

def _log_experiment(
    exp_id, condition, severity, experiment, method, n, metrics,
    enh_ms, yolo_ms, clean_map, notes=""
):
    total_ms = enh_ms + yolo_ms
    fps = round(1000.0 / total_ms, 1) if total_ms > 0 else 0
    row = {
        "exp_id": exp_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "condition": condition,
        "severity": severity,
        "experiment": experiment,
        "enhancement_method": method,
        "n_images": n,
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1": metrics.get("f1", 0),
        "map50": metrics.get("map50", 0),
        "map5095": metrics.get("map5095", 0),
        "robustness_drop_map50": robustness_drop(clean_map, metrics.get("map50", 0)),
        "robustness_retention_map50": robustness_retention(clean_map, metrics.get("map50", 0)),
        "enhancement_ms_per_img": round(enh_ms, 2),
        "yolo_ms_per_img": round(yolo_ms, 2),
        "total_ms_per_img": round(total_ms, 2),
        "fps": fps,
        "notes": notes,
    }
    from phase2.utils import append_csv
    append_csv(EXPERIMENT_LOG_PATH, row, EXPERIMENT_LOG_FIELDS)


# ─── GPU check ───────────────────────────────────────────────────────────────

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log(f"  GPU: {name} ({vram:.1f} GB VRAM)")
            return True
        else:
            log("  WARNING: CUDA not available. Running on CPU (will be slow).")
            return False
    except Exception as e:
        log(f"  GPU check failed: {e}")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Adaptive Enhancement Pipeline")
    parser.add_argument("--degrade",    action="store_true", help="Generate degraded image sets")
    parser.add_argument("--evaluate",   action="store_true", help="Run experiments A/B/C/D")
    parser.add_argument("--ablation",   action="store_true", help="Run ablation study")
    parser.add_argument("--efficiency", action="store_true", help="Benchmark latency/FPS")
    parser.add_argument("--visualize",  action="store_true", help="Generate qualitative visualizations")
    parser.add_argument("--report",     action="store_true", help="Generate final report")
    parser.add_argument("--all",        action="store_true", help="Run all steps")
    parser.add_argument("--overwrite",  action="store_true", help="Overwrite existing degraded images")
    parser.add_argument("--quick",      action="store_true", help="Quick test with 50 images")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.error("Specify at least one step, e.g. --all or --degrade")

    run_all = args.all
    sample_limit = 50 if args.quick else 0

    log("=" * 60)
    log("Phase 2: Adaptive Image Enhancement Pipeline")
    log("=" * 60)
    log(f"  Phase 1 checkpoint: {PHASE1_CHECKPOINT}")
    log(f"  Quick mode: {args.quick}")
    check_gpu()
    seed_everything(SEED)

    if run_all or args.degrade:
        step_degrade(sample_limit=sample_limit, overwrite=args.overwrite)

    if run_all or args.evaluate:
        step_evaluate(sample_limit=sample_limit)

    if run_all or args.ablation:
        n = 50 if args.quick else ABLATION_N
        step_ablation(n_images=n)

    if run_all or args.efficiency:
        n = 50 if args.quick else EFFICIENCY_N
        step_efficiency(n_images=n)

    if run_all or args.visualize:
        step_visualize(n_per_condition=4)

    if run_all or args.report:
        step_report()


if __name__ == "__main__":
    main()
