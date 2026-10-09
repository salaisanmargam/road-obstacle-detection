from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.utils.config import load_yaml, project_root
from src.utils.logging_utils import configure_logger


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def model_param_count(model) -> int:
    try:
        return int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        return -1


def gather_test_images(paths_cfg: dict[str, Any], limit: int = 200) -> list[Path]:
    images_dir = Path(paths_cfg["splits"]["test"]["images"])
    imgs: list[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs[:limit]


def estimate_latency(model, test_images: list[Path], imgsz: int) -> tuple[float | None, float | None]:
    if not test_images:
        return None, None

    warmup = test_images[: min(10, len(test_images))]
    for p in warmup:
        _ = model.predict(str(p), imgsz=imgsz, verbose=False)

    start = time.perf_counter()
    count = 0
    for p in test_images:
        _ = model.predict(str(p), imgsz=imgsz, verbose=False)
        count += 1
    elapsed = time.perf_counter() - start
    if elapsed <= 0 or count == 0:
        return None, None
    latency_ms = (elapsed / count) * 1000.0
    fps = count / elapsed
    return latency_ms, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline YOLO model on test split")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    root = project_root()
    logger = configure_logger("evaluate_baseline", root / "reports" / "phase1" / "evaluate_baseline.log")

    cfg = load_yaml(root / args.config)
    run_info = json.loads((root / "results" / "phase1" / "latest_run.json").read_text(encoding="utf-8"))
    paths_cfg = load_yaml(root / "configs" / "paths.yaml")

    best_ckpt = Path(run_info["best_checkpoint"])
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics package is not installed.") from e

    device_choice = cfg["training"].get("device", "auto")
    if device_choice == "auto":
        try:
            import torch

            device_choice = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_choice = "cpu"

    model = YOLO(str(best_ckpt))
    data_yaml = root / cfg["dataset"]["yolo_dataset_yaml"]

    eval_project = root / "experiments" / "phase1_baseline"
    eval_name = f"{Path(run_info['save_dir']).name}_eval_test"

    metrics = model.val(
        data=str(data_yaml.resolve()),
        split="test",
        imgsz=int(cfg["training"]["imgsz"]),
        batch=int(cfg["training"]["batch"]),
        device=device_choice,
        workers=int(cfg["training"].get("workers", 4)),
        project=str(eval_project.resolve()),
        name=eval_name,
        plots=True,
        save_json=True,
        verbose=True,
    )

    names = []
    ds_yaml = yaml.safe_load((root / "data" / "dataset.yaml").read_text(encoding="utf-8"))
    names = ds_yaml.get("names", [])

    mp = safe_float(getattr(metrics.box, "mp", None))
    mr = safe_float(getattr(metrics.box, "mr", None))
    map50 = safe_float(getattr(metrics.box, "map50", None))
    map5095 = safe_float(getattr(metrics.box, "map", None))
    f1 = None
    if mp is not None and mr is not None and (mp + mr) > 0:
        f1 = float(2 * mp * mr / (mp + mr))

    per_class_rows = []
    p_arr = getattr(metrics.box, "p", None)
    r_arr = getattr(metrics.box, "r", None)
    ap50_arr = getattr(metrics.box, "ap50", None)
    maps_arr = getattr(metrics.box, "maps", None)

    for i, cname in enumerate(names):
        p_i = safe_float(p_arr[i]) if p_arr is not None and i < len(p_arr) else None
        r_i = safe_float(r_arr[i]) if r_arr is not None and i < len(r_arr) else None
        ap50_i = safe_float(ap50_arr[i]) if ap50_arr is not None and i < len(ap50_arr) else None
        ap_i = safe_float(maps_arr[i]) if maps_arr is not None and i < len(maps_arr) else None
        f1_i = None
        if p_i is not None and r_i is not None and (p_i + r_i) > 0:
            f1_i = float(2 * p_i * r_i / (p_i + r_i))
        per_class_rows.append(
            {
                "class_id": i,
                "class_name": cname,
                "precision": p_i,
                "recall": r_i,
                "f1": f1_i,
                "ap50": ap50_i,
                "ap50_95": ap_i,
            }
        )

    params = model_param_count(model)
    model_size_mb = os.path.getsize(best_ckpt) / (1024 * 1024)
    test_imgs = gather_test_images(paths_cfg)
    latency_ms, fps = estimate_latency(model, test_imgs, int(cfg["training"]["imgsz"]))

    hw = paths_cfg.get("environment", {})

    out_json = {
        "run": run_info,
        "aggregate_metrics": {
            "precision": mp,
            "recall": mr,
            "f1": f1,
            "map50": map50,
            "map50_95": map5095,
        },
        "per_class_metrics": per_class_rows,
        "efficiency": {
            "params": params,
            "model_size_mb": model_size_mb,
            "inference_latency_ms": latency_ms,
            "fps": fps,
            "hardware": hw,
            "python_version": platform.python_version(),
        },
        "evaluation_artifacts_dir": str((eval_project / eval_name).resolve()),
    }

    metrics_json = root / "results" / "phase1" / "baseline_metrics.json"
    metrics_json.write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # Copy key visual artifacts into the required phase1 figures directory.
    figures_dir = root / "results" / "phase1" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = eval_project / eval_name
    train_dir = Path(run_info["save_dir"])
    candidate_files = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "results.png",
    ]
    for fname in candidate_files:
        for src_dir in [eval_dir, train_dir]:
            src = src_dir / fname
            if src.exists():
                dst = figures_dir / f"{src_dir.name}_{fname}"
                shutil.copy2(src, dst)
                break

    metrics_csv = root / "results" / "phase1" / "baseline_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["precision", mp])
        writer.writerow(["recall", mr])
        writer.writerow(["f1", f1])
        writer.writerow(["map50", map50])
        writer.writerow(["map50_95", map5095])
        writer.writerow(["params", params])
        writer.writerow(["model_size_mb", model_size_mb])
        writer.writerow(["inference_latency_ms", latency_ms])
        writer.writerow(["fps", fps])
        writer.writerow([])
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "ap50", "ap50_95"])
        for row in per_class_rows:
            writer.writerow(
                [
                    row["class_id"],
                    row["class_name"],
                    row["precision"],
                    row["recall"],
                    row["f1"],
                    row["ap50"],
                    row["ap50_95"],
                ]
            )

    logger.info("Evaluation complete. Metrics saved to %s", metrics_json)


if __name__ == "__main__":
    main()
