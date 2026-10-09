from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import yaml

from src.utils.config import load_yaml, project_root
from src.utils.logging_utils import configure_logger
from src.utils.seed import seed_everything


def select_training_params(base_cfg: dict, gpu_vram_mb: int | None) -> tuple[int, int]:
    batch = int(base_cfg["training"].get("batch", 8))
    imgsz = int(base_cfg["training"].get("imgsz", 640))

    # Conservative fallback for 4GB VRAM laptops.
    if gpu_vram_mb is not None and gpu_vram_mb <= 4096:
        batch = min(batch, 4)
        imgsz = min(imgsz, 640)
    return batch, imgsz


def next_run_name(project_dir: Path, prefix: str = "run") -> str:
    project_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p.name for p in project_dir.glob(f"{prefix}_*") if p.is_dir()])
    if not existing:
        return f"{prefix}_001"
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[-1]))
        except ValueError:
            continue
    nxt = max(nums) + 1 if nums else 1
    return f"{prefix}_{nxt:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight YOLO baseline")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    root = project_root()
    logger = configure_logger("train_baseline", root / "reports" / "phase1" / "train_baseline.log")

    cfg = load_yaml(root / args.config)
    paths_cfg = load_yaml(root / "configs" / "paths.yaml")
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics package is not installed.") from e

    device_choice = cfg["training"].get("device", "auto")
    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if device_choice == "auto":
        device_choice = "0" if cuda_available else "cpu"

    gpu_vram = paths_cfg.get("environment", {}).get("gpu_vram_mb")
    batch, imgsz = select_training_params(cfg, gpu_vram)
    epochs = int(cfg["training"]["epochs"])
    workers = int(cfg["training"].get("workers", 4))
    fraction = float(cfg["training"].get("fraction", 1.0))

    if device_choice == "cpu":
        batch = int(cfg["training"].get("cpu_fallback_batch", 2))
        imgsz = int(cfg["training"].get("cpu_fallback_imgsz", 512))
        epochs = int(cfg["training"].get("cpu_fallback_epochs", 1))
        workers = min(workers, 2)

    project_dir = root / cfg["training"]["project_dir"]
    run_name = next_run_name(project_dir, cfg["training"].get("run_name_prefix", "run"))

    data_yaml = root / cfg["dataset"]["yolo_dataset_yaml"]
    model_name = cfg["training"]["model"]

    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device_choice,
        optimizer=cfg["training"].get("optimizer", "auto"),
        lr0=float(cfg["training"].get("lr0", 0.01)),
        cos_lr=bool(cfg["training"].get("cos_lr", False)),
        patience=int(cfg["training"].get("patience", 10)),
        amp=bool(cfg["training"].get("amp", True)),
        fraction=fraction,
        seed=seed,
        project=str(project_dir.resolve()),
        name=run_name,
        exist_ok=False,
        pretrained=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    training_cfg_used = {
        "model": model_name,
        "imgsz": imgsz,
        "batch": batch,
        "epochs": epochs,
        "workers": workers,
        "fraction": fraction,
        "optimizer": cfg["training"].get("optimizer", "auto"),
        "lr0": float(cfg["training"].get("lr0", 0.01)),
        "patience": int(cfg["training"].get("patience", 10)),
        "amp": bool(cfg["training"].get("amp", True)),
        "seed": seed,
        "device": device_choice,
        "python_version": platform.python_version(),
    }

    (save_dir / "training_config_used.yaml").write_text(yaml.safe_dump(training_cfg_used, sort_keys=False), encoding="utf-8")
    run_info = {
        "run_name": run_name,
        "save_dir": str(save_dir.resolve()),
        "best_checkpoint": str((save_dir / "weights" / "best.pt").resolve()),
        "last_checkpoint": str((save_dir / "weights" / "last.pt").resolve()),
    }
    (root / "results" / "phase1" / "latest_run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    logger.info("Training complete. Run dir: %s", save_dir)


if __name__ == "__main__":
    main()
