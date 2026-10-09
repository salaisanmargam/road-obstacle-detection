from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.utils.config import load_yaml, project_root
from src.utils.logging_utils import configure_logger
from src.utils.seed import seed_everything


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_label(path: Path, iw: int, ih: int) -> list[tuple[int, int, int, int, int]]:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    rows: list[tuple[int, int, int, int, int]] = []
    if not txt:
        return rows
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c = int(float(p[0]))
        x, y, w, h = map(float, p[1:])
        x1 = int((x - w / 2) * iw)
        y1 = int((y - h / 2) * ih)
        x2 = int((x + w / 2) * iw)
        y2 = int((y + h / 2) * ih)
        rows.append((c, max(0, x1), max(0, y1), min(iw - 1, x2), min(ih - 1, y2)))
    return rows


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(1, (ax2 - ax1) * (ay2 - ay1))
    ub = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (ua + ub - inter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative prediction examples.")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    root = project_root()
    logger = configure_logger("inference_examples", root / "reports" / "phase1" / "inference_examples.log")

    run_info = json.loads((root / "results" / "phase1" / "latest_run.json").read_text(encoding="utf-8"))
    paths_cfg = load_yaml(root / "configs" / "paths.yaml")
    names = yaml.safe_load((root / "data" / "dataset.yaml").read_text(encoding="utf-8")).get("names", [])

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics package is not installed") from e

    model = YOLO(run_info["best_checkpoint"])

    pred_dir = root / "results" / "phase1" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    test_images_dir = Path(paths_cfg["splits"]["test"]["images"])
    test_labels_dir = Path(paths_cfg["splits"]["test"]["labels"])

    image_files = [p for p in sorted(test_images_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    random.shuffle(image_files)
    image_files = image_files[: min(args.samples, len(image_files))]

    stats = {"success": 0, "false_positive": 0, "false_negative": 0, "difficult": 0}

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        label_path = test_labels_dir / f"{img_path.stem}.txt"
        gts = parse_label(label_path, iw, ih) if label_path.exists() else []

        result = model.predict(str(img_path), conf=0.25, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else np.zeros((0, 4), dtype=int)
        pred_cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.zeros((0,), dtype=int)
        pred_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros((0,), dtype=float)

        matched_gt = set()
        matched_pred = set()

        for pi, pb in enumerate(pred_boxes):
            best_iou = 0.0
            best_gi = -1
            for gi, gt in enumerate(gts):
                i = iou(tuple(pb.tolist()), tuple(gt[1:]))
                if i > best_iou:
                    best_iou = i
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0 and pred_cls[pi] == gts[best_gi][0]:
                matched_pred.add(pi)
                matched_gt.add(best_gi)

        fp = len(pred_boxes) - len(matched_pred)
        fn = len(gts) - len(matched_gt)

        category = "success"
        if fn > 0 and fp > 0:
            category = "difficult"
        elif fn > 0:
            category = "false_negative"
        elif fp > 0:
            category = "false_positive"

        stats[category] += 1

        canvas = img.copy()
        for gi, gt in enumerate(gts):
            c, x1, y1, x2, y2 = gt
            color = (0, 220, 0) if gi in matched_gt else (0, 0, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cname = names[c] if c < len(names) else str(c)
            cv2.putText(canvas, f"GT:{cname}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        for pi, pb in enumerate(pred_boxes):
            x1, y1, x2, y2 = pb.tolist()
            color = (255, 170, 0) if pi in matched_pred else (255, 0, 255)
            cname = names[pred_cls[pi]] if pred_cls[pi] < len(names) else str(pred_cls[pi])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                f"PR:{cname} {pred_conf[pi]:.2f}",
                (x1, min(ih - 5, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        cv2.putText(canvas, f"Category: {category}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 255), 2)
        out_name = f"{category}_{img_path.name}"
        cv2.imwrite(str(pred_dir / out_name), canvas)

    (root / "results" / "phase1" / "qualitative_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Saved qualitative predictions and summary: %s", stats)


if __name__ == "__main__":
    main()
