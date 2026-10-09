from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import load_yaml, project_root, resolve_from_root
from src.utils.logging_utils import configure_logger
from src.utils.seed import seed_everything


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return rows
    for ln in txt.splitlines():
        if not ln.strip():
            continue
        p = ln.split()
        if len(p) != 5:
            raise ValueError(f"invalid token count in {path.name}: {ln}")
        c = int(float(p[0]))
        x, y, w, h = map(float, p[1:])
        if c < 0:
            raise ValueError(f"negative class id in {path.name}: {ln}")
        if any(v < 0 or v > 1 for v in [x, y, w, h]):
            raise ValueError(f"out-of-range bbox in {path.name}: {ln}")
        if w <= 0 or h <= 0:
            raise ValueError(f"non-positive bbox in {path.name}: {ln}")
        if x - w / 2 < 0 or y - h / 2 < 0 or x + w / 2 > 1 or y + h / 2 > 1:
            raise ValueError(f"bbox exceeds bounds in {path.name}: {ln}")
        rows.append((c, x, y, w, h))
    return rows


def yolo_to_xyxy(x: float, y: float, w: float, h: float, iw: int, ih: int) -> tuple[int, int, int, int]:
    x1 = int((x - w / 2) * iw)
    y1 = int((y - h / 2) * ih)
    x2 = int((x + w / 2) * iw)
    y2 = int((y + h / 2) * ih)
    x1 = max(0, min(iw - 1, x1))
    y1 = max(0, min(ih - 1, y1))
    x2 = max(0, min(iw - 1, x2))
    y2 = max(0, min(ih - 1, y2))
    return x1, y1, x2, y2


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO annotations and save visual checks.")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    root = project_root()
    logger = configure_logger("validate_annotations", root / "reports" / "phase1" / "annotation_validation.log")
    paths = load_yaml(root / "configs" / "paths.yaml")

    issues: list[dict[str, str]] = []
    class_counter: Counter[int] = Counter()
    valid_files = 0
    total_objects = 0
    all_label_files: list[tuple[str, Path, Path]] = []

    for split, split_paths in paths["splits"].items():
        images_dir = Path(split_paths["images"])
        labels_dir = Path(split_paths["labels"])
        for label_path in sorted(labels_dir.glob("*.txt")):
            all_label_files.append((split, images_dir, label_path))

    for split, images_dir, label_path in all_label_files:
        img_path = find_image(images_dir, label_path.stem)
        if img_path is None:
            issues.append({"level": "INVALID", "split": split, "file": label_path.name, "reason": "missing image"})
            continue

        # Full image decode for every sample is expensive at dataset scale.
        # Corrupt-image detection is handled in inspect_dataset.py.

        try:
            rows = read_label(label_path)
        except Exception as e:
            issues.append({"level": "INVALID", "split": split, "file": label_path.name, "reason": str(e)})
            continue

        if len(rows) == 0:
            issues.append({"level": "WARNING", "split": split, "file": label_path.name, "reason": "empty annotation file"})

        for c, _x, _y, _w, _h in rows:
            class_counter[c] += 1
        total_objects += len(rows)
        valid_files += 1

    vis_dir = resolve_from_root("results/phase1/figures/annotation_visual_checks")
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample_pool = [it for it in all_label_files if find_image(it[1], it[2].stem) is not None]
    random.shuffle(sample_pool)
    samples = sample_pool[: min(args.samples, len(sample_pool))]

    for idx, (split, images_dir, label_path) in enumerate(samples, start=1):
        img_path = find_image(images_dir, label_path.stem)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        try:
            rows = read_label(label_path)
        except Exception:
            continue

        for c, x, y, bw, bh in rows:
            x1, y1, x2, y2 = yolo_to_xyxy(x, y, bw, bh, w, h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 220, 50), 2)
            cv2.putText(img, f"{c}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 50), 1)

        out_path = vis_dir / f"{idx:03d}_{split}_{img_path.name}"
        cv2.imwrite(str(out_path), img)

    summary = {
        "total_label_files": len(all_label_files),
        "valid_label_files": valid_files,
        "total_objects": total_objects,
        "class_distribution": {str(k): int(v) for k, v in sorted(class_counter.items())},
        "issues_count": len(issues),
    }

    out_json = resolve_from_root("results/phase1/annotation_validation.json")
    out_json.write_text(json.dumps({"summary": summary, "issues": issues}, indent=2), encoding="utf-8")

    report = resolve_from_root("reports/phase1/annotation_validation_report.md")
    lines = [
        "# Annotation Validation Report",
        "",
        "## Summary",
        f"- Total label files: {summary['total_label_files']}",
        f"- Validated files: {summary['valid_label_files']}",
        f"- Total objects: {summary['total_objects']}",
        f"- Class distribution: {summary['class_distribution']}",
        f"- Issues: {summary['issues_count']}",
        "",
        "## Issues (VALID/INVALID/WARNING)",
    ]

    if not issues:
        lines.append("No issues detected.")
    else:
        lines.append("| Level | Split | File | Reason |")
        lines.append("|---|---|---|---|")
        for r in issues:
            lines.append(f"| {r['level']} | {r['split']} | {r['file']} | {r['reason']} |")

    report.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Annotation validation complete. Issues=%d", len(issues))


if __name__ == "__main__":
    main()
