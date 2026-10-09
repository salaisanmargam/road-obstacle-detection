from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.utils.config import project_root, save_yaml
from src.utils.logging_utils import configure_logger


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class InvalidRecord:
    split: str
    file: str
    reason: str
    level: str  # INVALID or WARNING


def _run_cmd(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return (completed.stdout or completed.stderr or "").strip()
    except Exception:
        return ""


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_mb": None,
    }

    nvidia_smi = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if nvidia_smi:
        first = nvidia_smi.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 2:
            info["cuda_available"] = True
            info["gpu_name"] = parts[0]
            m = re.search(r"(\d+)", parts[1])
            if m:
                info["gpu_vram_mb"] = int(m.group(1))

    try:
        import torch

        info["torch_installed"] = True
        info["torch_version"] = str(torch.__version__)
        info["cuda_available_torch"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["gpu_name_torch"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_mb_torch"] = int(props.total_memory / (1024 * 1024))
            info["cuda_available"] = True
            info["gpu_name"] = info.get("gpu_name") or info["gpu_name_torch"]
            info["gpu_vram_mb"] = info.get("gpu_vram_mb") or info["gpu_vram_mb_torch"]
    except Exception:
        info["torch_installed"] = False

    return info


def detect_dataset_root(user_path: str | None) -> Path:
    if user_path:
        root = Path(user_path).resolve()
        if root.exists():
            return root

    candidate = (project_root().parent / "rdd2022" / "RDD_SPLIT").resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError("Could not locate RDD2022 split root. Pass --dataset-root explicitly.")


def list_files(folder: Path, exts: set[str] | None = None) -> list[Path]:
    if not folder.exists():
        return []
    paths = [p for p in folder.rglob("*") if p.is_file()]
    if exts is None:
        return paths
    return [p for p in paths if p.suffix.lower() in exts]


def parse_label_line(line: str) -> tuple[bool, str, tuple[int, float, float, float, float] | None]:
    s = line.strip()
    if not s:
        return False, "empty line", None
    parts = s.split()
    if len(parts) != 5:
        return False, "expected 5 values per line", None

    try:
        class_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
    except Exception:
        return False, "non-numeric annotation values", None

    if class_id < 0:
        return False, "negative class id", None

    vals = [x, y, w, h]
    if any(v < 0 or v > 1 for v in vals):
        return False, "normalized bbox values out of [0,1]", None
    if w <= 0 or h <= 0:
        return False, "bbox width/height must be > 0", None
    if x - w / 2 < 0 or y - h / 2 < 0 or x + w / 2 > 1 or y + h / 2 > 1:
        return False, "bbox exceeds image bounds after normalization", None

    return True, "", (class_id, x, y, w, h)


def hash_file(path: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_report(stats: dict[str, Any], invalid: list[InvalidRecord], out_md: Path) -> None:
    valid_count = sum(1 for r in invalid if r.level == "INVALID")
    warn_count = sum(1 for r in invalid if r.level == "WARNING")

    lines: list[str] = []
    lines.append("# Dataset Audit Report (Phase 1)")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Dataset root: `{stats['dataset_root']}`")
    lines.append(f"- Total images: {stats['total_images']}")
    lines.append(f"- Total annotation files: {stats['total_labels']}")
    lines.append(f"- Splits: {json.dumps(stats['split_counts'])}")
    lines.append(f"- Unique class IDs: {stats['class_ids']}")
    lines.append("")

    lines.append("## Validation Status")
    lines.append("- VALID: completed checks listed in this report")
    lines.append(f"- INVALID: {valid_count} issues")
    lines.append(f"- WARNING: {warn_count} issues")
    lines.append("")

    lines.append("## Key Statistics")
    lines.append(f"- Image formats: {stats['image_formats']}")
    lines.append(f"- Image resolution (top 10): {stats['resolution_top10']}")
    lines.append(f"- Objects total: {stats['total_objects']}")
    lines.append(f"- Objects/image: {stats['objects_per_image']}")
    lines.append(f"- BBox width stats: {stats['bbox_width_stats']}")
    lines.append(f"- BBox height stats: {stats['bbox_height_stats']}")
    lines.append(f"- BBox area stats: {stats['bbox_area_stats']}")
    lines.append(f"- Class distribution: {stats['class_distribution']}")
    lines.append(f"- Country/source distribution: {stats['country_distribution']}")
    lines.append("")

    lines.append("## Issues")
    if not invalid:
        lines.append("No invalid or warning records detected.")
    else:
        lines.append("| Level | Split | File | Reason |")
        lines.append("|---|---|---|---|")
        for r in invalid:
            lines.append(f"| {r.level} | {r.split} | {r.file} | {r.reason} |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RDD2022 dataset and generate statistics.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to RDD_SPLIT root")
    parser.add_argument("--skip-dup-content", action="store_true", help="Skip duplicate content hashing")
    args = parser.parse_args()

    root = project_root()
    logger = configure_logger("dataset_audit", root / "reports" / "phase1" / "dataset_audit.log")

    dataset_root = detect_dataset_root(args.dataset_root)
    splits = ["train", "val", "test"]

    split_dirs: dict[str, dict[str, Path]] = {}
    for split in splits:
        split_dirs[split] = {
            "images": dataset_root / split / "images",
            "labels": dataset_root / split / "labels",
        }

    invalid_records: list[InvalidRecord] = []
    image_paths: list[Path] = []
    label_paths: list[Path] = []
    image_resolutions: Counter[tuple[int, int]] = Counter()
    image_formats: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    country_counts: Counter[str] = Counter()
    objects_per_image: list[int] = []
    bbox_ws: list[float] = []
    bbox_hs: list[float] = []
    bbox_areas: list[float] = []

    split_counts: dict[str, dict[str, int]] = {}

    file_name_counter: Counter[str] = Counter()

    for split in splits:
        img_dir = split_dirs[split]["images"]
        lbl_dir = split_dirs[split]["labels"]

        imgs = list_files(img_dir, IMG_EXTS)
        lbls = list_files(lbl_dir, {".txt"})
        split_counts[split] = {"images": len(imgs), "labels": len(lbls)}

        image_paths.extend(imgs)
        label_paths.extend(lbls)

        image_by_stem = {p.stem: p for p in imgs}
        label_by_stem = {p.stem: p for p in lbls}

        for p in imgs:
            file_name_counter[p.name] += 1
            prefix = p.stem.split("_")[0] if "_" in p.stem else "unknown"
            country_counts[prefix] += 1

        # Missing pair checks
        for stem in sorted(set(image_by_stem) - set(label_by_stem)):
            invalid_records.append(InvalidRecord(split, image_by_stem[stem].name, "missing annotation file", "WARNING"))
        for stem in sorted(set(label_by_stem) - set(image_by_stem)):
            invalid_records.append(InvalidRecord(split, label_by_stem[stem].name, "missing image file", "INVALID"))

        # Image checks
        for stem, img_path in image_by_stem.items():
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im2:
                    w, h = im2.size
                    fmt = (im2.format or "unknown").upper()
                image_resolutions[(w, h)] += 1
                image_formats[fmt] += 1
                if w <= 0 or h <= 0:
                    invalid_records.append(InvalidRecord(split, img_path.name, "invalid image dimensions", "INVALID"))
            except (UnidentifiedImageError, OSError, ValueError):
                invalid_records.append(InvalidRecord(split, img_path.name, "corrupt/unreadable image", "INVALID"))

            if stem in label_by_stem:
                label_path = label_by_stem[stem]
                text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    invalid_records.append(InvalidRecord(split, label_path.name, "empty annotation file", "WARNING"))
                    objects_per_image.append(0)
                    continue

                lines = [ln for ln in text.splitlines() if ln.strip()]
                objects_per_image.append(len(lines))

                for i, ln in enumerate(lines, start=1):
                    ok, reason, parsed = parse_label_line(ln)
                    if not ok:
                        invalid_records.append(
                            InvalidRecord(split, label_path.name, f"line {i}: {reason}", "INVALID")
                        )
                        continue

                    class_id, _x, _y, bw, bh = parsed
                    class_counts[class_id] += 1
                    bbox_ws.append(bw)
                    bbox_hs.append(bh)
                    bbox_areas.append(bw * bh)

    duplicate_filenames = sorted([name for name, c in file_name_counter.items() if c > 1])
    if duplicate_filenames:
        for name in duplicate_filenames:
            invalid_records.append(InvalidRecord("all", name, "duplicate filename across dataset", "WARNING"))

    duplicate_content_groups: list[list[str]] = []
    if not args.skip_dup_content:
        # Fast prefilter: only files sharing identical byte sizes can be content duplicates.
        size_groups: dict[int, list[Path]] = defaultdict(list)
        for p in image_paths:
            try:
                size_groups[p.stat().st_size].append(p)
            except OSError:
                invalid_records.append(InvalidRecord("all", p.name, "failed to read file size", "WARNING"))

        hash_map: dict[str, list[str]] = defaultdict(list)
        for _, files in size_groups.items():
            if len(files) < 2:
                continue
            for p in files:
                try:
                    hash_map[hash_file(p)].append(str(p))
                except OSError:
                    invalid_records.append(InvalidRecord("all", p.name, "failed to hash image content", "WARNING"))

        duplicate_content_groups = [v for v in hash_map.values() if len(v) > 1]
        if duplicate_content_groups:
            invalid_records.append(
                InvalidRecord(
                    "all",
                    Path(duplicate_content_groups[0][0]).name,
                    f"duplicate image content groups: {len(duplicate_content_groups)}",
                    "WARNING",
                )
            )

    class_ids = sorted(class_counts.keys())
    class_mapping = {str(i): f"class_{i}" for i in class_ids}

    def describe(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        arr = np.array(values)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
        }

    objects_stats = {
        "min": int(min(objects_per_image)) if objects_per_image else 0,
        "max": int(max(objects_per_image)) if objects_per_image else 0,
        "mean": float(statistics.mean(objects_per_image)) if objects_per_image else 0.0,
        "median": float(statistics.median(objects_per_image)) if objects_per_image else 0.0,
    }

    stats: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "split_counts": split_counts,
        "total_images": len(image_paths),
        "total_labels": len(label_paths),
        "image_formats": dict(image_formats),
        "resolution_top10": [
            {"width": w, "height": h, "count": c}
            for (w, h), c in image_resolutions.most_common(10)
        ],
        "missing_images": len([r for r in invalid_records if "missing image" in r.reason]),
        "missing_annotations": len([r for r in invalid_records if "missing annotation" in r.reason]),
        "empty_annotations": len([r for r in invalid_records if "empty annotation" in r.reason]),
        "corrupt_images": len([r for r in invalid_records if "corrupt" in r.reason]),
        "invalid_annotations": len([r for r in invalid_records if "line" in r.reason]),
        "duplicate_filenames_count": len(duplicate_filenames),
        "duplicate_filenames": duplicate_filenames[:200],
        "duplicate_content_group_count": len(duplicate_content_groups),
        "class_ids": class_ids,
        "class_mapping_discovered": class_mapping,
        "class_distribution": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "total_objects": int(sum(class_counts.values())),
        "objects_per_image": objects_stats,
        "bbox_width_stats": describe(bbox_ws),
        "bbox_height_stats": describe(bbox_hs),
        "bbox_area_stats": describe(bbox_areas),
        "country_distribution": dict(country_counts),
        "hardware": detect_hardware(),
        "suspicious_notes": [
            "Class names metadata not found in provided dataset tree; using discovered class IDs.",
        ],
    }

    results_json = root / "results" / "phase1" / "dataset_statistics.json"
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    report_md = root / "reports" / "phase1" / "dataset_audit_report.md"
    build_report(stats, invalid_records, report_md)

    invalid_json = root / "results" / "phase1" / "dataset_invalid_samples.json"
    invalid_json.write_text(
        json.dumps([r.__dict__ for r in invalid_records], indent=2),
        encoding="utf-8",
    )

    paths_cfg = {
        "dataset_root": str(dataset_root),
        "splits": {
            split: {
                "images": str(split_dirs[split]["images"]),
                "labels": str(split_dirs[split]["labels"]),
            }
            for split in splits
        },
        "project_root": str(root),
        "artifacts": {
            "dataset_statistics": "results/phase1/dataset_statistics.json",
            "dataset_audit_report": "reports/phase1/dataset_audit_report.md",
            "invalid_samples": "results/phase1/dataset_invalid_samples.json",
        },
        "environment": stats["hardware"],
    }
    save_yaml(paths_cfg, root / "configs" / "paths.yaml")

    logger.info("Dataset audit complete. Images=%d Labels=%d", len(image_paths), len(label_paths))
    logger.info("Saved stats to %s", results_json)
    logger.info("Saved report to %s", report_md)


if __name__ == "__main__":
    main()
