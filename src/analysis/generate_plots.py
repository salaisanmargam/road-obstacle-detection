from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def yolo_to_xyxy(x: float, y: float, w: float, h: float, iw: int, ih: int) -> tuple[int, int, int, int]:
    return (
        max(0, int((x - w / 2) * iw)),
        max(0, int((y - h / 2) * ih)),
        min(iw - 1, int((x + w / 2) * iw)),
        min(ih - 1, int((y + h / 2) * ih)),
    )


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    rows: list[tuple[int, float, float, float, float]] = []
    if not txt:
        return rows
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        rows.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    seed_everything(42)
    root = project_root()
    logger = configure_logger("generate_plots", root / "reports" / "phase1" / "generate_plots.log")
    fig_dir = root / "results" / "phase1" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dist = json.loads((root / "results" / "phase1" / "distribution_data.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "results" / "phase1" / "dataset_statistics.json").read_text(encoding="utf-8"))

    sns.set_theme(style="whitegrid")

    cls = dist["class_distribution"]
    plt.figure(figsize=(8, 4))
    plt.bar(list(cls.keys()), list(cls.values()), color="#1f77b4")
    plt.title("Class Distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    save_fig(fig_dir / "class_distribution.png")

    plt.figure(figsize=(8, 4))
    plt.hist(dist["objects_per_image"], bins=30, color="#2ca02c", edgecolor="black")
    plt.title("Objects Per Image")
    plt.xlabel("Objects")
    plt.ylabel("Frequency")
    save_fig(fig_dir / "objects_per_image_hist.png")

    plt.figure(figsize=(8, 4))
    plt.hist(dist["bbox_width"], bins=50, color="#ff7f0e", edgecolor="black")
    plt.title("BBox Width Distribution (normalized)")
    plt.xlabel("Width")
    plt.ylabel("Frequency")
    save_fig(fig_dir / "bbox_width_hist.png")

    plt.figure(figsize=(8, 4))
    plt.hist(dist["bbox_height"], bins=50, color="#9467bd", edgecolor="black")
    plt.title("BBox Height Distribution (normalized)")
    plt.xlabel("Height")
    plt.ylabel("Frequency")
    save_fig(fig_dir / "bbox_height_hist.png")

    plt.figure(figsize=(8, 4))
    plt.hist(dist["bbox_area"], bins=50, color="#8c564b", edgecolor="black")
    plt.title("BBox Area Distribution (normalized)")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    save_fig(fig_dir / "bbox_area_hist.png")

    resolutions = dist["resolution_distribution"]
    top_res = sorted(resolutions.items(), key=lambda x: x[1], reverse=True)[:15]
    plt.figure(figsize=(10, 5))
    plt.bar([k for k, _v in top_res], [v for _k, v in top_res], color="#17becf")
    plt.title("Top Image Resolutions")
    plt.xlabel("Resolution")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_fig(fig_dir / "resolution_distribution.png")

    splits = dist["split_distribution"]
    plt.figure(figsize=(6, 4))
    plt.bar(list(splits.keys()), list(splits.values()), color="#d62728")
    plt.title("Split Distribution")
    plt.xlabel("Split")
    plt.ylabel("Count")
    save_fig(fig_dir / "split_distribution.png")

    countries = dist["country_distribution"]
    plt.figure(figsize=(8, 4))
    plt.bar(list(countries.keys()), list(countries.values()), color="#7f7f7f")
    plt.title("Country/Source Distribution")
    plt.xlabel("Source")
    plt.ylabel("Count")
    save_fig(fig_dir / "country_distribution.png")

    # Random GT visualization grid
    paths = load_yaml(root / "configs" / "paths.yaml")
    class_map = {int(k): v for k, v in stats.get("class_mapping_discovered", {}).items()}
    sample_items: list[tuple[str, Path, Path]] = []
    for split, split_paths in paths["splits"].items():
        images_dir = Path(split_paths["images"])
        labels_dir = Path(split_paths["labels"])
        for lb in labels_dir.glob("*.txt"):
            img = find_image(images_dir, lb.stem)
            if img is not None:
                sample_items.append((split, img, lb))

    random.shuffle(sample_items)
    n = min(16, len(sample_items))
    grid = int(math.ceil(math.sqrt(n)))

    fig, axes = plt.subplots(grid, grid, figsize=(16, 16))
    axes = np.array(axes).reshape(-1)

    for i in range(grid * grid):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue

        split, img_path, lb_path = sample_items[i]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        rows = parse_label(lb_path)

        for c, x, y, bw, bh in rows:
            x1, y1, x2, y2 = yolo_to_xyxy(x, y, bw, bh, w, h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = class_map.get(c, f"class_{c}")
            cv2.putText(img, name, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ax.imshow(img)
        ax.set_title(f"{split}:{img_path.name}", fontsize=8)

    plt.suptitle("Random Annotated Ground-Truth Samples", fontsize=16)
    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    plt.savefig(fig_dir / "annotated_gt_grid.png", dpi=300)
    plt.close(fig)

    logger.info("Saved plots to %s", fig_dir)


if __name__ == "__main__":
    main()
