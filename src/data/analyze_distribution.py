from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.utils.config import load_yaml, project_root
from src.utils.logging_utils import configure_logger


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return rows
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        rows.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows


def main() -> None:
    root = project_root()
    logger = configure_logger("analyze_distribution", root / "reports" / "phase1" / "analyze_distribution.log")
    paths = load_yaml(root / "configs" / "paths.yaml")

    class_dist: Counter[int] = Counter()
    objs_per_img: list[int] = []
    bbox_w: list[float] = []
    bbox_h: list[float] = []
    bbox_a: list[float] = []
    resolution_counter: Counter[str] = Counter()
    split_counts: dict[str, int] = {}
    country_counter: Counter[str] = Counter()

    stats_path = root / "results" / "phase1" / "dataset_statistics.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    for split, split_paths in paths["splits"].items():
        images_dir = Path(split_paths["images"])
        labels_dir = Path(split_paths["labels"])
        labels = sorted(labels_dir.glob("*.txt"))
        split_counts[split] = len(labels)

        for lb in labels:
            img = find_image(images_dir, lb.stem)
            if img is not None:
                country = img.stem.split("_")[0] if "_" in img.stem else "unknown"
                country_counter[country] += 1

            rows = parse_rows(lb)
            objs_per_img.append(len(rows))
            for c, _x, _y, w, h in rows:
                class_dist[c] += 1
                bbox_w.append(w)
                bbox_h.append(h)
                bbox_a.append(w * h)

    # Prefer authoritative resolution/country summaries from full audit when available.
    if stats.get("resolution_top10"):
        resolution_counter = Counter(
            {
                f"{r['width']}x{r['height']}": int(r["count"])
                for r in stats["resolution_top10"]
            }
        )
    if stats.get("country_distribution"):
        country_counter = Counter({k: int(v) for k, v in stats["country_distribution"].items()})

    out = {
        "class_distribution": {str(k): int(v) for k, v in sorted(class_dist.items())},
        "objects_per_image": objs_per_img,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "bbox_area": bbox_a,
        "resolution_distribution": dict(resolution_counter),
        "split_distribution": split_counts,
        "country_distribution": dict(country_counter),
    }

    out_path = root / "results" / "phase1" / "distribution_data.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Saved distribution data to %s", out_path)


if __name__ == "__main__":
    main()
