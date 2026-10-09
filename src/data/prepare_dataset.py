from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml, project_root
from src.utils.logging_utils import configure_logger


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
RDD2022_STANDARD = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
}


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def write_manifest(items: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p.resolve()) for p in items) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO manifests and dataset config.")
    parser.add_argument("--paths-config", type=str, default="configs/paths.yaml")
    args = parser.parse_args()

    root = project_root()
    logger = configure_logger("prepare_dataset", root / "reports" / "phase1" / "prepare_dataset.log")

    paths = load_yaml(root / args.paths_config)
    stats = json.loads((root / "results" / "phase1" / "dataset_statistics.json").read_text(encoding="utf-8"))

    split_manifests: dict[str, Path] = {}
    split_counts: dict[str, int] = {}

    for split, split_paths in paths["splits"].items():
        images_dir = Path(split_paths["images"])
        labels_dir = Path(split_paths["labels"])

        selected_images: list[Path] = []
        for label_path in sorted(labels_dir.glob("*.txt")):
            img_path = find_image(images_dir, label_path.stem)
            if img_path is not None:
                selected_images.append(img_path)

        manifest_path = root / "data" / "splits" / f"{split}.txt"
        write_manifest(selected_images, manifest_path)
        split_manifests[split] = manifest_path
        split_counts[split] = len(selected_images)

    class_ids = [int(c) for c in stats.get("class_ids", [])]
    class_ids = sorted(class_ids)

    class_mapping: dict[int, str] = {}
    used_standard = True
    for cid in class_ids:
        if cid in RDD2022_STANDARD:
            class_mapping[cid] = RDD2022_STANDARD[cid]
        else:
            used_standard = False
            class_mapping[cid] = f"class_{cid}"

    names = [class_mapping[cid] for cid in class_ids]
    yolo_yaml: dict[str, Any] = {
        "path": str(root.resolve()),
        "train": str(split_manifests["train"].resolve()),
        "val": str(split_manifests["val"].resolve()),
        "test": str(split_manifests["test"].resolve()),
        "nc": len(names),
        "names": names,
    }

    dataset_yaml = root / "data" / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {yolo_yaml['path']}",
                f"train: {yolo_yaml['train']}",
                f"val: {yolo_yaml['val']}",
                f"test: {yolo_yaml['test']}",
                f"nc: {yolo_yaml['nc']}",
                "names:",
            ] + [f"  - {n}" for n in names]
        )
        + "\n",
        encoding="utf-8",
    )

    class_map_md = root / "reports" / "phase1" / "class_mapping.md"
    lines = [
        "# Class Mapping",
        "",
        "Source: class IDs discovered from provided RDD2022 split labels.",
        "",
        "| Class ID | Class Name |",
        "|---|---|",
    ]
    for cid in class_ids:
        lines.append(f"| {cid} | {class_mapping[cid]} |")
    lines.append("")
    if used_standard:
        lines.append("Mapping uses standard RDD2022 road-damage class names where IDs are available.")
    else:
        lines.append("Some class IDs are non-standard; fallback names class_<id> were used for unknown IDs.")
    class_map_md.write_text("\n".join(lines), encoding="utf-8")

    split_manifest_json = root / "data" / "splits" / "split_manifest.json"
    split_manifest_json.write_text(
        json.dumps(
            {
                "origin": "Official provided split directories in RDD_SPLIT (no reshuffle)",
                "counts": split_counts,
                "manifests": {k: str(v.resolve()) for k, v in split_manifests.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_readme = root / "data" / "raw" / "README.md"
    raw_readme.write_text(
        "Raw data is kept in the external source directory configured in configs/paths.yaml.\n"
        "No raw files are copied, modified, or overwritten by this pipeline.\n",
        encoding="utf-8",
    )

    logger.info("Prepared dataset manifests and dataset.yaml")


if __name__ == "__main__":
    main()
