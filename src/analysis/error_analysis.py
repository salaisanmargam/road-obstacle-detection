from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.config import project_root
from src.utils.logging_utils import configure_logger


def main() -> None:
    root = project_root()
    logger = configure_logger("error_analysis", root / "reports" / "phase1" / "error_analysis.log")

    metrics_path = root / "results" / "phase1" / "baseline_metrics.json"
    dist_path = root / "results" / "phase1" / "distribution_data.json"
    qual_path = root / "results" / "phase1" / "qualitative_summary.json"

    if not metrics_path.exists():
        raise FileNotFoundError("baseline_metrics.json not found. Run evaluation first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    dist = json.loads(dist_path.read_text(encoding="utf-8")) if dist_path.exists() else {}
    qual = json.loads(qual_path.read_text(encoding="utf-8")) if qual_path.exists() else {}

    bbox_area = np.array(dist.get("bbox_area", []), dtype=float)
    small_thr = float(np.quantile(bbox_area, 0.25)) if bbox_area.size else None
    large_thr = float(np.quantile(bbox_area, 0.75)) if bbox_area.size else None

    agg = metrics.get("aggregate_metrics", {})
    per_class = metrics.get("per_class_metrics", [])

    hard_classes = sorted(
        [pc for pc in per_class if pc.get("ap50") is not None],
        key=lambda x: x.get("ap50", 0.0),
    )

    lines = [
        "# Baseline Error Analysis (Phase 1)",
        "",
        "## Observed Results",
        f"- Precision: {agg.get('precision')}",
        f"- Recall: {agg.get('recall')}",
        f"- F1: {agg.get('f1')}",
        f"- mAP@50: {agg.get('map50')}",
        f"- mAP@50:95: {agg.get('map50_95')}",
        f"- Qualitative category counts: {qual}",
        "",
        "## Error Patterns",
        "- Small objects: expected higher miss rate due to limited pixel footprint.",
        "- Large objects: generally easier, but fragmented damage may reduce IoU quality.",
        "- Overlapping objects: can trigger NMS suppression errors or class confusion.",
        "- Visually similar damage: likely contributes to inter-class confusion.",
        "- Low-quality images/background clutter: observed as false positives in hard scenes.",
        "",
    ]

    if small_thr is not None and large_thr is not None:
        lines += [
            "## Bounding-Box Size Regimes",
            f"- Small-object threshold (25th percentile area): {small_thr:.6f}",
            f"- Large-object threshold (75th percentile area): {large_thr:.6f}",
            "",
        ]

    lines += [
        "## Per-Class Difficulty (by AP@50 ascending)",
    ]
    if hard_classes:
        lines.append("| Class | Precision | Recall | F1 | AP@50 | AP@50:95 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in hard_classes:
            lines.append(
                f"| {row.get('class_name')} | {row.get('precision')} | {row.get('recall')} | {row.get('f1')} | {row.get('ap50')} | {row.get('ap50_95')} |"
            )
    else:
        lines.append("Per-class details unavailable from current evaluation output.")

    lines += [
        "",
        "## Interpretation",
        "- Current baseline provides a reproducible reference, not a robustness-optimized system.",
        "- False negatives are expected to concentrate on small or low-contrast damage regions.",
        "",
        "## Limitations",
        "- No weather/illumination-adaptive preprocessing in Phase 1 by design.",
        "- No domain-held-out robustness experiments in this phase.",
        "",
        "## Future Research (Phase 2 direction)",
        "- Introduce adaptive image enhancement while preserving this baseline protocol for fair comparison.",
    ]

    out = root / "reports" / "phase1" / "baseline_error_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved baseline error analysis to %s", out)


if __name__ == "__main__":
    main()
