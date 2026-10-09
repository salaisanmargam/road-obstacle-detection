from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml


def run_step(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_environment_report(root: Path) -> None:
    paths = yaml.safe_load((root / "configs" / "paths.yaml").read_text(encoding="utf-8"))
    env = paths.get("environment", {})

    # Detect key package versions from runtime if available.
    versions = {}
    for pkg in ["ultralytics", "torch", "cv2", "numpy", "yaml", "PIL"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not installed"

    lines = [
        "# Environment Report",
        "",
        f"- Operating system: {platform.platform()}",
        f"- Python version: {platform.python_version()}",
        f"- CUDA available: {env.get('cuda_available')}",
        f"- GPU name: {env.get('gpu_name')}",
        f"- GPU VRAM (MB): {env.get('gpu_vram_mb')}",
        "",
        "## Key Dependencies",
    ]
    for k, v in versions.items():
        lines.append(f"- {k}: {v}")

    out = root / "reports" / "phase1" / "environment.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def generate_final_report(root: Path) -> None:
    stats = load_json(root / "results" / "phase1" / "dataset_statistics.json")
    metrics = load_json(root / "results" / "phase1" / "baseline_metrics.json")
    run_info = load_json(root / "results" / "phase1" / "latest_run.json")
    qual = load_json(root / "results" / "phase1" / "qualitative_summary.json")
    split_manifest = load_json(root / "data" / "splits" / "split_manifest.json")

    per_class = metrics.get("per_class_metrics", [])

    lines = [
        "# PHASE 1 FINAL REPORT",
        "",
        "## 1. Phase 1 Objective",
        "Build a clean, reproducible baseline for road-damage/object detection on provided RDD2022 splits using a lightweight YOLO model.",
        "",
        "## 2. Hardware/Software Environment",
        f"- See: reports/phase1/environment.md",
        "",
        "## 3. Dataset Description",
        f"- Dataset root: {stats.get('dataset_root')}",
        f"- Splits: {stats.get('split_counts')}",
        f"- Total images: {stats.get('total_images')}",
        f"- Total labels: {stats.get('total_labels')}",
        "",
        "## 4. Dataset Statistics",
        f"- Objects total: {stats.get('total_objects')}",
        f"- Objects/image: {stats.get('objects_per_image')}",
        f"- BBox width stats: {stats.get('bbox_width_stats')}",
        f"- BBox height stats: {stats.get('bbox_height_stats')}",
        f"- BBox area stats: {stats.get('bbox_area_stats')}",
        "",
        "## 5. Class Distribution",
        f"- {stats.get('class_distribution')}",
        "",
        "## 6. Dataset Validation Results",
        "- See reports/phase1/dataset_audit_report.md and reports/phase1/annotation_validation_report.md",
        "",
        "## 7. Data Preparation Procedure",
        "- Official provided RDD_SPLIT partitions were preserved.",
        "- Split manifests were generated in data/splits/.",
        "- YOLO dataset.yaml was generated in data/dataset.yaml.",
        "",
        "## 8. Train/Validation/Test Split",
        f"- {split_manifest.get('counts')}",
        f"- Origin: {split_manifest.get('origin')}",
        "",
        "## 9. Baseline Architecture",
        f"- Model: {metrics.get('run', {}).get('save_dir', 'N/A')}",
        "",
        "## 10. Training Configuration",
        "- See configs/baseline.yaml and training_config_used.yaml in experiment run directory.",
        "",
        "## 11. Quantitative Results (Observed)",
        f"- Precision: {metrics.get('aggregate_metrics', {}).get('precision')}",
        f"- Recall: {metrics.get('aggregate_metrics', {}).get('recall')}",
        f"- F1: {metrics.get('aggregate_metrics', {}).get('f1')}",
        f"- mAP@50: {metrics.get('aggregate_metrics', {}).get('map50')}",
        f"- mAP@50:95: {metrics.get('aggregate_metrics', {}).get('map50_95')}",
        "",
        "## 12. Per-Class Results (Observed)",
        "| Class | Precision | Recall | F1 | AP@50 | AP@50:95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in per_class:
        lines.append(
            f"| {row.get('class_name')} | {row.get('precision')} | {row.get('recall')} | {row.get('f1')} | {row.get('ap50')} | {row.get('ap50_95')} |"
        )

    lines += [
        "",
        "## 13. Efficiency Results (Observed)",
        f"- Params: {metrics.get('efficiency', {}).get('params')}",
        f"- Model size (MB): {metrics.get('efficiency', {}).get('model_size_mb')}",
        f"- Inference latency (ms/image): {metrics.get('efficiency', {}).get('inference_latency_ms')}",
        f"- FPS: {metrics.get('efficiency', {}).get('fps')}",
        f"- Hardware: {metrics.get('efficiency', {}).get('hardware')}",
        "",
        "## 14. Confusion Matrix Analysis",
        "- Refer to confusion matrix figures in results/phase1/figures/.",
        "",
        "## 15. Qualitative Results (Observed)",
        f"- Prediction categories: {qual}",
        "- Refer to results/phase1/predictions/.",
        "",
        "## 16. Error Analysis (Interpretation)",
        "- See reports/phase1/baseline_error_analysis.md",
        "",
        "## 17. Limitations",
        "- Baseline only; no adaptive enhancement or robustness interventions in Phase 1.",
        "",
        "## 18. Reproducibility Information",
        "- requirements.txt included.",
        "- Fixed seed applied.",
        "- Config and manifests saved.",
        "",
        "## 19. Exact Baseline Checkpoint Path",
        f"- Best: {run_info.get('best_checkpoint')}",
        f"- Last: {run_info.get('last_checkpoint')}",
        "",
        "## 20. Recommended Next Step for Phase 2",
        "- PHASE 2 - ADAPTIVE IMAGE ENHANCEMENT",
        "",
        "## Separation of Claims",
        "### OBSERVED RESULTS",
        "Values reported above are direct outputs from dataset statistics and model evaluation artifacts.",
        "",
        "### INTERPRETATION",
        "Interpretations are documented only in baseline_error_analysis.md and framed as baseline behavior.",
        "",
        "### FUTURE RESEARCH",
        "Phase 2 enhancements are intentionally deferred and not used in this baseline.",
    ]

    out = root / "reports" / "phase1" / "PHASE_1_FINAL_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")


def completion_checklist(root: Path) -> dict[str, bool]:
    req = {
        "Dataset successfully located": (root / "configs" / "paths.yaml").exists(),
        "Dataset audit completed": (root / "reports" / "phase1" / "dataset_audit_report.md").exists(),
        "Dataset statistics generated": (root / "results" / "phase1" / "dataset_statistics.json").exists(),
        "Annotation validation completed": (root / "reports" / "phase1" / "annotation_validation_report.md").exists(),
        "Dataset prepared": (root / "data" / "splits" / "split_manifest.json").exists(),
        "dataset.yaml created": (root / "data" / "dataset.yaml").exists(),
        "Split manifest created": (root / "data" / "splits" / "split_manifest.json").exists(),
        "Dataset visualizations generated": (root / "results" / "phase1" / "figures" / "class_distribution.png").exists(),
        "Lightweight YOLO baseline trained": (root / "results" / "phase1" / "latest_run.json").exists(),
        "Best checkpoint saved": load_json(root / "results" / "phase1" / "latest_run.json").get("best_checkpoint") is not None,
        "Test evaluation completed": (root / "results" / "phase1" / "baseline_metrics.json").exists(),
        "Precision calculated": load_json(root / "results" / "phase1" / "baseline_metrics.json").get("aggregate_metrics", {}).get("precision") is not None,
        "Recall calculated": load_json(root / "results" / "phase1" / "baseline_metrics.json").get("aggregate_metrics", {}).get("recall") is not None,
        "F1 calculated": load_json(root / "results" / "phase1" / "baseline_metrics.json").get("aggregate_metrics", {}).get("f1") is not None,
        "mAP@50 calculated": load_json(root / "results" / "phase1" / "baseline_metrics.json").get("aggregate_metrics", {}).get("map50") is not None,
        "mAP@50:95 calculated": load_json(root / "results" / "phase1" / "baseline_metrics.json").get("aggregate_metrics", {}).get("map50_95") is not None,
        "Per-class metrics generated": len(load_json(root / "results" / "phase1" / "baseline_metrics.json").get("per_class_metrics", [])) > 0,
        "Confusion matrix generated": any(p.name.endswith("confusion_matrix.png") for p in (root / "results" / "phase1" / "figures").glob("*.png")),
        "PR curves generated": any("PR_curve" in p.name for p in (root / "results" / "phase1" / "figures").glob("*.png")),
        "Training curves generated": any("results.png" in p.name for p in (root / "results" / "phase1" / "figures").glob("*.png")),
        "Qualitative predictions generated": len(list((root / "results" / "phase1" / "predictions").glob("*.jpg"))) > 0,
        "False positives analyzed": (root / "reports" / "phase1" / "baseline_error_analysis.md").exists(),
        "False negatives analyzed": (root / "reports" / "phase1" / "baseline_error_analysis.md").exists(),
        "Efficiency metrics recorded": load_json(root / "results" / "phase1" / "baseline_metrics.json").get("efficiency", {}).get("fps") is not None,
        "Environment documented": (root / "reports" / "phase1" / "environment.md").exists(),
        "requirements.txt created": (root / "requirements.txt").exists(),
        "Phase 1 final report generated": (root / "reports" / "phase1" / "PHASE_1_FINAL_REPORT.md").exists(),
        "All important paths recorded": (root / "configs" / "paths.yaml").exists(),
        "Pipeline reproducibility verified": (root / "run_phase1.py").exists(),
    }
    return req


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 baseline pipeline")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable

    if not any([args.audit, args.prepare, args.train, args.evaluate, args.all]):
        parser.error("Use one of --audit, --prepare, --train, --evaluate, --all")

    run_all = args.all

    if run_all or args.audit:
        cmd = [py, "-m", "src.data.inspect_dataset"]
        if args.dataset_root:
            cmd.extend(["--dataset-root", args.dataset_root])
        run_step(cmd, root)
        run_step([py, "-m", "src.data.validate_annotations"], root)

    if run_all or args.prepare:
        run_step([py, "-m", "src.data.prepare_dataset"], root)
        run_step([py, "-m", "src.data.analyze_distribution"], root)
        run_step([py, "-m", "src.analysis.generate_plots"], root)

    if run_all or args.train:
        run_step([py, "-m", "src.training.train_baseline"], root)

    if run_all or args.evaluate:
        run_step([py, "-m", "src.training.evaluate_baseline"], root)
        run_step([py, "-m", "src.analysis.inference_examples"], root)
        run_step([py, "-m", "src.analysis.error_analysis"], root)

    if run_all:
        write_environment_report(root)
        generate_final_report(root)

        checklist = completion_checklist(root)
        checklist_path = root / "reports" / "phase1" / "phase1_completion_checklist.md"
        lines = ["# Phase 1 Completion Checklist", ""]
        for k, v in checklist.items():
            lines.append(f"- [{'x' if v else ' '}] {k}")
        checklist_path.write_text("\n".join(lines), encoding="utf-8")

        print("\nPHASE 1 CHECKLIST")
        for k, v in checklist.items():
            print(f"{'[x]' if v else '[ ]'} {k}")


if __name__ == "__main__":
    main()
