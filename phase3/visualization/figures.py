# phase3/visualization/figures.py
"""
Publication-quality figure generation for Phase 3.
Generates all required figures as PNG files.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

FIG_DIR = Path("D:/road obstacle detection/road_damage_research/results/phase3/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "phase1": "#2196F3",
    "phase2": "#FF9800",
    "phase3": "#4CAF50",
    "attack": "#F44336",
    "benign": "#2196F3",
}


def _save(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {path.name}")
    return path


# ── Fig 1: Pipeline architecture (text-based) ──────────────────────────────
def fig_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    boxes = [
        (0.5, 0.92, "🎥 Roadside Camera", "#E3F2FD", 12),
        (0.5, 0.78, "🌐 Network Traffic", "#E3F2FD", 11),
        (0.5, 0.64, "🛡  Security Monitor\n(Random Forest)", "#FFEBEE", 11),
        (0.2, 0.50, "✗ SUSPICIOUS\n→ Alert + Block", "#FFCDD2", 10),
        (0.8, 0.50, "✓ NORMAL\n→ Continue", "#C8E6C9", 10),
        (0.8, 0.36, "🔍 Condition Analyzer\n(Classical CV)", "#FFF9C4", 10),
        (0.8, 0.22, "✨ Adaptive Enhancement\n(if degraded)", "#FFF9C4", 10),
        (0.8, 0.08, "🚗 YOLOv8n Detection\n(Road Damage)", "#E8F5E9", 11),
    ]

    for x, y, text, color, fs in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#555", lw=1.5),
                transform=ax.transAxes, wrap=True)

    arrows = [
        (0.5, 0.88, 0.5, 0.83),
        (0.5, 0.74, 0.5, 0.69),
        (0.5, 0.60, 0.2, 0.55),
        (0.5, 0.60, 0.8, 0.55),
        (0.8, 0.45, 0.8, 0.41),
        (0.8, 0.31, 0.8, 0.27),
        (0.8, 0.17, 0.8, 0.13),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=2, color="#333"))

    ax.set_title("Phase 3: Security-Aware Road Damage Detection Pipeline",
                 fontsize=14, fontweight="bold", pad=15)
    return _save(fig, "fig1_pipeline_architecture")


# ── Fig 2: mAP@50 comparison across phases ─────────────────────────────────
def fig_map_comparison(results: dict):
    systems = ["Phase 1\n(YOLOv8n)", "Phase 2\n(+Enhancement)", "Phase 3\n(+Security)"]
    map50   = [
        results.get("phase1_map50",   0.5724),
        results.get("phase2_map50",   0.5866),
        results.get("phase3_map50",   0.0),
    ]
    map5095 = [
        results.get("phase1_map5095", 0.3066),
        results.get("phase2_map5095", 0.3070),
        results.get("phase3_map5095", 0.0),
    ]

    x = np.arange(len(systems))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, map50,   w, label="mAP@50",    color=["#2196F3","#FF9800","#4CAF50"])
    b2 = ax.bar(x + w/2, map5095, w, label="mAP@50:95", color=["#90CAF9","#FFCC80","#A5D6A7"])
    ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=11)
    ax.set_ylabel("mAP", fontsize=12)
    ax.set_title("Detection Performance: Phase 1 vs Phase 2 vs Phase 3", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.75)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig2_map_comparison")


# ── Fig 3: Robustness comparison ────────────────────────────────────────────
def fig_robustness_comparison(phase2_results: dict, phase3_results: dict):
    conditions = ["low_light", "low_contrast", "blur", "noise", "haze", "rain_like"]
    severities = ["mild", "moderate", "severe"]

    p2_d = [phase2_results.get(f"D_{c}_avg", 0) for c in conditions]
    p3_d = [phase3_results.get(f"D_{c}_avg", p2) for c, p2 in zip(conditions, p2_d)]

    x = np.arange(len(conditions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, p2_d, w, label="Phase 2 (Adaptive)", color="#FF9800", alpha=0.85)
    ax.bar(x + w/2, p3_d, w, label="Phase 3 (Secure+Adaptive)", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=10)
    ax.set_ylabel("Avg mAP@50 (across severities)", fontsize=11)
    ax.set_title("Robustness: Phase 2 vs Phase 3 (Adaptive Enhancement, Exp D)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5724, color="#2196F3", lw=1.5, linestyle="--", label="Phase 1 baseline")
    ax.legend(fontsize=10)
    return _save(fig, "fig3_robustness_comparison")


# ── Fig 4: Security confusion matrix ────────────────────────────────────────
def fig_security_confusion_matrix(cm_data: list, model_name: str = "Random Forest"):
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted\nNormal", "Predicted\nSuspicious"],
                yticklabels=["Actual\nNormal", "Actual\nSuspicious"],
                annot_kws={"size": 13})
    ax.set_title(f"Security Model Confusion Matrix\n({model_name})", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    return _save(fig, "fig4_security_confusion_matrix")


# ── Fig 5: Security metrics bar chart ───────────────────────────────────────
def fig_security_metrics(rf_metrics: dict, lr_metrics: dict):
    metrics = ["precision", "recall", "f1", "roc_auc"]
    labels  = ["Precision", "Recall", "F1", "ROC-AUC"]
    rf_vals = [rf_metrics.get(m, 0) for m in metrics]
    lr_vals = [lr_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, rf_vals, w, label="Random Forest",        color="#4CAF50", alpha=0.85)
    ax.bar(x + w/2, lr_vals, w, label="Logistic Regression",  color="#9E9E9E", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Security Model Evaluation Metrics (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    for bars in [ax.containers[0], ax.containers[1]]:
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig5_security_metrics")


# ── Fig 6: Latency breakdown comparison ─────────────────────────────────────
def fig_latency_comparison(latency_results: list):
    """latency_results: list of dicts with system and avg_*_ms keys."""
    systems = [r["system"] for r in latency_results]
    totals  = [r.get("avg_total_ms", 0) for r in latency_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]
    bars = ax.bar(systems, totals, color=colors[:len(systems)], alpha=0.85, width=0.5)
    ax.set_ylabel("Average Latency (ms/image)", fontsize=12)
    ax.set_title("End-to-End Latency Comparison by System", fontsize=13, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f} ms",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(totals) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig6_latency_comparison")


# ── Fig 7: FPS comparison ────────────────────────────────────────────────────
def fig_fps_comparison(latency_results: list):
    systems = [r["system"] for r in latency_results]
    fps_vals = [r.get("fps", 0) for r in latency_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]
    bars = ax.bar(systems, fps_vals, color=colors[:len(systems)], alpha=0.85, width=0.5)
    ax.set_ylabel("Frames Per Second (FPS)", fontsize=12)
    ax.set_title("Throughput (FPS) Comparison by System", fontsize=13, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(fps_vals) * 1.3)
    ax.axhline(30, color="red", lw=1.5, linestyle="--", alpha=0.6, label="30 FPS threshold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig7_fps_comparison")


# ── Fig 8: Accuracy/Security/Latency trade-off ──────────────────────────────
def fig_tradeoff(latency_results: list, cv_results: list):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]
    for i, (lat, cv) in enumerate(zip(latency_results, cv_results)):
        sys_name = lat["system"]
        fps   = lat.get("fps", 0)
        map50 = cv.get("map50", 0)
        ax.scatter(fps, map50, s=200, color=colors[i], zorder=5,
                   label=sys_name, edgecolors="black", lw=1.5)
        ax.annotate(sys_name, (fps, map50), textcoords="offset points",
                    xytext=(8, 5), fontsize=10)
    ax.set_xlabel("FPS (Throughput)", fontsize=12)
    ax.set_ylabel("mAP@50 (Accuracy)", fontsize=12)
    ax.set_title("Accuracy vs Throughput Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    return _save(fig, "fig8_tradeoff")


# ── Fig 9: Ablation study ────────────────────────────────────────────────────
def fig_ablation(ablation_results: list):
    configs = [r["config"] for r in ablation_results]
    map50s  = [r.get("map50", 0) for r in ablation_results]
    fps_vals= [r.get("fps", 0) for r in ablation_results]

    x = np.arange(len(configs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]
    ax1.bar(x, map50s, color=colors[:len(x)], alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(configs, fontsize=9, rotation=15, ha="right")
    ax1.set_ylabel("mAP@50", fontsize=12)
    ax1.set_title("Ablation: Detection Accuracy", fontsize=12, fontweight="bold")
    for i, v in enumerate(map50s):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, 0.75); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, fps_vals, color=colors[:len(x)], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(configs, fontsize=9, rotation=15, ha="right")
    ax2.set_ylabel("FPS", fontsize=12)
    ax2.set_title("Ablation: Throughput", fontsize=12, fontweight="bold")
    for i, v in enumerate(fps_vals):
        ax2.text(i, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Ablation Study: Component Contributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "fig9_ablation_study")


# ── Fig 10: Security latency overhead ───────────────────────────────────────
def fig_security_overhead(latency_results: list):
    fig, ax = plt.subplots(figsize=(10, 5))
    systems = [r["system"] for r in latency_results]
    sec_ms  = [r.get("avg_security_ms", 0)  for r in latency_results]
    cond_ms = [r.get("avg_condition_ms", 0) for r in latency_results]
    enh_ms  = [r.get("avg_enhance_ms", 0)   for r in latency_results]
    yolo_ms = [r.get("avg_yolo_ms", 0)      for r in latency_results]

    x = np.arange(len(systems))
    w = 0.5
    ax.bar(x, yolo_ms, w, label="YOLO",         color="#2196F3")
    ax.bar(x, enh_ms,  w, bottom=yolo_ms,        label="Enhancement", color="#FF9800")
    bot2 = [a+b for a,b in zip(yolo_ms, enh_ms)]
    ax.bar(x, cond_ms, w, bottom=bot2,           label="Condition Analyzer", color="#9C27B0")
    bot3 = [a+b for a,b in zip(bot2, cond_ms)]
    ax.bar(x, sec_ms,  w, bottom=bot3,           label="Security Check", color="#F44336")

    ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=11)
    ax.set_ylabel("Avg Latency (ms/image)", fontsize=12)
    ax.set_title("Latency Breakdown by Component", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig10_latency_breakdown")


def generate_all_figures(results: dict):
    """Generate all figures from a results dictionary."""
    print("\n=== Generating Figures ===")

    fig_pipeline_architecture()

    fig_map_comparison(results)

    p2 = results.get("phase2_robustness", {})
    p3 = results.get("phase3_robustness", {})
    fig_robustness_comparison(p2, p3)

    rf_test = results.get("rf_test_metrics", {})
    lr_test = results.get("lr_test_metrics", {})
    if "confusion_matrix" in rf_test:
        fig_security_confusion_matrix(rf_test["confusion_matrix"])
    if rf_test and lr_test:
        fig_security_metrics(rf_test, lr_test)

    latency_list = results.get("latency_results", [])
    if latency_list:
        fig_latency_comparison(latency_list)
        fig_fps_comparison(latency_list)
        fig_security_overhead(latency_list)

    cv_list = results.get("cv_results", [])
    if latency_list and cv_list:
        fig_tradeoff(latency_list, cv_list)

    ablation = results.get("ablation_results", [])
    if ablation:
        fig_ablation(ablation)

    print(f"\n  All figures saved to: {FIG_DIR}")
