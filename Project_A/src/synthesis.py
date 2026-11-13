"""Phase 7: Engineering synthesis, diagnostic plots, and ablation reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .classification import ClassificationMetrics, TrainingHistory
from .localization_defense import ObservabilityAudit
from .regression import RegressionMetrics


def compare_feature_regimes(
    raw_metrics: Sequence[ClassificationMetrics],
    invariant_metrics: Sequence[ClassificationMetrics],
) -> pd.DataFrame:
    """Quantify performance deltas between raw displacement inputs and kinematic invariants."""
    raw_by_name = {m.model_name: m for m in raw_metrics}
    inv_by_name = {m.model_name: m for m in invariant_metrics}

    rows: list[dict[str, float | int | str]] = []
    for name in inv_by_name:
        inv_m = inv_by_name[name]
        raw_m = raw_by_name.get(name)
        if raw_m is None:
            continue
        raw_fn = int(raw_m.confusion_matrix[1, 0])
        inv_fn = int(inv_m.confusion_matrix[1, 0])
        rows.append(
            {
                "model_name": name,
                "raw_acc": round(raw_m.accuracy, 4),
                "inv_acc": round(inv_m.accuracy, 4),
                "delta_acc": round(inv_m.accuracy - raw_m.accuracy, 4),
                "raw_recall": round(raw_m.recall, 4),
                "inv_recall": round(inv_m.recall, 4),
                "raw_f1": round(raw_m.f1_score, 4),
                "inv_f1": round(inv_m.f1_score, 4),
                "delta_f1": round(inv_m.f1_score - raw_m.f1_score, 4),
                "raw_auc": round(raw_m.roc_auc, 4),
                "inv_auc": round(inv_m.roc_auc, 4),
                "delta_auc": round(inv_m.roc_auc - raw_m.roc_auc, 4),
                "raw_fn": raw_fn,
                "inv_fn": inv_fn,
            }
        )
    return pd.DataFrame(rows)


def diagnose_fit_regime(
    history: TrainingHistory,
    underfit_loss_tol: float = 0.4,
) -> dict[str, str | float | bool]:
    """Verify underfitting resolution prior to overfitting mitigation."""
    if not history.train_loss or not history.val_loss:
        raise ValueError("TrainingHistory must contain non-empty loss trajectories")

    final_train = float(history.train_loss[-1])
    final_val = float(history.val_loss[-1])
    min_train = float(min(history.train_loss))
    min_val = float(min(history.val_loss))
    gap = final_val - final_train

    underfit_resolved = min_train <= underfit_loss_tol and min_val <= underfit_loss_tol
    if not underfit_resolved:
        regime = "underfitting"
    elif gap > 0.15:
        regime = "overfitting_mitigated_by_early_stopping"
    else:
        regime = "well_fitted"

    return {
        "regime": regime,
        "underfit_resolved": underfit_resolved,
        "min_train_loss": round(min_train, 6),
        "min_val_loss": round(min_val, 6),
        "final_train_loss": round(final_train, 6),
        "final_val_loss": round(final_val, 6),
        "generalization_gap": round(gap, 6),
        "stopped_epoch": int(history.stopped_epoch),
    }


def render_diagnostic_figures(
    output_dir: Path,
    cls_metrics: Sequence[ClassificationMetrics],
    reg_metrics: Sequence[RegressionMetrics],
    histories: dict[str, TrainingHistory],
) -> list[Path]:
    """Generate diagnostic plots for training curves, ROC, confusion matrices, and regression."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # 1. MLP Loss Curves (Raw + Smoothed)
    if histories:
        fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4.5), squeeze=False)
        for idx, (label, hist) in enumerate(histories.items()):
            ax = axes[0, idx]
            epochs = np.arange(1, len(hist.train_loss) + 1)
            ax.plot(epochs, hist.train_loss, alpha=0.25, color="tab:blue", label="Train (raw)")
            ax.plot(epochs, hist.val_loss, alpha=0.25, color="tab:orange", label="Val (raw)")
            ax.plot(epochs, hist.smoothed_train_loss, color="tab:blue", lw=2, label="Train (MA)")
            ax.plot(epochs, hist.smoothed_val_loss, color="tab:orange", lw=2, label="Val (MA)")
            ax.axvline(hist.stopped_epoch, color="tab:red", ls="--", lw=1.2, label=f"Stop @ {hist.stopped_epoch}")
            ax.set_title(f"{label} — BCE Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Binary Cross-Entropy")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        p_loss = out / "mlp_loss_curves.png"
        fig.savefig(p_loss, dpi=150)
        plt.close(fig)
        saved_paths.append(p_loss)

    # 2. Classification ROC Curves
    if cls_metrics:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for m in cls_metrics:
            ax.plot(m.fpr, m.tpr, lw=2, label=f"{m.model_name} (AUC = {m.roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random Baseline")
        ax.set_title("Task 1: Receiver Operating Characteristic (ROC)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        p_roc = out / "classification_roc_curves.png"
        fig.savefig(p_roc, dpi=150)
        plt.close(fig)
        saved_paths.append(p_roc)

        # 3. Confusion Matrix Grid
        n_cls = len(cls_metrics)
        fig, axes = plt.subplots(1, n_cls, figsize=(3.6 * n_cls, 3.6), squeeze=False)
        for idx, m in enumerate(cls_metrics):
            ax = axes[0, idx]
            im = ax.imshow(m.confusion_matrix, cmap="Blues", vmin=0)
            for r in range(2):
                for c in range(2):
                    val = int(m.confusion_matrix[r, c])
                    ax.text(c, r, str(val), ha="center", va="center", color="black", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(f"{m.model_name}\nRec={m.recall:.2f}, tau={m.threshold:.2f}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        p_cm = out / "classification_confusion_matrices.png"
        fig.savefig(p_cm, dpi=150)
        plt.close(fig)
        saved_paths.append(p_cm)

    # 4. Regression Benchmark & Parity Plot
    if reg_metrics:
        names = [r.model_name for r in reg_metrics]
        rmses = [r.rmse for r in reg_metrics]
        r2s = [r.r2 for r in reg_metrics]
        colors = ["tab:blue", "tab:green", "tab:purple"]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.3))
        ax1.bar(names, rmses, color=colors[: len(names)])
        ax1.set_title("Task 3: Test RMSE (meters)")
        ax1.set_ylabel("RMSE [m]")
        ax1.tick_params(axis="x", rotation=15)
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(names, r2s, color=colors[: len(names)])
        ax2.set_title("Task 3: Test Coefficient of Determination (R²)")
        ax2.set_ylabel("R²")
        ax2.set_ylim(min(0.9, min(r2s) - 0.02), 1.005)
        ax2.tick_params(axis="x", rotation=15)
        ax2.grid(axis="y", alpha=0.3)

        for idx, r in enumerate(reg_metrics):
            if r.y_true is not None and r.y_pred is not None:
                ax3.scatter(
                    r.y_true,
                    r.y_pred,
                    s=14,
                    alpha=0.65,
                    color=colors[idx % len(colors)],
                    label=f"{r.model_name} (R²={r.r2:.4f})",
                )
        if any(r.y_true is not None for r in reg_metrics):
            y_all = np.concatenate([r.y_true for r in reg_metrics if r.y_true is not None])
            lims = [float(np.min(y_all)), float(np.max(y_all))]
            ax3.plot(lims, lims, "k--", lw=1.2, label="Parity (y = ŷ)")
            ax3.legend(fontsize=8, loc="upper left")
        ax3.set_title("Task 3: Regression Parity Plot")
        ax3.set_xlabel("True Displacement [m]")
        ax3.set_ylabel("Predicted Displacement [m]")
        ax3.grid(alpha=0.3)

        fig.tight_layout()
        p_reg = out / "regression_benchmark.png"
        fig.savefig(p_reg, dpi=150)
        plt.close(fig)
        saved_paths.append(p_reg)

    return saved_paths


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a GitHub-flavored Markdown table without external dependencies."""
    if df.empty:
        return "_No records._"
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def export_synthesis_report(
    report_path: Path,
    ablation_df: pd.DataFrame,
    cls_metrics: Sequence[ClassificationMetrics],
    reg_metrics: Sequence[RegressionMetrics],
    audit: ObservabilityAudit,
    histories: dict[str, TrainingHistory] | None = None,
) -> Path:
    """Export complete engineering synthesis report across all Project A tasks."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cls_df = pd.DataFrame(
        [
            {
                "Model": m.model_name,
                "Threshold (tau)": round(m.threshold, 3),
                "Accuracy": round(m.accuracy, 4),
                "Precision": round(m.precision, 4),
                "Recall": round(m.recall, 4),
                "F1": round(m.f1_score, 4),
                "ROC-AUC": round(m.roc_auc, 4),
                "FN (Missed Damage)": int(m.confusion_matrix[1, 0]),
                "FP (False Alarm)": int(m.confusion_matrix[0, 1]),
            }
            for m in cls_metrics
        ]
    )

    reg_df = pd.DataFrame(
        [
            {
                "Model": r.model_name,
                "Test MSE [m^2]": f"{r.mse:.4e}",
                "Test RMSE [m]": round(r.rmse, 6),
                "Test R^2": round(r.r2, 6),
            }
            for r in reg_metrics
        ]
    )

    lines = [
        "# Project A — Engineering Synthesis Report",
        "",
        "## 1. Task 1: Global Anomaly Detection (Engineered Kinematic Invariants)",
        "",
        _df_to_markdown(cls_df),
        "",
        "## 2. Feature Engineering Ablation (Raw Displacements vs. Kinematic Invariants)",
        "",
        _df_to_markdown(ablation_df),
        "",
        "## 3. Task 2: Critical Observability Audit (Defect Localization Defense)",
        "",
        f"- **Monitored Nodes**: {audit.monitored_nodes}",
        f"- **Measured DOFs**: {audit.measured_dofs}",
        f"- **Measurement Operator Rank Bound**: {audit.rank_bound}",
        "",
        audit.formal_justification_markdown.strip(),
        "",
        "### Empirical Global Compliance Separations (Cross-Class)",
        "",
        _df_to_markdown(audit.empirical_compliance_collisions.head(8)),
        "",
        "### Required Sensor Modalities for Level 2 Localization",
        "",
    ]
    for mod in audit.required_modalities:
        lines.append(f"- {mod}")

    lines.extend(
        [
            "",
            "## 4. Task 3: Top-Node Displacement Prediction (System Identification)",
            "",
            _df_to_markdown(reg_df),
            "",
        ]
    )

    if histories:
        diag_rows = [{"Model": name, **diagnose_fit_regime(hist)} for name, hist in histories.items()]
        diag_df = pd.DataFrame(diag_rows)
        lines.extend(
            [
                "## 5. Underfitting-Before-Overfitting Diagnostic Verification",
                "",
                _df_to_markdown(diag_df),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
