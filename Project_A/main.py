"""CLI orchestrator executing all seven roadmap phases for Project A."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification import (
    ClassificationMetrics,
    TrainingHistory,
    calibrate_recall_threshold,
    evaluate_classifier,
    train_classical_classifiers,
    train_mlp_classifier,
)
from src.ingestion import load_dataset, verify_runs
from src.localization_defense import audit_localization_observability
from src.partitioning import split_by_simulation_run, standardize_splits
from src.preprocessing import (
    apply_pca_projection,
    build_feature_matrix,
    build_raw_feature_matrix,
    compute_kinematic_invariants,
    extract_temporal_statistics,
    fit_pca_projection,
    mask_initial_steps,
)
from src.regression import (
    RegressionMetrics,
    build_regression_dataset,
    evaluate_regressor,
    train_classical_regressors,
    train_mlp_regressor,
)
from src.synthesis import (
    compare_feature_regimes,
    diagnose_fit_regime,
    export_synthesis_report,
    render_diagnostic_figures,
)


def _run_classification_suite(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: list[str],
    strata_tags: list[str],
    random_state: int,
) -> tuple[list[ClassificationMetrics], TrainingHistory]:
    """Partition by run, standardize without leakage, train classifiers, and calibrate recall."""
    tr_idx, va_idx, te_idx = split_by_simulation_run(
        X, y, run_ids, strata_tags, random_state=random_state
    )
    splits = standardize_splits(X, y, run_ids, tr_idx, va_idx, te_idx)

    models = train_classical_classifiers(splits, random_state=random_state)
    mlp, history = train_mlp_classifier(splits, random_state=random_state)
    models["MLP"] = mlp

    metrics_list: list[ClassificationMetrics] = []
    for name, model in models.items():
        val_prob = model.predict_proba(splits.X_val)[:, 1]
        tau = calibrate_recall_threshold(splits.y_val, val_prob, target_recall=1.0)
        test_prob = model.predict_proba(splits.X_test)[:, 1]
        metrics_list.append(evaluate_classifier(name, splits.y_test, test_prob, threshold=tau))

    return metrics_list, history


def run_pipeline(data_dir: Path, output_dir: Path, random_state: int = 42) -> None:
    """Execute Phases 1-7 sequentially and export diagnostic figures and synthesis report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Phase 1: Ingestion & Verification
    runs = load_dataset(Path(data_dir))
    verification_df = verify_runs(runs)
    if not bool(verification_df["is_valid"].all()):
        raise RuntimeError("Dataset verification failed for one or more .rpt files")
    print("=== Phase 1: Dataset Verification ===")
    print(verification_df.to_string(index=False))

    run_strata_map = {r.run_id: f"{r.geometry_tag}_{int(round(r.load_mn))}MN" for r in runs}
    run_label_map = {r.run_id: r.label for r in runs}

    # Phase 2 & 3: Invariant Engineering, Partitioning, Scaling, and PCA Inspection
    temporal_stats_df = pd.DataFrame(
        [
            extract_temporal_statistics(compute_kinematic_invariants(mask_initial_steps(r)))
            for r in runs
        ],
        index=[r.run_id for r in runs],
    )
    X_inv_df, y_inv, inv_run_ids = build_feature_matrix(runs)
    inv_strata = [run_strata_map[rid] for rid in inv_run_ids]
    tr_idx, va_idx, te_idx = split_by_simulation_run(
        X_inv_df.to_numpy(), y_inv, inv_run_ids, inv_strata, random_state=random_state
    )
    inv_splits = standardize_splits(
        X_inv_df.to_numpy(), y_inv, inv_run_ids, tr_idx, va_idx, te_idx
    )
    pca = fit_pca_projection(inv_splits.X_train, n_components=0.95)
    X_test_pca = apply_pca_projection(pca, inv_splits.X_test)
    print(
        f"\n=== Phase 2 & 3: Partitioning, Temporal Stats & PCA ==="
        f"\nRun-level temporal statistics matrix shape: {temporal_stats_df.shape}"
        f"\nTrain runs: {inv_splits.train_run_ids}"
        f"\nVal runs:   {inv_splits.val_run_ids}"
        f"\nTest runs:  {inv_splits.test_run_ids}"
        f"\nPCA 95% variance components: {pca.n_components_} (test proj shape={X_test_pca.shape})"
    )

    # Phase 4: Task 1 Modeling (Invariants vs. Raw Ablation)
    inv_metrics, inv_history = _run_classification_suite(
        X_inv_df.to_numpy(), y_inv, inv_run_ids, inv_strata, random_state=random_state
    )
    X_raw_df, y_raw, raw_run_ids = build_raw_feature_matrix(runs)
    raw_strata = [run_strata_map[rid] for rid in raw_run_ids]
    raw_metrics, raw_history = _run_classification_suite(
        X_raw_df.to_numpy(), y_raw, raw_run_ids, raw_strata, random_state=random_state
    )

    print("\n=== Phase 4: Task 1 MLP Fit Diagnostics ===")
    print("Invariants MLP:", diagnose_fit_regime(inv_history))
    print("Raw MLP:       ", diagnose_fit_regime(raw_history))

    # Phase 5: Task 2 Technical Defense
    audit = audit_localization_observability(runs)
    print(f"\n=== Phase 5: Task 2 Observability Audit (DOFs={audit.measured_dofs}, Rank<={audit.rank_bound}) ===")

    # Phase 6: Task 3 Modeling (Top-Node Displacement Regression)
    X_reg, y_reg, reg_run_ids = build_regression_dataset(runs, target_channel="top_U1", lag_steps=5)
    reg_cls_labels = [run_label_map[rid] for rid in reg_run_ids]
    reg_strata = [run_strata_map[rid] for rid in reg_run_ids]
    r_tr, r_va, r_te = split_by_simulation_run(
        X_reg, reg_cls_labels, reg_run_ids, reg_strata, random_state=random_state
    )
    reg_splits = standardize_splits(X_reg, y_reg, reg_run_ids, r_tr, r_va, r_te)

    reg_models = train_classical_regressors(reg_splits, random_state=random_state)
    mlp_reg, _reg_hist = train_mlp_regressor(reg_splits, random_state=random_state)
    reg_models["MLPRegressor"] = mlp_reg

    reg_metrics: list[RegressionMetrics] = []
    print("\n=== Phase 6: Task 3 Displacement Regression ===")
    for name, model in reg_models.items():
        preds = model.predict(reg_splits.X_test)
        m = evaluate_regressor(name, reg_splits.y_test, preds)
        reg_metrics.append(m)
        print(f"{m.model_name:22s} | MSE={m.mse:.4e} RMSE={m.rmse:.6f} m R2={m.r2:.6f}")

    # Phase 7: Engineering Synthesis & Reporting
    ablation_df = compare_feature_regimes(raw_metrics, inv_metrics)
    print("\n=== Phase 7: Raw vs. Invariants Ablation ===")
    print(ablation_df.to_string(index=False))

    fig_paths = render_diagnostic_figures(
        out,
        cls_metrics=inv_metrics,
        reg_metrics=reg_metrics,
        histories={"Invariants_MLP": inv_history, "Raw_MLP": raw_history},
    )
    report_path = export_synthesis_report(
        out / "synthesis_report.md",
        ablation_df=ablation_df,
        cls_metrics=inv_metrics,
        reg_metrics=reg_metrics,
        audit=audit,
        histories={"Invariants_MLP": inv_history, "Raw_MLP": raw_history},
    )
    print(f"\nArtifacts exported to {out}: {[p.name for p in fig_paths]} + {report_path.name}")


def main() -> None:
    """Parse CLI arguments and run the end-to-end Project A pipeline."""
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Project A Structural Health Monitoring Pipeline")
    parser.add_argument("--data-dir", type=Path, default=default_root, help="Directory containing .rpt files")
    parser.add_argument("--output-dir", type=Path, default=default_root / "outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.output_dir, random_state=args.seed)


if __name__ == "__main__":
    main()
