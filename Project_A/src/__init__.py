"""Structural Health Monitoring (SHM) ML pipeline package for Project A."""

from .classification import (
    ClassificationMetrics,
    NumpyMLPClassifier,
    TrainingHistory,
    calibrate_recall_threshold,
    evaluate_classifier,
    train_classical_classifiers,
    train_mlp_classifier,
)
from .ingestion import CANONICAL_CHANNELS, SimulationRun, load_dataset, parse_rpt_file, verify_runs
from .localization_defense import ObservabilityAudit, audit_localization_observability
from .partitioning import DatasetSplits, split_by_simulation_run, standardize_splits
from .preprocessing import (
    apply_pca_projection,
    build_feature_matrix,
    build_raw_feature_matrix,
    compute_kinematic_invariants,
    extract_temporal_statistics,
    fit_pca_projection,
    mask_initial_steps,
)
from .regression import (
    NumpyMLPRegressor,
    RegressionMetrics,
    build_regression_dataset,
    evaluate_regressor,
    train_classical_regressors,
    train_mlp_regressor,
)
from .synthesis import (
    compare_feature_regimes,
    diagnose_fit_regime,
    export_synthesis_report,
    render_diagnostic_figures,
)

__all__ = [
    "CANONICAL_CHANNELS",
    "ClassificationMetrics",
    "DatasetSplits",
    "NumpyMLPClassifier",
    "NumpyMLPRegressor",
    "ObservabilityAudit",
    "RegressionMetrics",
    "SimulationRun",
    "TrainingHistory",
    "apply_pca_projection",
    "audit_localization_observability",
    "build_feature_matrix",
    "build_raw_feature_matrix",
    "build_regression_dataset",
    "calibrate_recall_threshold",
    "compare_feature_regimes",
    "compute_kinematic_invariants",
    "diagnose_fit_regime",
    "evaluate_classifier",
    "evaluate_regressor",
    "export_synthesis_report",
    "extract_temporal_statistics",
    "fit_pca_projection",
    "load_dataset",
    "mask_initial_steps",
    "parse_rpt_file",
    "render_diagnostic_figures",
    "split_by_simulation_run",
    "standardize_splits",
    "train_classical_classifiers",
    "train_classical_regressors",
    "train_mlp_classifier",
    "train_mlp_regressor",
    "verify_runs",
]
