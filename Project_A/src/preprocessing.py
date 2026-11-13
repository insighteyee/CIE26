"""Phase 2: Temporal masking, kinematic invariants, temporal statistics, and PCA."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .ingestion import CANONICAL_CHANNELS, SimulationRun


def _stabilized_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    eps: float = 1e-12,
    floor: float = 1e-4,
) -> np.ndarray:
    """Divide numerator by denominator while clamping small magnitudes away from zero crossings."""
    effective_floor = max(float(eps), float(floor))
    sign = np.where(denominator >= 0.0, 1.0, -1.0)
    safe_denom = np.where(np.abs(denominator) >= effective_floor, denominator + eps, sign * effective_floor)
    return numerator / safe_denom


def mask_initial_steps(
    run: SimulationRun,
    t_load: float = 0.0,
    min_disp_tol: float = 1e-9,
) -> SimulationRun:
    """Truncate zero-load initialization or pre-excitation intervals from a SimulationRun."""
    max_mag = np.maximum(run.channels["top_U_mag"].to_numpy(), run.channels["arm_U_mag"].to_numpy())
    if t_load > 0.0:
        mask = (run.time >= t_load) & (max_mag > min_disp_tol)
    else:
        mask = (run.time > 0.0) & (max_mag > min_disp_tol)

    if not np.any(mask):
        raise ValueError(f"Temporal masking removed all steps for run {run.run_id}")

    masked_time = run.time[mask].copy()
    masked_channels = run.channels.loc[mask].reset_index(drop=True)

    return SimulationRun(
        run_id=run.run_id,
        file_path=run.file_path,
        label=run.label,
        geometry_tag=run.geometry_tag,
        load_mn=run.load_mn,
        node_ids=run.node_ids,
        time=masked_time,
        channels=masked_channels,
    )


def compute_kinematic_invariants(run: SimulationRun, eps: float = 1e-12) -> pd.DataFrame:
    """Compute load- and geometry-invariant kinematic descriptors at each time step."""
    ch = run.channels

    top_u1 = ch["top_U1"].to_numpy(dtype=np.float64)
    top_u2 = ch["top_U2"].to_numpy(dtype=np.float64)
    top_u3 = ch["top_U3"].to_numpy(dtype=np.float64)
    top_mag = ch["top_U_mag"].to_numpy(dtype=np.float64)

    arm_u1 = ch["arm_U1"].to_numpy(dtype=np.float64)
    arm_u2 = ch["arm_U2"].to_numpy(dtype=np.float64)
    arm_u3 = ch["arm_U3"].to_numpy(dtype=np.float64)
    arm_mag = ch["arm_U_mag"].to_numpy(dtype=np.float64)

    delta_u1 = top_u1 - arm_u1
    delta_u2 = top_u2 - arm_u2
    delta_u3 = top_u3 - arm_u3
    delta_mag = np.sqrt(delta_u1**2 + delta_u2**2 + delta_u3**2)

    arm_mag_safe = arm_mag + eps
    top_mag_safe = top_mag + eps

    # Load-invariant normalized relative node deflections
    norm_delta_u1 = delta_u1 / arm_mag_safe
    norm_delta_u2 = delta_u2 / arm_mag_safe
    norm_delta_u3 = delta_u3 / arm_mag_safe
    norm_delta_mag = delta_mag / arm_mag_safe

    # Kinematic compliance ratios
    kappa = top_mag / arm_mag_safe
    kappa_u1 = np.abs(top_u1) / (np.abs(arm_u1) + max(eps, 1e-4))

    # Cross-axis out-of-plane coupling ratios (U2 / U1 and U2 / ||U||)
    gamma_top = _stabilized_divide(top_u2, top_u1, eps=eps, floor=1e-4)
    gamma_arm = _stabilized_divide(arm_u2, arm_u1, eps=eps, floor=1e-4)
    gamma_top_mag = top_u2 / top_mag_safe
    gamma_arm_mag = arm_u2 / arm_mag_safe

    # Vertical-to-lateral bending coupling ratio on cross-arm
    arm_u3_u1_ratio = _stabilized_divide(arm_u3, arm_u1, eps=eps, floor=1e-4)

    return pd.DataFrame(
        {
            "delta_U1": delta_u1,
            "delta_U2": delta_u2,
            "delta_U3": delta_u3,
            "delta_U_mag": delta_mag,
            "norm_delta_U1": norm_delta_u1,
            "norm_delta_U2": norm_delta_u2,
            "norm_delta_U3": norm_delta_u3,
            "norm_delta_U_mag": norm_delta_mag,
            "kappa": kappa,
            "kappa_u1": kappa_u1,
            "gamma_top": gamma_top,
            "gamma_arm": gamma_arm,
            "gamma_top_mag": gamma_top_mag,
            "gamma_arm_mag": gamma_arm_mag,
            "arm_u3_u1_ratio": arm_u3_u1_ratio,
        }
    )


def extract_temporal_statistics(df_invariants: pd.DataFrame) -> pd.Series:
    """Compute peak amplitude, RMS, variance, and skewness across the loading duration."""
    stats: dict[str, float] = {}
    for col in df_invariants.columns:
        series = df_invariants[col].astype(np.float64)
        vals = series.to_numpy()
        stats[f"{col}__peak"] = float(np.max(np.abs(vals)))
        stats[f"{col}__rms"] = float(np.sqrt(np.mean(vals**2)))
        stats[f"{col}__var"] = float(np.var(vals, ddof=0))
        skew_val = float(series.skew()) if len(series) > 2 and np.std(vals) > 1e-15 else 0.0
        stats[f"{col}__skew"] = 0.0 if np.isnan(skew_val) else skew_val
    return pd.Series(stats, dtype=np.float64)


def build_feature_matrix(
    runs: Sequence[SimulationRun],
    t_load: float = 0.0,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Construct per-timestep invariant feature matrix X, labels y, and grouping run_ids."""
    feature_frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    run_ids: list[str] = []

    for run in runs:
        masked = mask_initial_steps(run, t_load=t_load)
        inv_df = compute_kinematic_invariants(masked, eps=eps)
        # Include Delta U_i, compliance ratio kappa, and cross-axis coupling gamma
        invariant_cols = [
            "delta_U1",
            "delta_U2",
            "delta_U3",
            "delta_U_mag",
            "norm_delta_U1",
            "norm_delta_U2",
            "norm_delta_U3",
            "norm_delta_U_mag",
            "kappa",
            "kappa_u1",
            "gamma_top",
            "gamma_arm",
            "gamma_top_mag",
            "gamma_arm_mag",
            "arm_u3_u1_ratio",
        ]
        selected = inv_df[invariant_cols].copy()
        feature_frames.append(selected)
        labels.append(np.full(len(selected), run.label, dtype=np.int64))
        run_ids.extend([run.run_id] * len(selected))

    X_df = pd.concat(feature_frames, ignore_index=True)
    y = np.concatenate(labels)
    return X_df, y, run_ids


def build_raw_feature_matrix(
    runs: Sequence[SimulationRun],
    t_load: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Construct raw displacement baseline feature matrix X, labels y, and run_ids for ablation."""
    feature_frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    run_ids: list[str] = []

    for run in runs:
        masked = mask_initial_steps(run, t_load=t_load)
        raw_df = masked.channels[list(CANONICAL_CHANNELS)].copy()
        feature_frames.append(raw_df)
        labels.append(np.full(len(raw_df), run.label, dtype=np.int64))
        run_ids.extend([run.run_id] * len(raw_df))

    X_df = pd.concat(feature_frames, ignore_index=True)
    y = np.concatenate(labels)
    return X_df, y, run_ids


def fit_pca_projection(X_train_scaled: np.ndarray, n_components: int | float = 0.95) -> PCA:
    """Fit PCA exclusively on standardized training features."""
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train_scaled)
    return pca


def apply_pca_projection(pca: PCA, X_scaled: np.ndarray) -> np.ndarray:
    """Project standardized features using a pre-fitted PCA transformation."""
    return pca.transform(X_scaled)
