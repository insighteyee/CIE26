"""Phase 3: Run-grouped dataset partitioning and leak-free standardization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetSplits:
    """Container for leak-free standardized train, validation, and test partitions."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_run_ids: list[str]
    val_run_ids: list[str]
    test_run_ids: list[str]
    scaler: StandardScaler


def split_by_simulation_run(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: Sequence[str],
    strata_tags: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition row indices by simulation run while maintaining stratified class coverage."""
    if len(X) != len(y) or len(X) != len(run_ids) or len(X) != len(strata_tags):
        raise ValueError("X, y, run_ids, and strata_tags must have identical lengths")
    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.0")

    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y)
    run_arr = np.asarray(run_ids)
    strata_arr = np.asarray(strata_tags)

    unique_runs: list[str] = list(dict.fromkeys(run_arr.tolist()))
    run_to_label: dict[str, int] = {}
    run_to_strata: dict[str, str] = {}
    for r in unique_runs:
        mask = run_arr == r
        unique_y = np.unique(y_arr[mask])
        if len(unique_y) != 1:
            raise ValueError(f"Run {r} contains multiple class labels: {unique_y}")
        run_to_label[r] = int(unique_y[0])
        run_to_strata[r] = str(strata_arr[mask][0])

    # Group runs by class label so every partition receives both healthy and damaged runs.
    runs_by_class: dict[int, list[str]] = {}
    for r in unique_runs:
        runs_by_class.setdefault(run_to_label[r], []).append(r)

    train_runs: list[str] = []
    val_runs: list[str] = []
    test_runs: list[str] = []

    for cls_label in sorted(runs_by_class):
        cls_runs = list(runs_by_class[cls_label])
        cls_runs.sort(key=lambda r: (run_to_strata[r], r))

        # Interleave across strata_tags (e.g., geometry_load) so that contiguous slices
        # for train/val/test receive balanced geometry and load diversity across classes.
        by_stratum: dict[str, list[str]] = {}
        for r in cls_runs:
            by_stratum.setdefault(run_to_strata[r], []).append(r)
        stratum_keys = sorted(by_stratum.keys())
        for k in stratum_keys:
            rng.shuffle(by_stratum[k])

        # Reorder sorted stratum keys by interleaving forward and reversed halves
        # so any prefix slice spans distinct geometries and load levels simultaneously.
        half = (len(stratum_keys) + 1) // 2
        first_half = stratum_keys[:half]
        second_half = list(reversed(stratum_keys[half:]))
        alternating_keys: list[str] = []
        for idx in range(half):
            alternating_keys.append(first_half[idx])
            if idx < len(second_half):
                alternating_keys.append(second_half[idx])

        interleaved: list[str] = []
        while any(by_stratum.values()):
            for k in alternating_keys:
                if by_stratum[k]:
                    interleaved.append(by_stratum[k].pop(0))

        # Rotate across classes so train covers complementary strata across class 0 and class 1.
        shift = (2 * cls_label) % len(interleaved)
        ordered = interleaved[shift:] + interleaved[:shift]

        n_cls = len(ordered)
        if n_cls < 3:
            raise ValueError(
                f"Class {cls_label} has only {n_cls} runs; need >= 3 runs for 3-way run-grouped split."
            )

        n_val = max(1, int(round(n_cls * val_ratio)))
        n_test = max(1, int(round(n_cls * (1.0 - train_ratio - val_ratio))))
        n_train = n_cls - n_val - n_test
        if n_train < 1:
            n_train = 1
            remainder = n_cls - n_train
            n_val = max(1, remainder // 2)
            n_test = max(1, remainder - n_val)

        train_runs.extend(ordered[:n_train])
        val_runs.extend(ordered[n_train : n_train + n_val])
        test_runs.extend(ordered[n_train + n_val : n_train + n_val + n_test])

    train_set = set(train_runs)
    val_set = set(val_runs)
    test_set = set(test_runs)
    if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
        raise AssertionError("Run leakage detected across train/val/test sets")

    all_idx = np.arange(len(run_arr))
    train_idx = all_idx[np.isin(run_arr, list(train_set))]
    val_idx = all_idx[np.isin(run_arr, list(val_set))]
    test_idx = all_idx[np.isin(run_arr, list(test_set))]
    return train_idx, val_idx, test_idx


def standardize_splits(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> DatasetSplits:
    """Fit StandardScaler strictly on X[train_idx] and transform all three splits."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    run_arr = np.asarray(run_ids)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_arr[train_idx])
    X_val_scaled = scaler.transform(X_arr[val_idx])
    X_test_scaled = scaler.transform(X_arr[test_idx])

    train_runs = list(dict.fromkeys(run_arr[train_idx].tolist()))
    val_runs = list(dict.fromkeys(run_arr[val_idx].tolist()))
    test_runs = list(dict.fromkeys(run_arr[test_idx].tolist()))

    if (set(train_runs) & set(val_runs)) or (set(train_runs) & set(test_runs)) or (set(val_runs) & set(test_runs)):
        raise ValueError("Data leakage detected: shared run_id across dataset partitions")

    return DatasetSplits(
        X_train=X_train_scaled,
        y_train=y_arr[train_idx].copy(),
        X_val=X_val_scaled,
        y_val=y_arr[val_idx].copy(),
        X_test=X_test_scaled,
        y_test=y_arr[test_idx].copy(),
        train_run_ids=train_runs,
        val_run_ids=val_runs,
        test_run_ids=test_runs,
        scaler=scaler,
    )
