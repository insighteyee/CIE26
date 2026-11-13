"""Phase 1: Ingestion and verification for Abaqus .rpt displacement time histories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

CANONICAL_CHANNELS: tuple[str, ...] = (
    "top_U_mag",
    "top_U1",
    "top_U2",
    "top_U3",
    "arm_U_mag",
    "arm_U1",
    "arm_U2",
    "arm_U3",
)

_NODE_PATTERN = re.compile(r"N:\s*(\d+)")
_LOAD_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*MN", re.IGNORECASE)
_NUMERIC_ROW_PATTERN = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:\s+[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)+$")


@dataclass(frozen=True)
class SimulationRun:
    """Structured container for a single FEM simulation displacement history."""

    run_id: str
    file_path: Path
    label: int
    geometry_tag: str
    load_mn: float
    node_ids: tuple[int, int]
    time: np.ndarray
    channels: pd.DataFrame


def _extract_label(file_path: Path) -> int:
    """Infer binary damage label from parent folder naming."""
    folder_name = file_path.parent.name.lower()
    if "imperfect" in folder_name or "class_1" in folder_name or "damaged" in folder_name:
        return 1
    if "perfect" in folder_name or "class_0" in folder_name or "healthy" in folder_name:
        return 0
    raise ValueError(f"Cannot infer binary label from folder name: {file_path.parent.name}")


def _extract_geometry_tag(file_path: Path) -> str:
    """Infer canonical tower geometry tag ('long' or 'short') from filename."""
    stem_lower = file_path.stem.lower()
    if "long" in stem_lower:
        return "long"
    if "short" in stem_lower or "small" in stem_lower:
        return "short"
    raise ValueError(f"Cannot infer geometry tag from filename: {file_path.name}")


def _extract_load_mn(file_path: Path) -> float:
    """Extract peak dynamic load magnitude in MN from filename."""
    match = _LOAD_PATTERN.search(file_path.stem)
    if match is None:
        raise ValueError(f"Cannot extract load magnitude (MN) from filename: {file_path.name}")
    return float(match.group(1))


def _parse_node_order(header_lines: list[str]) -> list[int]:
    """Extract the ordered sequence of 8 node IDs from the Abaqus .rpt header lines."""
    node_ids: list[int] = []
    for line in header_lines:
        matches = [int(m) for m in _NODE_PATTERN.findall(line)]
        if len(matches) == 8:
            node_ids = matches
            break
    if len(node_ids) != 8:
        raise ValueError(f"Expected 8 node column headers in .rpt file, found {len(node_ids)}")
    unique_nodes = list(dict.fromkeys(node_ids))
    if len(unique_nodes) != 2:
        raise ValueError(f"Expected exactly 2 monitored nodes, found {unique_nodes}")
    return node_ids


def _identify_top_and_arm_nodes(
    raw_matrix: np.ndarray,
    col_node_ids: list[int],
) -> tuple[int, int, dict[str, np.ndarray]]:
    """Map raw column pairs to canonical pinnacle ('top') and cross-arm ('arm') nodes."""
    # Columns 1..8 follow Abaqus order: Mag(A), Mag(B), U1(A), U1(B), U2(A), U2(B), U3(A), U3(B)
    node_a, node_b = col_node_ids[0], col_node_ids[1]
    expected = [node_a, node_b, node_a, node_b, node_a, node_b, node_a, node_b]
    if col_node_ids != expected:
        raise ValueError(f"Unexpected node column ordering: {col_node_ids}")

    a_u1, a_u3 = raw_matrix[:, 3], raw_matrix[:, 7]
    b_u1, b_u3 = raw_matrix[:, 4], raw_matrix[:, 8]

    ratio_a = float(np.max(np.abs(a_u3)) / (np.max(np.abs(a_u1)) + 1e-12))
    ratio_b = float(np.max(np.abs(b_u3)) / (np.max(np.abs(b_u1)) + 1e-12))

    # Cross-arm tip undergoes large vertical bending deflection (U3) compared to the pinnacle top.
    if ratio_a <= ratio_b:
        top_node, arm_node = node_a, node_b
        top_offset, arm_offset = 0, 1
    else:
        top_node, arm_node = node_b, node_a
        top_offset, arm_offset = 1, 0

    channel_arrays = {
        "top_U_mag": raw_matrix[:, 1 + top_offset],
        "top_U1": raw_matrix[:, 3 + top_offset],
        "top_U2": raw_matrix[:, 5 + top_offset],
        "top_U3": raw_matrix[:, 7 + top_offset],
        "arm_U_mag": raw_matrix[:, 1 + arm_offset],
        "arm_U1": raw_matrix[:, 3 + arm_offset],
        "arm_U2": raw_matrix[:, 5 + arm_offset],
        "arm_U3": raw_matrix[:, 7 + arm_offset],
    }
    return top_node, arm_node, channel_arrays


def parse_rpt_file(file_path: Path) -> SimulationRun:
    """Parse a single Abaqus .rpt displacement report into a SimulationRun."""
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    header_lines: list[str] = []
    numeric_rows: list[list[float]] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if _NUMERIC_ROW_PATTERN.match(stripped):
            values = [float(tok) for tok in stripped.split()]
            if len(values) != 9:
                raise ValueError(f"Expected 9 numeric columns in {path}, got {len(values)}")
            numeric_rows.append(values)
        else:
            header_lines.append(stripped)

    if not numeric_rows:
        raise ValueError(f"No numeric time-history rows parsed from {path}")

    col_node_ids = _parse_node_order(header_lines)
    raw_matrix = np.asarray(numeric_rows, dtype=np.float64)
    time = raw_matrix[:, 0].copy()

    top_node, arm_node, channel_arrays = _identify_top_and_arm_nodes(raw_matrix, col_node_ids)
    channels = pd.DataFrame(channel_arrays, columns=list(CANONICAL_CHANNELS))

    label = _extract_label(path)
    geometry_tag = _extract_geometry_tag(path)
    load_mn = _extract_load_mn(path)
    run_id = f"c{label}_{geometry_tag}_{int(round(load_mn))}MN"

    return SimulationRun(
        run_id=run_id,
        file_path=path,
        label=label,
        geometry_tag=geometry_tag,
        load_mn=load_mn,
        node_ids=(top_node, arm_node),
        time=time,
        channels=channels,
    )


def load_dataset(root_dir: Path) -> list[SimulationRun]:
    """Recursively discover and parse all .rpt files under root_dir."""
    root = Path(root_dir)
    rpt_files = sorted(root.rglob("*.rpt"))
    if not rpt_files:
        raise FileNotFoundError(f"No .rpt files found under {root}")

    runs = [parse_rpt_file(p) for p in rpt_files]
    run_ids = [r.run_id for r in runs]
    if len(set(run_ids)) != len(run_ids):
        # Disambiguate duplicate run_id values using file stem if multiple defects exist.
        disambiguated: list[SimulationRun] = []
        for idx, run in enumerate(runs):
            disambiguated.append(
                SimulationRun(
                    run_id=f"{run.run_id}_{idx}",
                    file_path=run.file_path,
                    label=run.label,
                    geometry_tag=run.geometry_tag,
                    load_mn=run.load_mn,
                    node_ids=run.node_ids,
                    time=run.time,
                    channels=run.channels,
                )
            )
        return disambiguated
    return runs


def verify_runs(runs: Sequence[SimulationRun]) -> pd.DataFrame:
    """Validate time-step monotonicity, numerical integrity, and Euclidean norm consistency."""
    records: list[dict[str, object]] = []
    for run in runs:
        t = run.time
        diffs = np.diff(t)
        monotonic_time = bool(len(t) > 1 and np.all(diffs > 0.0))
        dt_mean = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        dt_uniform = bool(len(diffs) > 0 and np.allclose(diffs, dt_mean, atol=1e-9))

        ch = run.channels
        has_nans = bool(np.isnan(t).any() or ch.isna().any().any())
        has_infs = bool(np.isinf(t).any() or np.isinf(ch.to_numpy()).any())

        top_euclid = np.sqrt(ch["top_U1"] ** 2 + ch["top_U2"] ** 2 + ch["top_U3"] ** 2)
        arm_euclid = np.sqrt(ch["arm_U1"] ** 2 + ch["arm_U2"] ** 2 + ch["arm_U3"] ** 2)
        top_mag_err = float(np.max(np.abs(ch["top_U_mag"] - top_euclid)))
        arm_mag_err = float(np.max(np.abs(ch["arm_U_mag"] - arm_euclid)))
        max_mag_error = max(top_mag_err, arm_mag_err)

        top_ratio = float(np.max(np.abs(ch["top_U3"])) / (np.max(np.abs(ch["top_U1"])) + 1e-12))
        arm_ratio = float(np.max(np.abs(ch["arm_U3"])) / (np.max(np.abs(ch["arm_U1"])) + 1e-12))

        is_valid = (
            monotonic_time
            and dt_uniform
            and not has_nans
            and not has_infs
            and max_mag_error < 1e-5
            and list(ch.columns) == list(CANONICAL_CHANNELS)
        )

        records.append(
            {
                "run_id": run.run_id,
                "label": run.label,
                "geometry": run.geometry_tag,
                "load_mn": run.load_mn,
                "top_node": run.node_ids[0],
                "arm_node": run.node_ids[1],
                "n_steps": len(t),
                "t_min_s": float(t[0]),
                "t_max_s": float(t[-1]),
                "dt_s": round(dt_mean, 6),
                "max_mag_err_m": max_mag_error,
                "top_u3_u1_ratio": round(top_ratio, 5),
                "arm_u3_u1_ratio": round(arm_ratio, 5),
                "is_valid": is_valid,
            }
        )

    return pd.DataFrame(records)
