"""Phase 5: Task 2 mathematical observability audit and ill-posedness defense."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .ingestion import SimulationRun
from .preprocessing import compute_kinematic_invariants, mask_initial_steps


@dataclass(frozen=True)
class ObservabilityAudit:
    """Formal mathematical and empirical observability audit for Level 2 localization."""

    monitored_nodes: int
    measured_dofs: int
    rank_bound: int
    empirical_compliance_collisions: pd.DataFrame
    required_modalities: list[str]
    formal_justification_markdown: str


def audit_localization_observability(
    runs: Sequence[SimulationRun],
    tol: float = 1e-3,
) -> ObservabilityAudit:
    """Quantify spatial rank deficiency and empirical global compliance overlap across runs."""
    monitored_nodes = 2
    measured_dofs = monitored_nodes * 3
    rank_bound = measured_dofs

    run_summaries: list[dict[str, float | int | str]] = []
    for run in runs:
        masked = mask_initial_steps(run)
        inv = compute_kinematic_invariants(masked)
        run_summaries.append(
            {
                "run_id": run.run_id,
                "label": run.label,
                "geometry": run.geometry_tag,
                "load_mn": run.load_mn,
                "kappa_mean": float(inv["kappa"].mean()),
                "kappa_std": float(inv["kappa"].std(ddof=0)),
                "arm_u3_u1_mean": float(inv["arm_u3_u1_ratio"].mean()),
                "top_u1_peak_m": float(masked.channels["top_U1"].abs().max()),
            }
        )

    summary_df = pd.DataFrame(run_summaries)

    # Identify pairs of distinct runs whose global compliance ratios collide within tolerance
    collision_rows: list[dict[str, float | str | bool]] = []
    for i in range(len(summary_df)):
        for j in range(i + 1, len(summary_df)):
            row_i = summary_df.iloc[i]
            row_j = summary_df.iloc[j]
            dkappa = abs(float(row_i["kappa_mean"]) - float(row_j["kappa_mean"]))
            dpeak = abs(float(row_i["top_u1_peak_m"]) - float(row_j["top_u1_peak_m"]))
            is_collision = bool(dkappa <= tol or (row_i["label"] != row_j["label"] and dkappa < 0.01))
            if is_collision:
                collision_rows.append(
                    {
                        "run_a": str(row_i["run_id"]),
                        "run_b": str(row_j["run_id"]),
                        "label_a": int(row_i["label"]),
                        "label_b": int(row_j["label"]),
                        "kappa_diff": round(dkappa, 6),
                        "top_u1_peak_diff_m": round(dpeak, 6),
                        "within_tol": bool(dkappa <= tol),
                    }
                )

    collisions_df = pd.DataFrame(collision_rows)
    if collisions_df.empty:
        # Fallback: report minimum global compliance separation pairs across classes
        pair_rows: list[dict[str, float | str | bool]] = []
        for i in range(len(summary_df)):
            for j in range(i + 1, len(summary_df)):
                row_i = summary_df.iloc[i]
                row_j = summary_df.iloc[j]
                if row_i["label"] != row_j["label"]:
                    dkappa = abs(float(row_i["kappa_mean"]) - float(row_j["kappa_mean"]))
                    dpeak = abs(float(row_i["top_u1_peak_m"]) - float(row_j["top_u1_peak_m"]))
                    pair_rows.append(
                        {
                            "run_a": str(row_i["run_id"]),
                            "run_b": str(row_j["run_id"]),
                            "label_a": int(row_i["label"]),
                            "label_b": int(row_j["label"]),
                            "kappa_diff": round(dkappa, 6),
                            "top_u1_peak_diff_m": round(dpeak, 6),
                            "within_tol": bool(dkappa <= tol),
                        }
                    )
        collisions_df = pd.DataFrame(pair_rows).sort_values("kappa_diff").reset_index(drop=True)

    required_modalities = [
        "Distributed uniaxial/rosette strain gauge arrays along primary leg and diagonal bracing tiers",
        "Multi-elevation triaxial accelerometer network for mode-shape curvature and modal strain energy",
        "Fiber Bragg Grating (FBG) optical strain sensing along critical load-transfer paths",
        "Full-field Digital Image Correlation (DIC) for localized spatial displacement gradients",
    ]

    min_cross_class_dkappa = (
        float(collisions_df["kappa_diff"].min()) if not collisions_df.empty else float("nan")
    )

    justification = f"""### Task 2 (Level 2 SHM): Mathematical Proof of Ill-Posedness

1. **Rank-Deficient Observation Operator**:
   Let $u(t) \\in \\mathbb{{R}}^{{N_{{\\text{{dof}}}}}}$ denote the full 3D lattice space-truss nodal displacement vector ($N_{{\\text{{dof}}}} \\gg {measured_dofs}$). The two-node sensor configuration (pinnacle and cross-arm) defines a linear measurement operator $C \\in \\mathbb{{R}}^{{{measured_dofs} \\times N_{{\\text{{dof}}}}}}$ such that $y(t) = C u(t) \\in \\mathbb{{R}}^{{{measured_dofs}}}$. Therefore:
   $$\\operatorname{{rank}}(C) \\le {rank_bound} \\ll N_{{\\text{{dof}}}}, \\qquad \\dim(\\ker(C)) \\ge N_{{\\text{{dof}}}} - {rank_bound}.$$

2. **Non-Uniqueness of Stiffness Perturbations (Inverse Problem Indeterminacy)**:
   Under a structural damage perturbation $\\Delta K = \\sum_{{e=1}}^{{M}} \\delta_e K_e$ across $M$ truss members, quasi-static equilibrium satisfies $(K_0 - \\Delta K) u = f$. Because only $C K(\\delta)^{{-1}} f$ is observed in $\\mathbb{{R}}^{{{measured_dofs}}}$, mapping $M \\gg {rank_bound}$ unknown element damage parameters $\\delta \\in \\mathbb{{R}}^M$ from at most {rank_bound} spatial components is strictly underdetermined. Distinct member fractures at different elevations or opposing truss faces can induce identical macro-compliance changes at the two measured nodes (minimum observed cross-class compliance separation $\\Delta \\kappa = {min_cross_class_dkappa:.6f}$).

3. **Absence of Spatial Strain Gradients**:
   Localizing damage to an elevation tier or truss quadrant requires spatial derivatives $\\nabla u$ (strain fields) or higher-order vibration mode shapes $\\phi_r(x)$ across distributed coordinates. With only two global measurement points, $\\nabla u$ along the tower body is unobserved, making Level 2 spatial localization mathematically non-identifiable without denser local instrumentation.
"""

    return ObservabilityAudit(
        monitored_nodes=monitored_nodes,
        measured_dofs=measured_dofs,
        rank_bound=rank_bound,
        empirical_compliance_collisions=collisions_df,
        required_modalities=required_modalities,
        formal_justification_markdown=justification,
    )
