# Project A — File-Scoped Implementation Plan

Based on [projectA-instructions.md](file:///home/nam/SiSc/CIE/Project_A/projectA-instructions.md), below is the target architecture mapped to the seven roadmap phases.

---

## 1. Target Directory Tree

```text
Project_A/
├── projectA-instructions.md
├── implementation_plan.md
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── partitioning.py
│   ├── classification.py
│   ├── localization_defense.py
│   ├── regression.py
│   └── synthesis.py
└── main.py
```

---

## 2. Roadmap Phase Mapping & Module Boundaries

### Phase 1: Ingestion & Verification (`.rpt` Parsing)

#### `Project_A/src/__init__.py`
* Exposes public package interfaces across all seven pipeline stages.

#### `Project_A/src/ingestion.py`
* Parses raw Abaqus `.rpt` simulation files into structured numerical arrays while extracting folder-encoded binary labels (`perfect=0`, `imperfect=1`) and simulation metadata.
* Validates unit consistency, column completeness, node identifiers, and monotonic time-step increments across all ingested runs.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SimulationRun:
    run_id: str
    file_path: Path
    label: int
    geometry_tag: str
    load_mn: float
    node_ids: tuple[int, int]
    time: np.ndarray
    channels: pd.DataFrame

def parse_rpt_file(file_path: Path) -> SimulationRun: ...
def load_dataset(root_dir: Path) -> list[SimulationRun]: ...
def verify_runs(runs: Sequence[SimulationRun]) -> pd.DataFrame: ...
```

---

### Phase 2: Preprocessing & Invariant Engineering (Kinematic Invariants)

#### `Project_A/src/preprocessing.py`
* Applies temporal masking to truncate static initialization or zero-load intervals prior to dynamic excitation.
* Computes load- and geometry-invariant kinematic time series ($\Delta U_i$, $\kappa$, $\gamma$), run-level temporal statistical descriptors, and Principal Component projections.

```python
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .ingestion import SimulationRun

def mask_initial_steps(run: SimulationRun, t_load: float = 0.0, min_disp_tol: float = 1e-9) -> SimulationRun: ...
def compute_kinematic_invariants(run: SimulationRun, eps: float = 1e-12) -> pd.DataFrame: ...
def extract_temporal_statistics(df_invariants: pd.DataFrame) -> pd.Series: ...
def build_feature_matrix(runs: Sequence[SimulationRun], t_load: float = 0.0, eps: float = 1e-12) -> tuple[pd.DataFrame, np.ndarray, list[str]]: ...
def fit_pca_projection(X_train_scaled: np.ndarray, n_components: int | float = 0.95) -> PCA: ...
def apply_pca_projection(pca: PCA, X_scaled: np.ndarray) -> np.ndarray: ...
```

---

### Phase 3: Partitioning & Scaling (Leak-Free Scaling)

#### `Project_A/src/partitioning.py`
* Partitions data into 70% training, 15% validation, and 15% test sets grouped strictly by simulation run with stratified labels and unseen condition holdouts.
* Fits `StandardScaler` exclusively on the training split and transforms validation and test partitions without data leakage.

```python
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class DatasetSplits:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def standardize_splits(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> DatasetSplits: ...
```

---

### Phase 4: Task 1 Modeling (Global Anomaly Detection)

#### `Project_A/src/classification.py`
* Trains and tunes classical baselines (Logistic Regression, Linear/RBF SVM, Random Forest) alongside a ReLU-activated MLP classifier optimized with Binary Cross-Entropy and early stopping.
* Evaluates confusion matrices, ROC/AUC, and precision/recall/F-score while calibrating the decision threshold $\tau$ to maximize damaged-class recall and eliminate False Negatives.

```python
from dataclasses import dataclass
from typing import Any
import numpy as np
from .partitioning import DatasetSplits

@dataclass(frozen=True)
class ClassificationMetrics:
    model_name: str
    threshold: float
    confusion_matrix: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    fpr: np.ndarray
    tpr: np.ndarray

@dataclass(frozen=True)
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    smoothed_train_loss: list[float]
    smoothed_val_loss: list[float]
    stopped_epoch: int

def train_classical_classifiers(splits: DatasetSplits, random_state: int = 42) -> dict[str, Any]: ...
def train_mlp_classifier(
    splits: DatasetSplits,
    hidden_units: tuple[int, ...] = (32, 16),
    lr: float = 1e-3,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 20,
    random_state: int = 42,
) -> tuple[Any, TrainingHistory]: ...
def calibrate_recall_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 1.0) -> float: ...
def evaluate_classifier(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics: ...
```

---

### Phase 5: Task 2 Technical Defense (Localization Defense)

#### `Project_A/src/localization_defense.py`
* Quantifies observability rank deficiency by comparing the 6-DOF measurement space against the full lattice truss stiffness parameterization and demonstrates non-unique compliance states empirically.
* Outputs a structured technical justification and specifies minimum sensor network modalities required for spatial defect localization.

```python
from dataclasses import dataclass
from typing import Sequence
import pandas as pd
from .ingestion import SimulationRun

@dataclass(frozen=True)
class ObservabilityAudit:
    monitored_nodes: int
    measured_dofs: int
    rank_bound: int
    empirical_compliance_collisions: pd.DataFrame
    required_modalities: list[str]
    formal_justification_markdown: str

def audit_localization_observability(runs: Sequence[SimulationRun], tol: float = 1e-3) -> ObservabilityAudit: ...
```

---

### Phase 6: Task 3 Modeling (Displacement Prediction / Regression)

#### `Project_A/src/regression.py`
* Formulates supervised regression inputs mapping historical step windows ($t < t^*$) and concurrent loading-node displacements to top-node displacement targets at $t^*$.
* Benchmarks Ridge Regression, Random Forest Regressor, and a linear-output MLP regressor using test MSE, RMSE, and $R^2$ metrics.

```python
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from .ingestion import SimulationRun
from .partitioning import DatasetSplits

@dataclass(frozen=True)
class RegressionMetrics:
    model_name: str
    mse: float
    rmse: float
    r2: float

def build_regression_dataset(
    runs: Sequence[SimulationRun],
    target_channel: str,
    lag_steps: int = 5,
    include_concurrent_load_node: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]: ...

def train_classical_regressors(splits: DatasetSplits, random_state: int = 42) -> dict[str, Any]: ...
def train_mlp_regressor(
    splits: DatasetSplits,
    hidden_units: tuple[int, ...] = (32, 16),
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 300,
    patience: int = 20,
    random_state: int = 42,
) -> tuple[Any, dict[str, list[float]]]: ...
def evaluate_regressor(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics: ...
```

---

### Phase 7: Engineering Synthesis & Reporting

#### `Project_A/src/synthesis.py`
* Compares classical baselines against neural architectures and quantifies classification/regression deltas between raw displacement features and engineered kinematic invariants.
* Generates diagnostic plots (loss curves, ROC curves, confusion matrices, regression parity plots) and verifies underfitting-before-overfitting resolution criteria.

```python
from pathlib import Path
from typing import Sequence
import pandas as pd
from .classification import ClassificationMetrics, TrainingHistory
from .localization_defense import ObservabilityAudit
from .regression import RegressionMetrics

def compare_feature_regimes(
    raw_metrics: Sequence[ClassificationMetrics],
    invariant_metrics: Sequence[ClassificationMetrics],
) -> pd.DataFrame: ...

def diagnose_fit_regime(history: TrainingHistory, underfit_loss_tol: float = 0.4) -> dict[str, str | float | bool]: ...
def render_diagnostic_figures(
    output_dir: Path,
    cls_metrics: Sequence[ClassificationMetrics],
    reg_metrics: Sequence[RegressionMetrics],
    histories: dict[str, TrainingHistory],
) -> list[Path]: ...
def export_synthesis_report(
    report_path: Path,
    ablation_df: pd.DataFrame,
    cls_metrics: Sequence[ClassificationMetrics],
    reg_metrics: Sequence[RegressionMetrics],
    audit: ObservabilityAudit,
) -> Path: ...
```

#### `Project_A/main.py`
* Orchestrates sequential execution from Phase 1 ingestion through Phase 7 synthesis via a CLI entry point.

```python
from pathlib import Path

def run_pipeline(data_dir: Path, output_dir: Path, random_state: int = 42) -> None: ...
def main() -> None: ...
```
