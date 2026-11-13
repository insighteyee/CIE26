"""Phase 4: Task 1 global anomaly detection models, diagnostics, and threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC

from .partitioning import DatasetSplits


@dataclass(frozen=True)
class ClassificationMetrics:
    """Evaluation summary for a binary damage classifier at a calibrated threshold."""

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
    """Epoch-level training and validation loss trajectories with moving-average smoothing."""

    train_loss: list[float]
    val_loss: list[float]
    smoothed_train_loss: list[float]
    smoothed_val_loss: list[float]
    stopped_epoch: int


def _smooth_curve(values: list[float], window: int = 5) -> list[float]:
    """Compute causal moving average for loss curve visualization."""
    if not values:
        return []
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(float(np.mean(values[start : idx + 1])))
    return smoothed


class NumpyMLPClassifier:
    """Compact ReLU feed-forward neural network trained via mini-batch Adam on BCE loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...] = (32, 16),
        l2_reg: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.layer_dims = (input_dim, *hidden_units, 1)
        self.l2_reg = l2_reg
        rng = np.random.default_rng(random_state)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for idx in range(len(self.layer_dims) - 1):
            fan_in = self.layer_dims[idx]
            fan_out = self.layer_dims[idx + 1]
            # He initialization for ReLU hidden layers, Xavier for final sigmoid layer
            scale = np.sqrt(2.0 / fan_in) if idx < len(self.layer_dims) - 2 else np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z_clip = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations: list[np.ndarray] = []
        a = X
        n_layers = len(self.weights)
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            pre_activations.append(z)
            if idx < n_layers - 1:
                a = np.maximum(0.0, z)
            else:
                a = self._sigmoid(z)
            activations.append(a)
        return activations, pre_activations

    def compute_bce_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss (excluding L2 penalty for clean diagnostics)."""
        probs = np.clip(self.predict_proba(X)[:, 1], 1e-7, 1.0 - 1e-7)
        y_col = y.astype(np.float64)
        return float(-np.mean(y_col * np.log(probs) + (1.0 - y_col) * np.log(1.0 - probs)))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lr: float = 1e-3,
        batch_size: int = 16,
        max_epochs: int = 300,
        patience: int = 20,
        random_state: int = 42,
    ) -> TrainingHistory:
        """Train with mini-batch Adam and early stopping on validation BCE loss."""
        rng = np.random.default_rng(random_state)
        n_samples = X_train.shape[0]
        y_train_col = y_train.astype(np.float64).reshape(-1, 1)

        m_W = [np.zeros_like(W) for W in self.weights]
        v_W = [np.zeros_like(W) for W in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        best_val_loss = float("inf")
        best_weights = [W.copy() for W in self.weights]
        best_biases = [b.copy() for b in self.biases]
        epochs_no_improve = 0
        stopped_epoch = max_epochs

        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(1, max_epochs + 1):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = perm[start : start + batch_size]
                Xb = X_train[batch_idx]
                yb = y_train_col[batch_idx]
                bs = float(len(batch_idx))

                activations, pre_activations = self._forward(Xb)
                dz = (activations[-1] - yb) / bs

                grad_W: list[np.ndarray] = [np.empty_like(W) for W in self.weights]
                grad_b: list[np.ndarray] = [np.empty_like(b) for b in self.biases]

                for layer_idx in reversed(range(len(self.weights))):
                    a_prev = activations[layer_idx]
                    grad_W[layer_idx] = a_prev.T @ dz + self.l2_reg * self.weights[layer_idx]
                    grad_b[layer_idx] = np.sum(dz, axis=0, keepdims=True)
                    if layer_idx > 0:
                        da_prev = dz @ self.weights[layer_idx].T
                        dz = da_prev * (pre_activations[layer_idx - 1] > 0.0)

                step += 1
                bc1 = 1.0 - beta1**step
                bc2 = 1.0 - beta2**step
                for idx in range(len(self.weights)):
                    m_W[idx] = beta1 * m_W[idx] + (1.0 - beta1) * grad_W[idx]
                    v_W[idx] = beta2 * v_W[idx] + (1.0 - beta2) * (grad_W[idx] ** 2)
                    m_hat = m_W[idx] / bc1
                    v_hat = v_W[idx] / bc2
                    self.weights[idx] -= lr * m_hat / (np.sqrt(v_hat) + eps)

                    m_b[idx] = beta1 * m_b[idx] + (1.0 - beta1) * grad_b[idx]
                    v_b[idx] = beta2 * v_b[idx] + (1.0 - beta2) * (grad_b[idx] ** 2)
                    mb_hat = m_b[idx] / bc1
                    vb_hat = v_b[idx] / bc2
                    self.biases[idx] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

            tr_loss = self.compute_bce_loss(X_train, y_train)
            va_loss = self.compute_bce_loss(X_val, y_val)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)

            if va_loss < best_val_loss - 1e-6:
                best_val_loss = va_loss
                best_weights = [W.copy() for W in self.weights]
                best_biases = [b.copy() for b in self.biases]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    stopped_epoch = epoch
                    break

        self.weights = best_weights
        self.biases = best_biases

        return TrainingHistory(
            train_loss=train_losses,
            val_loss=val_losses,
            smoothed_train_loss=_smooth_curve(train_losses),
            smoothed_val_loss=_smooth_curve(val_losses),
            stopped_epoch=stopped_epoch,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities [P(y=0|x), P(y=1|x)] matching scikit-learn convention."""
        activations, _ = self._forward(np.asarray(X, dtype=np.float64))
        p1 = activations[-1].ravel()
        return np.column_stack([1.0 - p1, p1])


def train_classical_classifiers(splits: DatasetSplits, random_state: int = 42) -> dict[str, Any]:
    """Train and validation-tune classical baselines for global anomaly detection."""
    models: dict[str, Any] = {}

    # 1. Logistic Regression (tuned on validation log-loss)
    best_lr, best_lr_loss = None, float("inf")
    for c_val in (0.01, 0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=c_val, max_iter=2000, random_state=random_state)
        clf.fit(splits.X_train, splits.y_train)
        val_prob = clf.predict_proba(splits.X_val)[:, 1]
        loss = float(log_loss(splits.y_val, val_prob, labels=[0, 1]))
        if loss < best_lr_loss:
            best_lr_loss = loss
            best_lr = clf
    models["LogisticRegression"] = best_lr

    # 2. Linear SVM
    best_lsvm, best_lsvm_loss = None, float("inf")
    for c_val in (0.1, 1.0, 10.0):
        clf = CalibratedClassifierCV(
            SVC(kernel="linear", C=c_val, random_state=random_state),
            ensemble=False,
        )
        clf.fit(splits.X_train, splits.y_train)
        val_prob = clf.predict_proba(splits.X_val)[:, 1]
        loss = float(log_loss(splits.y_val, val_prob, labels=[0, 1]))
        if loss < best_lsvm_loss:
            best_lsvm_loss = loss
            best_lsvm = clf
    models["SVM_Linear"] = best_lsvm

    # 3. RBF SVM
    best_rsvm, best_rsvm_loss = None, float("inf")
    for c_val in (0.1, 1.0, 10.0):
        clf = CalibratedClassifierCV(
            SVC(kernel="rbf", C=c_val, gamma="scale", random_state=random_state),
            ensemble=False,
        )
        clf.fit(splits.X_train, splits.y_train)
        val_prob = clf.predict_proba(splits.X_val)[:, 1]
        loss = float(log_loss(splits.y_val, val_prob, labels=[0, 1]))
        if loss < best_rsvm_loss:
            best_rsvm_loss = loss
            best_rsvm = clf
    models["SVM_RBF"] = best_rsvm

    # 4. Random Forest
    best_rf, best_rf_loss = None, float("inf")
    for depth in (3, 6, None):
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=depth,
            random_state=random_state,
        )
        clf.fit(splits.X_train, splits.y_train)
        val_prob = clf.predict_proba(splits.X_val)[:, 1]
        loss = float(log_loss(splits.y_val, val_prob, labels=[0, 1]))
        if loss < best_rf_loss:
            best_rf_loss = loss
            best_rf = clf
    models["RandomForest"] = best_rf

    return models


def train_mlp_classifier(
    splits: DatasetSplits,
    hidden_units: tuple[int, ...] = (32, 16),
    lr: float = 1e-3,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 20,
    random_state: int = 42,
) -> tuple[Any, TrainingHistory]:
    """Resolve underfitting first via unregularized capacity check, then address overfitting."""
    underfit_loss_tol = 0.4
    candidate_units = hidden_units

    # Stage 1: Verify training-loss convergence without regularization or early stopping
    probe = NumpyMLPClassifier(
        input_dim=splits.X_train.shape[1],
        hidden_units=candidate_units,
        l2_reg=0.0,
        random_state=random_state,
    )
    probe_hist = probe.fit(
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        lr=lr,
        batch_size=batch_size,
        max_epochs=min(100, max_epochs),
        patience=max_epochs,
        random_state=random_state,
    )
    if min(probe_hist.train_loss) > underfit_loss_tol:
        candidate_units = tuple(u * 2 for u in candidate_units)

    # Stage 2: Train capacity-verified architecture with L2 regularization and early stopping
    mlp = NumpyMLPClassifier(
        input_dim=splits.X_train.shape[1],
        hidden_units=candidate_units,
        l2_reg=1e-4,
        random_state=random_state,
    )
    history = mlp.fit(
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        random_state=random_state,
    )
    return mlp, history


def calibrate_recall_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 1.0,
) -> float:
    """Select the median probability threshold among validation candidates achieving >= target_recall and max F1."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), y_prob_arr]))
    candidates = candidates[(candidates > 0.0) & (candidates < 1.0)]

    feasible: list[tuple[float, float]] = []
    best_fallback_tau = 0.5
    best_fallback_score = (-1.0, -1.0)

    for tau in candidates:
        y_pred = (y_prob_arr >= tau).astype(np.int64)
        rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
        if rec >= target_recall:
            feasible.append((f1, float(tau)))
        if (rec, f1) > best_fallback_score:
            best_fallback_score = (rec, f1)
            best_fallback_tau = float(tau)

    if feasible:
        # Select the median threshold among those maximizing F1 while meeting target_recall,
        # avoiding extreme boundary thresholds that risk False Negatives on unseen conditions.
        max_f1 = max(item[0] for item in feasible)
        top_taus = sorted(tau for f1, tau in feasible if f1 >= max_f1 - 1e-9)
        return float(top_taus[len(top_taus) // 2])

    return best_fallback_tau


def evaluate_classifier(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute confusion matrix, precision/recall/F1, and ROC/AUC at a decision threshold."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    return ClassificationMetrics(
        model_name=model_name,
        threshold=float(threshold),
        confusion_matrix=cm,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        roc_auc=auc,
        fpr=fpr,
        tpr=tpr,
    )
