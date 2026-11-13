"""Phase 6: Task 3 top-node displacement prediction (system identification / regression)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from .ingestion import SimulationRun
from .partitioning import DatasetSplits
from .preprocessing import mask_initial_steps


@dataclass(frozen=True)
class RegressionMetrics:
    """Continuous displacement prediction performance summary."""

    model_name: str
    mse: float
    rmse: float
    r2: float
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None


def _smooth_curve(values: list[float], window: int = 5) -> list[float]:
    """Compute causal moving average for regression loss trajectories."""
    if not values:
        return []
    return [float(np.mean(values[max(0, i - window + 1) : i + 1])) for i in range(len(values))]


class NumpyMLPRegressor:
    """Compact ReLU feed-forward neural network with linear output trained on MSE loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...] = (32, 16),
        l2_reg: float = 1e-7,
        random_state: int = 42,
    ) -> None:
        self.layer_dims = (input_dim, *hidden_units, 1)
        self.l2_reg = l2_reg
        self._y_mean = 0.0
        self._y_std = 1.0
        rng = np.random.default_rng(random_state)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for idx in range(len(self.layer_dims) - 1):
            fan_in = self.layer_dims[idx]
            fan_out = self.layer_dims[idx + 1]
            scale = np.sqrt(2.0 / fan_in) if idx < len(self.layer_dims) - 2 else np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations: list[np.ndarray] = []
        a = X
        n_layers = len(self.weights)
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            pre_activations.append(z)
            # ReLU for hidden layers, linear identity for final output unit
            a = np.maximum(0.0, z) if idx < n_layers - 1 else z
            activations.append(a)
        return activations, pre_activations

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous displacement in meters."""
        activations, _ = self._forward(np.asarray(X, dtype=np.float64))
        return activations[-1].ravel() * self._y_std + self._y_mean

    def compute_mse_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Squared Error loss in physical units (m^2)."""
        preds = self.predict(X)
        return float(np.mean((y.astype(np.float64) - preds) ** 2))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 300,
        patience: int = 20,
        random_state: int = 42,
    ) -> dict[str, list[float]]:
        """Train with mini-batch Adam on MSE loss and early stopping on validation MSE."""
        rng = np.random.default_rng(random_state)
        n_samples = X_train.shape[0]
        y_train_arr = y_train.astype(np.float64)
        self._y_mean = float(np.mean(y_train_arr))
        self._y_std = float(np.std(y_train_arr)) or 1.0
        y_train_norm = ((y_train_arr - self._y_mean) / self._y_std).reshape(-1, 1)

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

        train_losses: list[float] = []
        val_losses: list[float] = []

        for _epoch in range(1, max_epochs + 1):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_idx = perm[start : start + batch_size]
                Xb = X_train[batch_idx]
                yb = y_train_norm[batch_idx]
                bs = float(len(batch_idx))

                activations, pre_activations = self._forward(Xb)
                dz = (2.0 * (activations[-1] - yb)) / bs

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
                    self.weights[idx] -= lr * (m_W[idx] / bc1) / (np.sqrt(v_W[idx] / bc2) + eps)

                    m_b[idx] = beta1 * m_b[idx] + (1.0 - beta1) * grad_b[idx]
                    v_b[idx] = beta2 * v_b[idx] + (1.0 - beta2) * (grad_b[idx] ** 2)
                    self.biases[idx] -= lr * (m_b[idx] / bc1) / (np.sqrt(v_b[idx] / bc2) + eps)

            tr_loss = self.compute_mse_loss(X_train, y_train)
            va_loss = self.compute_mse_loss(X_val, y_val)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)

            if va_loss < best_val_loss - 1e-12:
                best_val_loss = va_loss
                best_weights = [W.copy() for W in self.weights]
                best_biases = [b.copy() for b in self.biases]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        self.weights = best_weights
        self.biases = best_biases

        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "smoothed_train_loss": _smooth_curve(train_losses),
            "smoothed_val_loss": _smooth_curve(val_losses),
        }


def build_regression_dataset(
    runs: Sequence[SimulationRun],
    target_channel: str = "top_U1",
    lag_steps: int = 5,
    include_concurrent_load_node: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build lagged historical and concurrent cross-arm features to predict top-node displacement."""
    if lag_steps < 1:
        raise ValueError("lag_steps must be >= 1")

    X_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    run_ids: list[str] = []

    concurrent_cols = ["arm_U1", "arm_U2", "arm_U3", "arm_U_mag"]

    for run in runs:
        masked = mask_initial_steps(run)
        ch = masked.channels
        if target_channel not in ch.columns:
            raise ValueError(f"Unknown target_channel: {target_channel}")

        target_series = ch[target_channel].to_numpy(dtype=np.float64)
        arm_mat = ch[concurrent_cols].to_numpy(dtype=np.float64)

        for t_idx in range(lag_steps, len(ch)):
            # Historical target displacements at t* - 1, ..., t* - lag_steps
            lags = target_series[t_idx - lag_steps : t_idx][::-1]
            if include_concurrent_load_node:
                feat = np.concatenate([lags, arm_mat[t_idx]])
            else:
                feat = lags
            X_rows.append(feat)
            y_vals.append(float(target_series[t_idx]))
            run_ids.append(run.run_id)

    return np.vstack(X_rows), np.asarray(y_vals, dtype=np.float64), run_ids


def train_classical_regressors(splits: DatasetSplits, random_state: int = 42) -> dict[str, Any]:
    """Train and validation-tune Ridge Regression and Random Forest Regressor."""
    models: dict[str, Any] = {}

    best_ridge, best_ridge_mse = None, float("inf")
    for alpha in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0):
        reg = Ridge(alpha=alpha, random_state=random_state)
        reg.fit(splits.X_train, splits.y_train)
        val_mse = float(mean_squared_error(splits.y_val, reg.predict(splits.X_val)))
        if val_mse < best_ridge_mse:
            best_ridge_mse = val_mse
            best_ridge = reg
    models["RidgeRegression"] = best_ridge

    best_rf, best_rf_mse = None, float("inf")
    for depth in (5, 10, None):
        reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=depth,
            random_state=random_state,
        )
        reg.fit(splits.X_train, splits.y_train)
        val_mse = float(mean_squared_error(splits.y_val, reg.predict(splits.X_val)))
        if val_mse < best_rf_mse:
            best_rf_mse = val_mse
            best_rf = reg
    models["RandomForestRegressor"] = best_rf

    return models


def train_mlp_regressor(
    splits: DatasetSplits,
    hidden_units: tuple[int, ...] = (32, 16),
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 300,
    patience: int = 20,
    random_state: int = 42,
) -> tuple[Any, dict[str, list[float]]]:
    """Train a compact ReLU MLP regressor with linear output on MSE loss."""
    mlp = NumpyMLPRegressor(
        input_dim=splits.X_train.shape[1],
        hidden_units=hidden_units,
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


def evaluate_regressor(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute test MSE, RMSE (meters), and R^2 score."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))
    return RegressionMetrics(model_name=model_name, mse=mse, rmse=rmse, r2=r2, y_true=y_t, y_pred=y_p)
