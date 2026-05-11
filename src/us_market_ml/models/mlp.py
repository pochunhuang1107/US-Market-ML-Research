"""Optional PyTorch tabular MLP regressor for ETF rotation research.

This module keeps PyTorch as an optional dependency. Importing
``us_market_ml.models`` should not import torch; torch is loaded only when the
MLP fit or predict helpers are called.
"""

from __future__ import annotations

import copy
import importlib
import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from us_market_ml.models.baseline import regression_metrics

_TORCH_INSTALL_MESSAGE = (
    'PyTorch is required for MLP models. Install with python -m pip install -e ".[dev,nn]"'
)


@dataclass(frozen=True)
class MLPRegressorConfig:
    """Configuration for the optional PyTorch tabular MLP regressor."""

    hidden_units: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    random_state: int = 42
    device: str = "cpu"
    validation_fraction: float | None = None
    scale_target: bool = False

    def __post_init__(self) -> None:
        hidden_units = tuple(self.hidden_units)
        object.__setattr__(self, "hidden_units", hidden_units)

        if not hidden_units or any(unit <= 0 for unit in hidden_units):
            raise ValueError("hidden_units must be non-empty and contain only positive integers.")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be >= 0 and < 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.validation_fraction is not None and not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be > 0 and < 1 when provided.")


@dataclass(frozen=True)
class MLPFitResult:
    """Summary of a tabular MLP supervised regression fit.

    ``best_validation_loss`` is the MSE optimized during training. When target
    scaling is enabled, this loss is measured in scaled target space; fit
    metrics remain in the original target units.
    """

    target_col: str
    feature_cols: list[str]
    train_rows: int
    validation_rows: int | None
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float] | None
    epochs_trained: int
    best_validation_loss: float | None


@dataclass(frozen=True)
class MLPRegressorModel:
    """Fitted MLP model plus train-only preprocessing statistics."""

    torch_model: Any
    feature_means: pd.Series
    feature_stds: pd.Series
    feature_cols: list[str]
    config: MLPRegressorConfig
    target_mean: float | None
    target_std: float | None
    scale_target: bool


def fit_mlp_regressor(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    config: MLPRegressorConfig | None = None,
    validation_df: pd.DataFrame | None = None,
) -> tuple[MLPRegressorModel, MLPFitResult]:
    """Fit a small PyTorch MLP regressor on explicit tabular columns.

    Rows with missing selected feature or target values are dropped before
    fitting. Feature scaling is learned from training rows only, then reused
    for validation and prediction.
    """

    torch, nn, data_utils = _require_torch()
    resolved_config = config if config is not None else MLPRegressorConfig()
    _set_deterministic_seeds(torch, resolved_config.random_state)
    _validate_model_columns(train_df, feature_cols, target_col)

    train_model_df = _drop_missing_model_rows(train_df, feature_cols, target_col)
    if train_model_df.empty:
        raise ValueError("No usable training rows remain after dropping missing model values.")

    fit_train_df, fit_validation_df, validation_rows = _resolve_validation_frame(
        train_model_df=train_model_df,
        feature_cols=feature_cols,
        target_col=target_col,
        config=resolved_config,
        validation_df=validation_df,
    )
    if fit_train_df.empty:
        raise ValueError("No usable training rows remain after creating the validation split.")

    feature_means, feature_stds = _fit_feature_standardizer(fit_train_df, feature_cols)
    x_train = _standardized_features(fit_train_df, feature_cols, feature_means, feature_stds)
    target_mean, target_std = _fit_target_standardizer(fit_train_df[target_col], resolved_config)
    y_train_raw = _to_writable_float_array(fit_train_df[target_col]).reshape(-1, 1)
    y_train = _scale_targets(y_train_raw, target_mean, target_std)

    x_validation: npt.NDArray[np.float32] | None = None
    y_validation: npt.NDArray[np.float32] | None = None
    if fit_validation_df is not None and not fit_validation_df.empty:
        x_validation = _standardized_features(
            fit_validation_df,
            feature_cols,
            feature_means,
            feature_stds,
        )
        y_validation_raw = _to_writable_float_array(fit_validation_df[target_col]).reshape(-1, 1)
        y_validation = _scale_targets(y_validation_raw, target_mean, target_std)

    device = torch.device(resolved_config.device)
    model = _create_tabular_mlp_regressor(
        input_dim=len(feature_cols),
        hidden_units=resolved_config.hidden_units,
        dropout=resolved_config.dropout,
        torch=torch,
        nn=nn,
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )

    train_dataset = data_utils.TensorDataset(
        torch.as_tensor(_to_writable_float_array(x_train), dtype=torch.float32),
        torch.as_tensor(_to_writable_float_array(y_train), dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(resolved_config.random_state)
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=resolved_config.batch_size,
        shuffle=True,
        generator=generator,
    )

    validation_tensors: tuple[Any, Any] | None = None
    if x_validation is not None and y_validation is not None:
        validation_tensors = (
            torch.as_tensor(_to_writable_float_array(x_validation), dtype=torch.float32).to(device),
            torch.as_tensor(_to_writable_float_array(y_validation), dtype=torch.float32).to(device),
        )

    epochs_trained, best_validation_loss = _train_model(
        model=model,
        train_loader=train_loader,
        validation_tensors=validation_tensors,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=resolved_config,
        torch=torch,
        device=device,
    )

    train_predictions_scaled = _predict_array(model, x_train, torch, device)
    train_predictions = _inverse_scale_predictions(
        train_predictions_scaled,
        target_mean,
        target_std,
    )
    train_metrics = regression_metrics(fit_train_df[target_col], train_predictions)

    validation_metrics: dict[str, float] | None = None
    if fit_validation_df is not None and not fit_validation_df.empty and x_validation is not None:
        validation_predictions_scaled = _predict_array(model, x_validation, torch, device)
        validation_predictions = _inverse_scale_predictions(
            validation_predictions_scaled,
            target_mean,
            target_std,
        )
        validation_metrics = regression_metrics(
            fit_validation_df[target_col],
            validation_predictions,
        )

    model_wrapper = MLPRegressorModel(
        torch_model=model,
        feature_means=feature_means,
        feature_stds=feature_stds,
        feature_cols=list(feature_cols),
        config=resolved_config,
        target_mean=target_mean,
        target_std=target_std,
        scale_target=resolved_config.scale_target,
    )
    result = MLPFitResult(
        target_col=target_col,
        feature_cols=list(feature_cols),
        train_rows=len(fit_train_df),
        validation_rows=validation_rows,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        epochs_trained=epochs_trained,
        best_validation_loss=best_validation_loss,
    )
    return model_wrapper, result


def predict_mlp(
    model: MLPRegressorModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Return ``df`` with MLP predictions for rows with complete features."""

    torch, _, _ = _require_torch()
    _validate_feature_columns(df, feature_cols)
    if list(feature_cols) != model.feature_cols:
        raise ValueError("feature_cols must match the columns used to fit the MLP model.")

    out = df.copy()
    out[prediction_col] = np.nan
    usable_mask = out[feature_cols].notna().all(axis=1)
    if usable_mask.any():
        device = torch.device(model.config.device)
        x = _standardized_features(
            out.loc[usable_mask],
            feature_cols,
            model.feature_means,
            model.feature_stds,
        )
        scaled_predictions = _predict_array(model.torch_model, x, torch, device)
        out.loc[usable_mask, prediction_col] = _inverse_scale_predictions(
            scaled_predictions,
            model.target_mean,
            model.target_std,
        )
    return out


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        data_utils = importlib.import_module("torch.utils.data")
    except ModuleNotFoundError as exc:
        raise ImportError(_TORCH_INSTALL_MESSAGE) from exc
    return torch, nn, data_utils


def _create_tabular_mlp_regressor(
    input_dim: int,
    hidden_units: tuple[int, ...],
    dropout: float,
    torch: Any,
    nn: Any,
) -> Any:
    class TabularMLPRegressor(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            current_dim = input_dim
            for hidden_dim in hidden_units:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, features: Any) -> Any:
            return self.network(features)

    return TabularMLPRegressor()


def _train_model(
    model: Any,
    train_loader: Any,
    validation_tensors: tuple[Any, Any] | None,
    optimizer: Any,
    loss_fn: Any,
    config: MLPRegressorConfig,
    torch: Any,
    device: Any,
) -> tuple[int, float | None]:
    best_validation_loss: float | None = None
    best_state_dict: dict[str, Any] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(config.max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        epochs_trained = epoch + 1
        if validation_tensors is None:
            continue

        validation_loss = _validation_loss(model, validation_tensors, loss_fn, torch)
        if best_validation_loss is None or validation_loss < best_validation_loss - 1e-12:
            best_validation_loss = validation_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return epochs_trained, best_validation_loss


def _validation_loss(
    model: Any,
    validation_tensors: tuple[Any, Any],
    loss_fn: Any,
    torch: Any,
) -> float:
    model.eval()
    validation_x, validation_y = validation_tensors
    with torch.no_grad():
        return float(loss_fn(model(validation_x), validation_y).detach().cpu().item())


def _predict_array(
    model: Any,
    x: npt.NDArray[np.float32],
    torch: Any,
    device: Any,
) -> npt.NDArray[np.float32]:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(_to_writable_float_array(x), dtype=torch.float32).to(device)
        predictions = model(tensor).detach().cpu().numpy().reshape(-1)
        return cast(npt.NDArray[np.float32], predictions)


def _resolve_validation_frame(
    train_model_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: MLPRegressorConfig,
    validation_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, int | None]:
    if validation_df is not None:
        _validate_model_columns(validation_df, feature_cols, target_col)
        validation_model_df = _drop_missing_model_rows(validation_df, feature_cols, target_col)
        return train_model_df, validation_model_df, len(validation_model_df)

    if config.validation_fraction is None:
        return train_model_df, None, None

    validation_rows = int(round(len(train_model_df) * config.validation_fraction))
    validation_rows = max(1, validation_rows)
    if validation_rows >= len(train_model_df):
        return train_model_df, None, 0

    split_index = len(train_model_df) - validation_rows
    return (
        train_model_df.iloc[:split_index].copy(),
        train_model_df.iloc[split_index:].copy(),
        validation_rows,
    )


def _fit_feature_standardizer(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, pd.Series]:
    feature_means = df[feature_cols].mean()
    feature_stds = df[feature_cols].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return feature_means, feature_stds


def _fit_target_standardizer(
    target: pd.Series,
    config: MLPRegressorConfig,
) -> tuple[float | None, float | None]:
    if not config.scale_target:
        return None, None

    target_mean = float(target.mean())
    target_std = float(target.std(ddof=0))
    if target_std == 0.0 or np.isnan(target_std):
        target_std = 1.0
    return target_mean, target_std


def _scale_targets(
    target: npt.NDArray[np.float32],
    target_mean: float | None,
    target_std: float | None,
) -> npt.NDArray[np.float32]:
    if target_mean is None or target_std is None:
        return target
    return _to_writable_float_array((target - target_mean) / target_std)


def _inverse_scale_predictions(
    predictions: npt.NDArray[np.float32],
    target_mean: float | None,
    target_std: float | None,
) -> npt.NDArray[np.float32]:
    if target_mean is None or target_std is None:
        return predictions
    return _to_writable_float_array((predictions * target_std) + target_mean)


def _standardized_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_means: pd.Series,
    feature_stds: pd.Series,
) -> npt.NDArray[np.float32]:
    scaled = (df[feature_cols] - feature_means) / feature_stds
    return _to_writable_float_array(scaled)


def _to_writable_float_array(values: Any) -> npt.NDArray[np.float32]:
    return np.ascontiguousarray(values, dtype=np.float32).copy()


def _set_deterministic_seeds(torch: Any, random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def _drop_missing_model_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    return df.dropna(subset=[*feature_cols, target_col]).copy()


def _validate_model_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    _validate_feature_columns(df, feature_cols)
    if target_col not in df.columns:
        raise ValueError(f"Input DataFrame is missing selected model columns: {target_col}")


def _validate_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    if not feature_cols:
        raise ValueError("At least one feature column must be selected.")

    missing_columns = [column for column in feature_cols if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input DataFrame is missing selected model columns: {missing}")


__all__ = [
    "MLPFitResult",
    "MLPRegressorConfig",
    "MLPRegressorModel",
    "fit_mlp_regressor",
    "predict_mlp",
]
