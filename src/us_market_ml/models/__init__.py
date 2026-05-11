"""Model interfaces for baseline and neural network research."""

from us_market_ml.models.baseline import (
    BaselineFitResult,
    BaselineModelConfig,
    create_baseline_model,
    fit_baseline_regressor,
    predict_baseline,
    regression_metrics,
)
from us_market_ml.models.mlp import (
    MLPFitResult,
    MLPRegressorConfig,
    MLPRegressorModel,
    fit_mlp_regressor,
    predict_mlp,
)

__all__ = [
    "BaselineFitResult",
    "BaselineModelConfig",
    "MLPFitResult",
    "MLPRegressorConfig",
    "MLPRegressorModel",
    "create_baseline_model",
    "fit_baseline_regressor",
    "fit_mlp_regressor",
    "predict_baseline",
    "predict_mlp",
    "regression_metrics",
]
