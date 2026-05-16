"""Walk-forward and time-series split interfaces."""

from us_market_ml.splits.sliding_window import (
    make_sliding_window_folds,
    make_sliding_window_folds_for_train_windows,
)
from us_market_ml.splits.walk_forward import (
    PurgeConfig,
    WalkForwardFold,
    assign_purged_split_labels,
    assign_split_labels,
    make_annual_walk_forward_folds,
    purge_fold_boundaries,
    summarize_purged_split,
)

__all__ = [
    "PurgeConfig",
    "WalkForwardFold",
    "assign_purged_split_labels",
    "assign_split_labels",
    "make_annual_walk_forward_folds",
    "make_sliding_window_folds",
    "make_sliding_window_folds_for_train_windows",
    "purge_fold_boundaries",
    "summarize_purged_split",
]
