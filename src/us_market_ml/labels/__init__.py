"""Label generation interfaces for supervised market research tasks."""

from us_market_ml.labels.excess_returns import add_excess_return_vs_benchmark
from us_market_ml.labels.forward_returns import add_forward_returns
from us_market_ml.labels.ranking import add_cross_sectional_rank_labels

__all__ = [
    "add_cross_sectional_rank_labels",
    "add_excess_return_vs_benchmark",
    "add_forward_returns",
]
