"""Signal backtesting interfaces for ETF rotation research."""

from us_market_ml.backtest.signals import (
    TopKSignalConfig,
    generate_top_k_signals,
    summarize_signals,
    validate_signal_weights,
)

__all__ = [
    "TopKSignalConfig",
    "generate_top_k_signals",
    "summarize_signals",
    "validate_signal_weights",
]
