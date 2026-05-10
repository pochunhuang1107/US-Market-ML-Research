"""Signal backtesting interfaces for ETF rotation research."""

from us_market_ml.backtest.signal_backtester import (
    SignalBacktestConfig,
    SignalBacktestResult,
    calculate_max_drawdown,
    run_signal_backtest,
    summarize_backtest,
)
from us_market_ml.backtest.signals import (
    TopKSignalConfig,
    generate_top_k_signals,
    summarize_signals,
    validate_signal_weights,
)

__all__ = [
    "SignalBacktestConfig",
    "SignalBacktestResult",
    "TopKSignalConfig",
    "calculate_max_drawdown",
    "generate_top_k_signals",
    "run_signal_backtest",
    "summarize_backtest",
    "summarize_signals",
    "validate_signal_weights",
]
