"""Signal backtesting interfaces for ETF rotation research."""

from us_market_ml.backtest.benchmark import (
    BenchmarkConfig,
    build_buy_and_hold_benchmark,
    build_equal_weight_benchmark,
    combine_strategy_and_benchmarks,
    summarize_equity_series,
)
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
    "BenchmarkConfig",
    "SignalBacktestConfig",
    "SignalBacktestResult",
    "TopKSignalConfig",
    "build_buy_and_hold_benchmark",
    "build_equal_weight_benchmark",
    "calculate_max_drawdown",
    "combine_strategy_and_benchmarks",
    "generate_top_k_signals",
    "run_signal_backtest",
    "summarize_backtest",
    "summarize_equity_series",
    "summarize_signals",
    "validate_signal_weights",
]
