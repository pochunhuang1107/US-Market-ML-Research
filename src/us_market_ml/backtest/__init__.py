"""Signal backtesting interfaces for ETF rotation research."""

from us_market_ml.backtest.benchmark import (
    BenchmarkConfig,
    build_buy_and_hold_benchmark,
    build_equal_weight_benchmark,
    combine_strategy_and_benchmarks,
    stitch_test_equity_curves,
    summarize_equity_series,
)
from us_market_ml.backtest.concentration import (
    build_concentration_report,
    calculate_hhi,
    effective_number_from_hhi,
    summarize_contribution_concentration,
    summarize_selection_concentration,
)
from us_market_ml.backtest.holdings_report import (
    build_daily_selection_summary,
    build_position_contribution_frame,
    build_selected_holdings_frame,
    summarize_holdings,
    summarize_position_contributions,
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
    "build_concentration_report",
    "build_daily_selection_summary",
    "build_equal_weight_benchmark",
    "build_position_contribution_frame",
    "build_selected_holdings_frame",
    "calculate_hhi",
    "calculate_max_drawdown",
    "combine_strategy_and_benchmarks",
    "effective_number_from_hhi",
    "generate_top_k_signals",
    "run_signal_backtest",
    "summarize_backtest",
    "summarize_contribution_concentration",
    "summarize_equity_series",
    "summarize_holdings",
    "summarize_position_contributions",
    "summarize_selection_concentration",
    "summarize_signals",
    "stitch_test_equity_curves",
    "validate_signal_weights",
]
