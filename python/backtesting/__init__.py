"""Historical backtesting framework (Section 12 of meridian-spec.md)."""

from backtesting.backtesting import (
    BacktestSpec,
    BenchmarkResult,
    load_model_and_produce_weights,
    print_results,
    run_benchmark_suite,
    run_monte_carlo_analysis,
    smoke_test,
)

__all__ = [
    "BacktestSpec",
    "BenchmarkResult",
    "load_model_and_produce_weights",
    "print_results",
    "run_benchmark_suite",
    "run_monte_carlo_analysis",
    "smoke_test",
]
