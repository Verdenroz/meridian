"""Classical spine: vol forecasts, shrunk correlations, regime filter, allocator v0.

Estimator choices here are the incumbents — RiskMetrics EWMA, Ledoit–Wolf
constant-correlation shrinkage, a Gaussian HMM filtered causally. Challengers
(GARCH-class, learned models) displace them only by winning in the harness.
"""

from spine.allocator import allocate
from spine.correlation import ledoit_wolf_constant_corr, rolling_correlation
from spine.regime import GaussianHMM
from spine.strategy import SpineStrategy
from spine.vol import ewma_vol_series, forecast_correlation, forward_realized_vol

__all__ = [
    "GaussianHMM",
    "SpineStrategy",
    "allocate",
    "ewma_vol_series",
    "forecast_correlation",
    "forward_realized_vol",
    "ledoit_wolf_constant_corr",
    "rolling_correlation",
]
