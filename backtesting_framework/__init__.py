"""Backtesting framework for investment strategies.

This package provides a comprehensive toolkit for backtesting and evaluating
trading strategies on historical market data. It supports both single-asset
and multi-asset portfolios, offering flexible strategy definition through
abstract base classes, position tracking, performance analysis, and rich
visualization capabilities.

Key Components
--------------
Strategy : ABC
    Abstract base class for defining custom trading strategies
Backtester : class
    Main engine for executing strategy backtests with transaction costs
Result : class
    Performance analysis and visualization of backtest results
DataHandler : class
    Utilities for loading and validating financial market data

Features
--------
- Multi-asset portfolio support with dictionary-based position tracking
- Transaction costs and slippage modeling
- Comprehensive performance metrics (Sharpe, Sortino, drawdowns, etc.)
- Multiple visualization backends (matplotlib, seaborn, plotly)
- Strategy comparison and benchmarking tools
- Flexible data loading (CSV, Parquet, DataFrame)

Examples
--------
>>> from backtesting_framework import Backtester, Strategy
>>> import pandas as pd
>>> 
>>> # Define a simple strategy
>>> class MyStrategy(Strategy):
...     def get_position(self, data, positions):
...         return {'asset': 1.0}  # Always long
>>> 
>>> # Load data and run backtest
>>> data = pd.read_csv('market_data.csv', index_col='date', parse_dates=True)
>>> backtester = Backtester(data)
>>> result = backtester.run_backtest(MyStrategy(), initial_capital=100000)
>>> print(result.summary())

Notes
-----
The framework uses a discrete-time event-driven simulation model. Strategies
receive historical data up to each timestamp and return target positions as
a dictionary mapping asset symbols to position sizes (between -1 and 1).

Authors
-------
Mariano Benjamin
Noah Chikhi
Antoine Moniz
"""

from .strategy import Strategy, strategy_decorator
from .backtester import Backtester
from .result import Result, compare_results
from .data_handler import DataHandler

__version__ = "1.0.0"
__author__ = "Mariano Benjamin, Noah Chikhi, Antoine Moniz"
__license__ = "MIT"

__all__ = [
    'Strategy',
    'strategy_decorator',
    'Backtester',
    'Result',
    'compare_results',
    'DataHandler',
]