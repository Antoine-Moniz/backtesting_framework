"""
Framework de Backtesting de Stratégies d'Investissement

Ce framework permet d'évaluer et de comparer différentes stratégies d'investissement
sur des données historiques.
"""

from .strategy import (
    Strategy, 
    strategy_decorator,
    BuyAndHoldStrategy,
    MovingAverageCrossStrategy,
    MeanReversionStrategy
)
from .backtester import Backtester
from .result import Result, compare_results

__version__ = "1.0.0"
__author__ = "Antoine Moniz"

__all__ = [
    "Strategy",
    "strategy_decorator",
    "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy", 
    "MeanReversionStrategy",
    "Backtester",
    "Result",
    "compare_results"
]