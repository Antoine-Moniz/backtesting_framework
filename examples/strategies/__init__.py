"""
Example investment strategies for the backtesting framework.

This package contains concrete implementations of various trading strategies
that can be used as examples or templates for creating custom strategies.
"""

from .buy_and_hold import BuyAndHoldStrategy
from .moving_average_cross import MovingAverageCrossStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = [
    'BuyAndHoldStrategy',
    'MovingAverageCrossStrategy',
    'MeanReversionStrategy',
]
