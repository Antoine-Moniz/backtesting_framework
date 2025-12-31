"""Moving Average Crossover momentum strategy.

This module implements a classic technical analysis strategy based on the
crossover of two moving averages (short and long period). The strategy generates
buy signals when the short-term moving average crosses above the long-term
average, and sell signals on the opposite crossover.

This is a trend-following approach that aims to capture sustained price movements
while filtering out short-term noise through the smoothing effect of moving
averages.

Authors
-------
Mariano Benjamin
Noah Chikhi
Antoine Moniz
"""

from typing import Dict
import pandas as pd
from backtesting_framework import Strategy


class MovingAverageCrossStrategy(Strategy):
    """
    Moving Average Crossover strategy.
    
    Buy when the short moving average crosses above the long moving average,
    and sell (or short) when the short MA crosses below the long MA.
    
    Parameters
    ----------
    asset_symbol : str, optional
        Symbol of the asset to trade (default: 'asset')
    short_window : int, optional
        Period for the short moving average (default: 5)
    long_window : int, optional
        Period for the long moving average (default: 20)
    
    Examples
    --------
    >>> strategy = MovingAverageCrossStrategy(
    ...     asset_symbol='SPY',
    ...     short_window=10,
    ...     long_window=50
    ... )
    >>> positions = strategy.get_position(data, {'SPY': 0.0})
    """
    
    def __init__(self, asset_symbol: str = 'asset', short_window: int = 5, long_window: int = 20):
        """
        Initialize the Moving Average Crossover strategy.
        
        Parameters
        ----------
        asset_symbol : str
            Symbol of the asset to trade
        short_window : int
            Period for the short moving average
        long_window : int
            Period for the long moving average
        """
        if short_window >= long_window:
            raise ValueError(
                f"Short window ({short_window}) must be less than long window ({long_window})"
            )
        
        super().__init__(rebalance_frequency="D")
        self._strategy_name = f"MA Cross ({short_window}/{long_window})"
        self.asset_symbol = asset_symbol
        self.short_window = short_window
        self.long_window = long_window
    
    @property
    def name(self) -> str:
        """Return strategy name."""
        return self._strategy_name
    
    def get_position(self, data: pd.DataFrame, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Determine position based on moving average crossover.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical market data with 'close' column
        positions : Dict[str, float]
            Current positions for all assets
            
        Returns
        -------
        Dict[str, float]
            Target positions: 1.0 (long) if short MA > long MA,
            -1.0 (short) if short MA < long MA,
            0.0 if insufficient data
        """
        if len(data) < self.long_window:
            return {self.asset_symbol: 0.0}  # Insufficient data
            
        # Calculate moving averages
        short_ma = data['close'].rolling(self.short_window).mean().iloc[-1]
        long_ma = data['close'].rolling(self.long_window).mean().iloc[-1]
        
        # Generate position signal
        if short_ma > long_ma:
            return {self.asset_symbol: 1.0}   # Long
        else:
            return {self.asset_symbol: -1.0}  # Short
