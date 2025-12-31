"""Buy and Hold passive investment strategy.

This module implements a simple buy-and-hold strategy that maintains a constant
100% long position in a single asset. It serves as a benchmark for comparing
more sophisticated trading strategies.

The strategy is parameter-free (aside from the asset symbol) and represents
a passive investment approach where the investor buys at the beginning and
holds indefinitely without any rebalancing or active management.

Authors
-------
Mariano Benjamin
Noah Chikhi
Antoine Moniz
"""

from typing import Dict
import pandas as pd
from backtesting_framework import Strategy


class BuyAndHoldStrategy(Strategy):
    """
    Simple Buy and Hold strategy - buy at the beginning and hold forever.
    
    This strategy maintains a 100% long position at all times, making it
    useful as a benchmark for comparing more complex strategies.
    
    Parameters
    ----------
    asset_symbol : str, optional
        Symbol of the asset to trade (default: 'asset')
    
    Examples
    --------
    >>> strategy = BuyAndHoldStrategy(asset_symbol='AAPL')
    >>> strategy.get_position(data, {'AAPL': 0.0})
    {'AAPL': 1.0}
    """
    
    def __init__(self, asset_symbol: str = 'asset'):
        """
        Initialize the Buy and Hold strategy.
        
        Parameters
        ----------
        asset_symbol : str
            Symbol of the asset to trade
        """
        super().__init__(rebalance_frequency="D")
        self.asset_symbol = asset_symbol
    
    @property
    def name(self) -> str:
        """Return strategy name."""
        return "Buy and Hold"
        
    def get_position(self, data: pd.DataFrame, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Return 100% long position for the asset.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical market data
        positions : Dict[str, float]
            Current positions for all assets
            
        Returns
        -------
        Dict[str, float]
            Target positions (always 1.0 for buy and hold)
        """
        return {self.asset_symbol: 1.0}
