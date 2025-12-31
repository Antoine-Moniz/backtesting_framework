"""Mean Reversion strategy using Bollinger Bands.

This module implements a mean reversion strategy based on Bollinger Bands,
which identify potential overbought and oversold conditions. The strategy
assumes that prices tend to revert to their mean after extreme movements.

Buy signals are generated when price falls below the lower Bollinger Band
(oversold), and sell signals occur when price exceeds the upper band (overbought).
The strategy remains neutral when price is within the bands.

Authors
-------
Mariano Benjamin
Noah Chikhi
Antoine Moniz
"""

from typing import Dict
import pandas as pd
from backtesting_framework import Strategy


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion strategy using Bollinger Bands.
    
    Buy when price is below the lower Bollinger Band (oversold),
    sell when price is above the upper Bollinger Band (overbought),
    and stay neutral when price is within the bands.
    
    Parameters
    ----------
    asset_symbol : str, optional
        Symbol of the asset to trade (default: 'asset')
    window : int, optional
        Period for calculating moving average and standard deviation (default: 20)
    num_std : float, optional
        Number of standard deviations for Bollinger Bands (default: 2.0)
    
    Examples
    --------
    >>> strategy = MeanReversionStrategy(
    ...     asset_symbol='AAPL',
    ...     window=20,
    ...     threshold=2.0
    ... )
    >>> positions = strategy.get_position(data, {'AAPL': 0.0})
    """
    
    def __init__(self, asset_symbol: str = 'asset', window: int = 20, threshold: float = 2.0):
        """
        Initialize the Mean Reversion strategy.
        
        Parameters
        ----------
        asset_symbol : str
            Symbol of the asset to trade
        window : int
            Period for moving average calculation
        threshold : float
            Number of standard deviations for bands
        """
        super().__init__(rebalance_frequency="D")
        self._strategy_name = f"Mean Reversion (BB {window}, {threshold}σ)"
        self.asset_symbol = asset_symbol
        self.window = window
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        """Return strategy name."""
        return self._strategy_name
    
    def get_position(self, data: pd.DataFrame, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Determine position based on Bollinger Bands for mean reversion.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical market data with 'close' column
        positions : Dict[str, float]
            Current positions for all assets
            
        Returns
        -------
        Dict[str, float]
            Target positions: 
            -1.0 (short) if price > upper band,
            1.0 (long) if price < lower band,
            0.0 (neutral) if within bands or insufficient data
        """
        if len(data) < self.window:
            return {self.asset_symbol: 0.0}  # Insufficient data
            
        closes = data['close']
        ma = closes.rolling(self.window).mean().iloc[-1]
        std = closes.rolling(self.window).std().iloc[-1]
        current_price = closes.iloc[-1]
        
        upper_band = ma + (self.threshold * std)
        lower_band = ma - (self.threshold * std)
        
        if current_price > upper_band:
            return {self.asset_symbol: 0.0}  # Overbought, don't buy
        elif current_price < lower_band:
            return {self.asset_symbol: 1.0}   # Oversold, buy
        else:
            return {self.asset_symbol: 0.5}   # Neutral, moderate position

