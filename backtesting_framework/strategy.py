"""Strategy abstraction layer.

This module defines the abstract `Strategy` base class that all trading
strategies must inherit from, as well as the `strategy_decorator` helper
function for creating simple strategies from functions. The Strategy class
enforces a consistent interface for position generation across the framework.

Notes
-----
All strategies must implement the `get_position` method, which takes historical
data and current positions (as a dictionary) and returns target positions. This
design supports multi-asset portfolios where positions are tracked per asset symbol.

The optional `fit` method allows strategies to perform pre-training or calibration
on historical data before backtesting begins.

Authors
-------
Mariano Benjamin
Noah Chikhi
Antoine Moniz
"""

from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional
import pandas as pd


class Strategy(ABC):
    """
    Abstract base class for defining investment strategies.
    
    This class must be inherited to create custom strategies.
    It defines the required interface that all strategies must implement.
    
    Parameters
    ----------
    name : str, optional
        Name of the strategy. Defaults to class name if not provided.
    rebalance_frequency : str, optional
        Rebalancing frequency: 'D' (daily), 'W' (weekly), 'M' (monthly),
        'Q' (quarterly), or 'Y' (yearly). Default is 'D'.
    
    Attributes
    ----------
    name : str
        The strategy name
    rebalance_frequency : str
        How often the strategy rebalances
    is_fitted : bool
        Whether the strategy has been fitted/trained
    
    Examples
    --------
    >>> class MyStrategy(Strategy):
    ...     def __init__(self):
    ...         super().__init__("My Strategy")
    ...     
    ...     def get_position(self, data, positions):
    ...         # Strategy logic here
    ...         return {'AAPL': 1.0}
    """
    
    def __init__(self, rebalance_frequency: str = "D"):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        rebalance_frequency : str, optional
            Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
        """
        self.rebalance_frequency = rebalance_frequency
        self.is_fitted = False
        
    @abstractmethod
    def get_position(self, data: pd.DataFrame, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Abstract method to determine target positions for all assets.
        
        This method must be implemented by all concrete strategy classes.
        It receives current market data and current positions, and returns
        the target positions for each asset in the portfolio.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical market data available up to the current moment
        positions : Dict[str, float]
            Current positions for all assets, where values are between -1 and 1:
            * 1.0 = 100% long
            * -1.0 = 100% short
            * 0.0 = neutral/no position
            
        Returns
        -------
        Dict[str, float]
            Target positions for each asset (values between -1 and 1)
        
        Examples
        --------
        >>> def get_position(self, data, positions):
        ...     return {'SPY': 0.6, 'AGG': 0.4}  # 60/40 portfolio
        """
        pass
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Optional method to train/calibrate the strategy on historical data.
        
        By default, this method does nothing. It can be overridden by strategies
        that require prior training (e.g., machine learning based strategies).
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data
        
        Examples
        --------
        >>> strategy = MyMLStrategy()
        >>> strategy.fit(training_data)
        >>> strategy.is_fitted
        True
        """
        self.is_fitted = True
    
    def __str__(self) -> str:
        return f"Strategy: {self.name} (rebalance: {self.rebalance_frequency})"
    
    def __repr__(self) -> str:
        return self.__str__()


def strategy_decorator(func: Optional[Callable] = None, *,
                       name: Optional[str] = None, 
                       rebalance_frequency: str = "D"):
    """
    Decorator for strategy methods that validates return values.
    
    Can be used with or without arguments:
    - @strategy_decorator  # Simple usage
    - @strategy_decorator(name="My Strategy")  # With arguments
    
    When used on a method within a Strategy class, it validates that the method
    returns a dictionary with numeric values.
    
    Parameters
    ----------
    func : Callable, optional
        The function to decorate (when used without parentheses)
    name : str, optional
        Strategy name for standalone function decoration
    rebalance_frequency : str, optional
        Rebalancing frequency for standalone function decoration
        
    Returns
    -------
    Callable
        Decorated function with validation
        
    Examples
    --------
    >>> class MyStrategy(Strategy):
    ...     @strategy_decorator
    ...     def get_position(self, data, positions):
    ...         return {'asset': 1.0}
    """
    import functools
    
    def decorator_impl(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            # Validate return value
            if not isinstance(result, dict):
                raise ValueError("get_position must return a dictionary")
            for key, value in result.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Position for '{key}' must be numeric, got {type(value)}")
            return result
        return wrapper
    
    # Handle both @strategy_decorator and @strategy_decorator()
    if func is not None:
        # Used without parentheses: @strategy_decorator
        return decorator_impl(func)
    else:
        # Used with parentheses: @strategy_decorator()
        return decorator_impl