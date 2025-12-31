"""Backtesting engine for investment strategies.

This module contains the `Backtester` class, which is the main engine for
executing backtests of trading strategies on historical market data. It handles
portfolio construction, trade execution, transaction costs, slippage modeling,
and multi-asset portfolio tracking. The backtester maintains a complete history
of portfolio values, positions, and trades throughout the simulation.

Notes
-----
The backtester uses a discrete-time simulation approach, iterating through
historical data one timestamp at a time. At each step, it queries the strategy
for target positions and executes trades as needed. Portfolio value is calculated
by summing cash plus the market value of all asset positions.

Transaction costs and slippage are applied to each trade independently. The
framework supports both single-asset and multi-asset portfolios through a
dictionary-based position tracking system.

Authors
-------
Mariano Benjamin
Noah Chikhi
Antoine Moniz
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any, List
from pathlib import Path
import warnings

from .strategy import Strategy
from .result import Result
from .data_handler import DataHandler


class Backtester:
    """
    Main class for executing investment strategy backtests.
    
    This class takes historical data and allows executing strategies
    to evaluate their performance. Supports multi-asset portfolios.
    """
    
    def __init__(self, data: Union[pd.DataFrame, str, Path], 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0001):
        """
        Initialize the backtester with historical data.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, str, Path]
            Historical data or path to CSV/Parquet file.
        initial_capital : float, default=100000
            Initial capital for the backtest.
        transaction_cost : float, default=0.001
            Transaction cost as a percentage (0.001 = 0.1%).
        slippage : float, default=0.0001
            Slippage as a percentage (0.0001 = 0.01%).
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Load data using DataHandler
        data_handler = DataHandler()
        self.data = data_handler.load(data)
        
    def run_backtest(self, strategy: Strategy, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    benchmark: Optional[str] = None,
                    asset_symbols: Optional[List[str]] = None) -> Result:
        """
        Execute a backtest with the given strategy.
        
        Parameters
        ----------
        strategy : Strategy
            Strategy to test.
        start_date : str, optional
            Start date (format YYYY-MM-DD).
        end_date : str, optional
            End date (format YYYY-MM-DD).
        benchmark : str, optional
            Column name to use as benchmark.
        asset_symbols : List[str], optional
            List of asset symbols for multi-asset portfolios.
            Defaults to ['asset'] for single-asset backtests.
            
        Returns
        -------
        Result
            Backtest results.
        """
        # Prepare data
        data_subset = self._prepare_data_subset(start_date, end_date)
        
        # Determine asset symbols (default to single asset)
        if asset_symbols is None:
            asset_symbols = ['asset']
        
        # Train strategy if necessary
        if hasattr(strategy, 'fit') and not strategy.is_fitted:
            # Use first 60% of data for training
            train_size = int(len(data_subset) * 0.6)
            train_data = data_subset.iloc[:train_size]
            strategy.fit(train_data)
            
        # Variables for backtest (multi-asset support)
        positions_history = []  # List of position dicts over time
        portfolio_values = [self.initial_capital]
        trades = []
        current_positions = {symbol: 0.0 for symbol in asset_symbols}  # Dict[str, float]
        current_shares = {symbol: 0.0 for symbol in asset_symbols}  # Dict[str, float]
        cash = self.initial_capital
        
        # Track cash and shares over time
        cash_history = []
        shares_history = []  # List of shares dicts over time
        
        # Calculate rebalancing dates
        rebalance_dates = self._get_rebalance_dates(data_subset, strategy.rebalance_frequency)
        
        # Main backtest loop
        for i, (date, row) in enumerate(data_subset.iterrows()):
            # Historical data available up to this date
            historical_data = data_subset.iloc[:i+1]
            
            # Check if it's a rebalancing date
            should_rebalance = date in rebalance_dates or i == 0
            
            if should_rebalance and len(historical_data) > 1:
                # Request target positions from strategy
                try:
                    new_positions = strategy.get_position(historical_data, current_positions.copy())
                    # Enforce long-only constraint (0-100% allocation per asset)
                    new_positions = {symbol: np.clip(pos, 0.0, 1.0) 
                                   for symbol, pos in new_positions.items()}
                except Exception as e:
                    warnings.warn(f"Strategy error at date {date}: {e}")
                    new_positions = current_positions.copy()
                    
                # Execute trades when positions change or on rebalance dates
                # to maintain target allocations despite price drift
                for symbol in asset_symbols:
                    old_pos = current_positions.get(symbol, 0.0)
                    new_pos = new_positions.get(symbol, 0.0)
                    
                    # Execute trade if target position changed or rebalancing required
                    if abs(new_pos - old_pos) > 0.001 or should_rebalance:
                        # Retrieve current price (column names normalized to lowercase by DataHandler)
                        if len(asset_symbols) == 1:
                            price = row['close']
                        else:
                            price_col = f'{symbol.lower()}_close'
                            price = row[price_col] if price_col in row.index else row['close']
                        
                        # Construct price dictionary for portfolio value calculation
                        current_prices = {}
                        for s in asset_symbols:
                            if len(asset_symbols) == 1:
                                current_prices[s] = row['close']
                            else:
                                price_col = f'{s.lower()}_close'
                                current_prices[s] = row[price_col] if price_col in row.index else row['close']
                        
                        trade_info = self._execute_trade(
                            symbol, old_pos, new_pos, price, 
                            cash, current_shares.get(symbol, 0.0), date,
                            current_shares, current_prices,
                            force_rebalance=should_rebalance
                        )
                        if trade_info:
                            trades.append(trade_info)
                            cash = trade_info['cash_after']
                            current_shares[symbol] = trade_info['shares_after']
                            current_positions[symbol] = new_pos
            
            # Compute total portfolio value from cash and asset holdings
            assets_value = 0.0
            for symbol in asset_symbols:
                shares = current_shares.get(symbol, 0.0)
                # Retrieve price using normalized column name
                if len(asset_symbols) == 1:
                    price = row['close']
                else:
                    price_col = f'{symbol.lower()}_close'
                    price = row[price_col] if price_col in row.index else row['close']
                assets_value += shares * price
            
            portfolio_value = cash + assets_value
            portfolio_values.append(portfolio_value)
            positions_history.append(current_positions.copy())
            cash_history.append(cash)
            shares_history.append(current_shares.copy())
            
        # Construct results DataFrame with portfolio state history
        results_df = pd.DataFrame({
            'date': data_subset.index,
            'close': data_subset['close'].values,
            'position': [pos.get(asset_symbols[0], 0.0) for pos in positions_history],  # Primary asset allocation
            'portfolio_value': portfolio_values[1:],  # Exclude initial value
            'cash': cash_history
        })
        results_df.set_index('date', inplace=True)
        
        # Append individual asset positions and share counts
        for symbol in asset_symbols:
            # Column names use lowercase convention for consistency
            symbol_lower = symbol.lower()
            results_df[f'position_{symbol_lower}'] = [pos.get(symbol, 0.0) for pos in positions_history]
            results_df[f'{symbol_lower}_shares'] = [shares.get(symbol, 0.0) for shares in shares_history]
        
        # Compute period and cumulative returns
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_returns'] = (results_df['portfolio_value'] / self.initial_capital) - 1
        
        # Establish benchmark comparison (defaults to buy-and-hold)
        if benchmark and benchmark in data_subset.columns:
            benchmark_data = data_subset[benchmark]
        else:
            benchmark_data = data_subset['close']
            
        benchmark_returns = benchmark_data.pct_change()
        benchmark_cumulative = (benchmark_data / benchmark_data.iloc[0]) - 1
        
        results_df['benchmark_returns'] = benchmark_returns
        results_df['benchmark_cumulative'] = benchmark_cumulative
        
        # Package backtest results into Result object
        return Result(
            strategy=strategy,
            results_df=results_df,
            trades=trades,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage,
            asset_symbols=asset_symbols
        )
    
    def _prepare_data_subset(self, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Prepare a data subset for the backtest.
        
        Parameters
        ----------
        start_date : str, optional
            Start date for filtering.
        end_date : str, optional
            End date for filtering.
            
        Returns
        -------
        pd.DataFrame
            Filtered data subset.
        """
        data_subset = self.data.copy()
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            data_subset = data_subset[data_subset.index >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            data_subset = data_subset[data_subset.index <= end_date]
            
        if len(data_subset) == 0:
            raise ValueError("No data available for the specified period")
            
        return data_subset
    
    def _get_rebalance_dates(self, data: pd.DataFrame, frequency: str) -> pd.DatetimeIndex:
        """
        Calculate rebalancing dates based on frequency.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with datetime index.
        frequency : str
            Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y').
            
        Returns
        -------
        pd.DatetimeIndex
            Rebalancing dates.
        """
        if frequency == 'D':
            return data.index  # Every day
        elif frequency == 'W':
            return pd.date_range(start=data.index[0], end=data.index[-1], freq='W')
        elif frequency == 'M':
            return pd.date_range(start=data.index[0], end=data.index[-1], freq='ME')
        elif frequency == 'Q':
            return pd.date_range(start=data.index[0], end=data.index[-1], freq='Q')
        elif frequency == 'Y':
            return pd.date_range(start=data.index[0], end=data.index[-1], freq='Y')
        else:
            return data.index  # Default: daily
    
    def _execute_trade(self, symbol: str, old_position: float, new_position: float, price: float,
                      cash: float, current_shares: float, date: pd.Timestamp,
                      all_shares: Dict[str, float], current_prices: Dict[str, float],
                      force_rebalance: bool = False) -> Optional[Dict]:
        """
        Execute a trade and calculate costs for a specific asset.
        
        Parameters
        ----------
        symbol : str
            Asset symbol being traded.
        old_position : float
            Previous position (-1 to 1).
        new_position : float
            New target position (-1 to 1).
        price : float
            Current market price for this asset.
        cash : float
            Available cash.
        current_shares : float
            Current number of shares held for this asset.
        date : pd.Timestamp
            Trade date.
        all_shares : Dict[str, float]
            All current share holdings across assets.
        current_prices : Dict[str, float]
            Current prices for all assets.
            
        Returns
        -------
        Dict or None
            Trade information dictionary, or None if no trade executed.
        """
        position_change = new_position - old_position
        
        # Bypass execution if position unchanged and rebalancing not required
        if abs(position_change) < 0.001 and not force_rebalance:
            return None
            
        # Determine total portfolio value across all holdings
        total_assets_value = sum(all_shares.get(s, 0.0) * current_prices.get(s, 0.0) 
                                for s in all_shares.keys())
        total_portfolio_value = cash + total_assets_value
        
        # Compute target allocation value for this asset
        target_value = new_position * total_portfolio_value
        current_value = current_shares * price
        trade_value = target_value - current_value
        
        # Apply slippage model to execution price
        effective_price = price * (1 + self.slippage * np.sign(trade_value))
        
        # Determine quantity to execute
        shares_to_trade = trade_value / effective_price
        
        # Calculate transaction cost impact
        transaction_cost_amount = abs(trade_value) * self.transaction_cost
        
        # Update portfolio state post-execution
        new_cash = cash - trade_value - transaction_cost_amount
        new_shares = current_shares + shares_to_trade
        
        return {
            'date': date,
            'symbol': symbol,
            'price': price,
            'effective_price': effective_price,
            'shares_traded': shares_to_trade,
            'trade_value': trade_value,
            'transaction_cost': transaction_cost_amount,
            'cash_before': cash,
            'cash_after': new_cash,
            'shares_before': current_shares,
            'shares_after': new_shares,
            'position_before': old_position,
            'position_after': new_position
        }