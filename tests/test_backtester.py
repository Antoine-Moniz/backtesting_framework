"""
Unit tests for the Backtester class.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from backtesting_framework import Backtester, Strategy, Result
from examples.strategies import BuyAndHoldStrategy, MovingAverageCrossStrategy


class TestBacktesterBasic:
    """Basic tests for Backtester initialization and execution."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample single-asset test data."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        returns = rng.normal(0.001, 0.02, 100)
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices[:-1],
            'high': [p * 1.01 for p in prices[:-1]],
            'low': [p * 0.99 for p in prices[:-1]],
            'close': prices[1:],
            'volume': rng.integers(1000000, 5000000, 100)
        })
        
        return data
    
    @pytest.fixture
    def multi_asset_data(self):
        """Create multi-asset test data."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        data = pd.DataFrame({
            'date': dates,
            'SPY_close': 300 + np.cumsum(rng.normal(0.001, 0.02, 100)),
            'AGG_close': 100 + np.cumsum(rng.normal(0.0002, 0.005, 100)),
            'close': 300 + np.cumsum(rng.normal(0.001, 0.02, 100))
        })
        
        return data
    
    def test_initialization(self, sample_data):
        """Test Backtester initialization with DataFrame."""
        backtester = Backtester(sample_data, initial_capital=50000)
        
        assert backtester.initial_capital == 50000
        assert len(backtester.data) == 100
        assert 'close' in backtester.data.columns
    
    def test_run_backtest_buy_and_hold(self, sample_data):
        """Test backtest execution with Buy and Hold strategy."""
        backtester = Backtester(sample_data, initial_capital=100000)
        strategy = BuyAndHoldStrategy()
        
        result = backtester.run_backtest(strategy)
        
        assert isinstance(result, Result)
        assert len(result.results_df) == 100
        assert 'portfolio_value' in result.results_df.columns
        assert 'returns' in result.results_df.columns
    
    def test_multi_asset_backtest(self, multi_asset_data):
        """Test multi-asset portfolio backtest."""
        
        class Portfolio60_40(Strategy):
            """60/40 portfolio strategy."""
            
            @property
            def name(self) -> str:
                return "60/40 Portfolio"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                return {'SPY': 0.6, 'AGG': 0.4}
        
        backtester = Backtester(multi_asset_data, initial_capital=100000)
        strategy = Portfolio60_40()
        
        result = backtester.run_backtest(strategy, asset_symbols=['SPY', 'AGG'])
        
        assert isinstance(result, Result)
        assert 'spy_shares' in result.results_df.columns
        assert 'agg_shares' in result.results_df.columns
        assert result.results_df['spy_shares'].max() > 0
        assert result.results_df['agg_shares'].max() > 0
    
    def test_transaction_costs(self, sample_data):
        """Test that transaction costs reduce returns."""
        backtester_no_cost = Backtester(sample_data, initial_capital=100000, 
                                       transaction_cost=0.0)
        backtester_with_cost = Backtester(sample_data, initial_capital=100000, 
                                         transaction_cost=0.001)
        
        strategy = BuyAndHoldStrategy()
        
        result_no_cost = backtester_no_cost.run_backtest(strategy)
        result_with_cost = backtester_with_cost.run_backtest(strategy)
        
        final_no_cost = result_no_cost.results_df['portfolio_value'].iloc[-1]
        final_with_cost = result_with_cost.results_df['portfolio_value'].iloc[-1]
        
        assert final_with_cost <= final_no_cost
    
    def test_portfolio_value_bug_fix(self, multi_asset_data):
        """Test that portfolio value includes all assets."""
        
        class ThreeAssetStrategy(Strategy):
            @property
            def name(self) -> str:
                return "Multi-Asset Test"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                return {'SPY': 0.7, 'AGG': 0.3}
        
        backtester = Backtester(multi_asset_data, initial_capital=100000)
        strategy = ThreeAssetStrategy()
        
        result = backtester.run_backtest(strategy, asset_symbols=['SPY', 'AGG'])
        
        # Verify final portfolio value calculation
        last_row = result.results_df.iloc[-1]
        
        calculated_value = last_row['cash']
        for asset in ['SPY', 'AGG']:
            shares_col = f'{asset.lower()}_shares'
            if shares_col in result.results_df.columns:
                shares = last_row[shares_col]
                # Column names normalized to lowercase by DataHandler
                price = backtester.data.iloc[-1][f'{asset.lower()}_close']
                calculated_value += shares * price
        
        actual_value = last_row['portfolio_value']
        
        # Values should match within floating-point precision
        assert abs(calculated_value - actual_value) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
