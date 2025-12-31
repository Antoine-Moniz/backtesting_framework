"""
Comprehensive tests for multi-asset portfolio functionality.

This test suite validates the critical bug fix and multi-asset support.
"""

import pytest
import pandas as pd
import numpy as np

from backtesting_framework import Backtester, Strategy, Result
from examples.strategies import BuyAndHoldStrategy


class Test60_40Portfolio:
    """Tests for classic 60/40 stock/bond portfolio."""
    
    @pytest.fixture
    def portfolio_data(self):
        """Create 60/40 portfolio test data (SPY/AGG)."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')  # 1 year of trading days
        
        # SPY: Higher return, higher volatility
        spy_returns = rng.normal(0.0008, 0.015, 252)  # ~20% annual, 24% vol
        spy_prices = 300 * np.exp(np.cumsum(spy_returns))
        
        # AGG: Lower return, lower volatility
        agg_returns = rng.normal(0.0002, 0.004, 252)  # ~5% annual, 6% vol
        agg_prices = 100 * np.exp(np.cumsum(agg_returns))
        
        data = pd.DataFrame({
            'date': dates,
            'SPY_close': spy_prices,
            'AGG_close': agg_prices,
            'close': spy_prices  # Fallback
        })
        
        return data
    
    def test_60_40_allocation(self, portfolio_data):
        """Test that 60/40 allocation is maintained."""
        
        class Portfolio60_40(Strategy):
            """60% SPY, 40% AGG strategy."""
            
            @property
            def name(self) -> str:
                return "60/40 Portfolio"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                return {'SPY': 0.6, 'AGG': 0.4}
        
        backtester = Backtester(portfolio_data, initial_capital=100000)
        strategy = Portfolio60_40()
        
        result = backtester.run_backtest(strategy, asset_symbols=['SPY', 'AGG'])
        
        # Check that positions exist (DataHandler normalizes to lowercase)
        assert 'spy_shares' in result.results_df.columns
        assert 'agg_shares' in result.results_df.columns
        
        # Check that positions were allocated
        assert result.results_df['spy_shares'].max() > 0
        assert result.results_df['agg_shares'].max() > 0
        
        # Check final allocation is approximately 60/40
        last_row = result.results_df.iloc[-1]
        spy_value = last_row['spy_shares'] * backtester.data.iloc[-1]['spy_close']
        agg_value = last_row['agg_shares'] * backtester.data.iloc[-1]['agg_close']
        total_invested = spy_value + agg_value
        
        if total_invested > 0:
            spy_pct = spy_value / total_invested
            agg_pct = agg_value / total_invested
            
            assert 0.55 < spy_pct < 0.65  # Allow 5% tolerance
            assert 0.35 < agg_pct < 0.45
    
    def test_60_40_rebalancing(self, portfolio_data):
        """Test monthly rebalancing of 60/40 portfolio."""
        
        class Rebalancing60_40(Strategy):
            """60/40 with monthly rebalancing."""
            
            def __init__(self):
                super().__init__(rebalance_frequency='M')  # Monthly rebalancing
                self.is_fitted = False
            
            @property
            def name(self) -> str:
                return "60/40 Rebalancing"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                # Rebalance to 60/40 when called (backtester handles frequency)
                return {'SPY': 0.6, 'AGG': 0.4}
        
        backtester = Backtester(portfolio_data, initial_capital=100000)
        strategy = Rebalancing60_40()
        
        result = backtester.run_backtest(strategy, asset_symbols=['SPY', 'AGG'])
        
        # Should have multiple position changes (rebalancing events)
        spy_shares = result.results_df['spy_shares']
        rebalance_events = (spy_shares.diff().abs() > 0.01).sum()
        
        assert rebalance_events >= 8  # At least 8 months worth
    
    def test_60_40_vs_100_stock(self, portfolio_data):
        """Test that 60/40 has lower volatility than 100% stocks."""
        
        class Portfolio60_40(Strategy):
            @property
            def name(self) -> str:
                return "60/40"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                return {'SPY': 0.6, 'AGG': 0.4}
        
        class Portfolio100Stock(Strategy):
            @property
            def name(self) -> str:
                return "100% SPY"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                return {'SPY': 1.0}
        
        backtester = Backtester(portfolio_data, initial_capital=100000)
        
        result_60_40 = backtester.run_backtest(Portfolio60_40(), asset_symbols=['SPY', 'AGG'])
        result_100_stock = backtester.run_backtest(Portfolio100Stock(), asset_symbols=['SPY', 'AGG'])
        
        metrics_60_40 = result_60_40.calculate_metrics()
        metrics_100_stock = result_100_stock.calculate_metrics()
        
        # 60/40 should have lower volatility
        assert metrics_60_40['volatility'] < metrics_100_stock['volatility']


class TestPortfolioValueBugFix:
    """Tests that validate the critical portfolio value calculation bug fix."""
    
    @pytest.fixture
    def three_asset_data(self):
        """Create data with three uncorrelated assets."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        data = pd.DataFrame({
            'date': dates,
            'ASSET_A_close': 100 + np.cumsum(rng.normal(0.001, 0.02, 100)),
            'ASSET_B_close': 50 + np.cumsum(rng.normal(0.0005, 0.01, 100)),
            'ASSET_C_close': 200 + np.cumsum(rng.normal(0.002, 0.03, 100)),
            'close': 100 + np.cumsum(rng.normal(0.001, 0.02, 100))
        })
        
        return data
    
    def test_portfolio_value_includes_all_assets(self, three_asset_data):
        """Test that portfolio value includes ALL assets, not just one."""
        
        class ThreeAssetStrategy(Strategy):
            """Equal weight across three assets."""
            
            @property
            def name(self) -> str:
                return "Three Asset Portfolio"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                return {
                    'ASSET_A': 1/3,
                    'ASSET_B': 1/3,
                    'ASSET_C': 1/3
                }
        
        backtester = Backtester(three_asset_data, initial_capital=90000)
        strategy = ThreeAssetStrategy()
        
        result = backtester.run_backtest(strategy)
        
        # Manually calculate portfolio value for last row
        last_row = result.results_df.iloc[-1]
        
        # Get shares and prices for all assets
        asset_a_shares = last_row.get('ASSET_A_shares', 0)
        asset_b_shares = last_row.get('ASSET_B_shares', 0)
        asset_c_shares = last_row.get('ASSET_C_shares', 0)
        
        asset_a_price = three_asset_data.iloc[-1]['ASSET_A_close']
        asset_b_price = three_asset_data.iloc[-1]['ASSET_B_close']
        asset_c_price = three_asset_data.iloc[-1]['ASSET_C_close']
        
        # Calculate expected portfolio value
        expected_value = (
            last_row['cash'] +
            asset_a_shares * asset_a_price +
            asset_b_shares * asset_b_price +
            asset_c_shares * asset_c_price
        )
        
        actual_value = last_row['portfolio_value']
        
        # Values should match (within floating point precision)
        assert abs(expected_value - actual_value) < 0.01, \
            f"Portfolio value mismatch: expected {expected_value}, got {actual_value}"
    
    def test_portfolio_value_calculation_each_step(self, three_asset_data):
        """Test portfolio value calculation is correct at every step."""
        
        class StaticThreeAsset(Strategy):
            @property
            def name(self) -> str:
                return "Static Three Asset"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                # Only allocate on first step
                if len(positions) == 0:
                    return {'ASSET_A': 0.5, 'ASSET_B': 0.3, 'ASSET_C': 0.2}
                return positions
        
        backtester = Backtester(three_asset_data, initial_capital=100000)
        strategy = StaticThreeAsset()
        
        result = backtester.run_backtest(strategy)
        
        # Check portfolio value calculation for multiple rows
        for idx in [10, 25, 50, 75, -1]:
            row = result.results_df.iloc[idx]
            data_row = three_asset_data.iloc[idx]
            
            calculated_value = row['cash']
            
            for asset in ['ASSET_A', 'ASSET_B', 'ASSET_C']:
                shares_col = f'{asset}_shares'
                if shares_col in result.results_df.columns:
                    shares = row[shares_col]
                    price = data_row[f'{asset}_close']
                    calculated_value += shares * price
            
            actual_value = row['portfolio_value']
            
            assert abs(calculated_value - actual_value) < 0.01, \
                f"Row {idx}: Portfolio value mismatch"


class TestDynamicRebalancing:
    """Tests for dynamic rebalancing strategies."""
    
    @pytest.fixture
    def volatile_data(self):
        """Create data with different volatilities."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # Low vol asset
        low_vol = 100 + np.cumsum(rng.normal(0.0005, 0.005, 200))
        
        # High vol asset
        high_vol = 100 + np.cumsum(rng.normal(0.001, 0.03, 200))
        
        data = pd.DataFrame({
            'date': dates,
            'STABLE_close': low_vol,
            'VOLATILE_close': high_vol,
            'close': low_vol
        })
        
        return data
    
    def test_volatility_targeting(self, volatile_data):
        """Test strategy that adjusts allocation based on volatility."""
        
        class VolatilityTargeting(Strategy):
            """Allocate inversely to volatility."""
            
            @property
            def name(self) -> str:
                return "Volatility Targeting"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                # Calculate recent volatility (last 20 days)
                if len(data) < 20:
                    return {'STABLE': 0.5, 'VOLATILE': 0.5}
                
                stable_vol = data['stable_close'].tail(20).pct_change().std()
                volatile_vol = data['volatile_close'].tail(20).pct_change().std()
                
                # Inverse volatility weighting
                stable_weight = (1/stable_vol) / ((1/stable_vol) + (1/volatile_vol))
                volatile_weight = 1 - stable_weight
                
                return {
                    'STABLE': stable_weight,
                    'VOLATILE': volatile_weight
                }
        
        backtester = Backtester(volatile_data, initial_capital=100000)
        strategy = VolatilityTargeting()
        
        result = backtester.run_backtest(strategy, asset_symbols=['VOLATILE', 'STABLE'])
        
        # Check that strategy allocated more to stable asset
        final_row = result.results_df.iloc[-1]
        
        # DataHandler converts column names to lowercase - use backtester.data for prices
        stable_value = final_row['stable_shares'] * backtester.data.iloc[-1]['stable_close']
        volatile_value = final_row['volatile_shares'] * backtester.data.iloc[-1]['volatile_close']
        
        total = stable_value + volatile_value
        
        if total > 0:
            stable_pct = stable_value / total
            # Should favor stable asset (higher weight)
            assert stable_pct > 0.5
    
    def test_momentum_rebalancing(self, volatile_data):
        """Test strategy that rebalances to winners."""
        
        class MomentumRebalancing(Strategy):
            """Allocate more to recent winners."""
            
            def __init__(self):
                super().__init__()
                self.step = 0
                self.is_fitted = False
            
            @property
            def name(self) -> str:
                return "Momentum Rebalancing"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                self.step += 1
                
                # Rebalance every 20 days
                if self.step % 20 != 0:
                    return positions
                
                if len(data) < 20:
                    return {'STABLE': 0.5, 'VOLATILE': 0.5}
                
                # Calculate 20-day returns
                stable_ret = (data['stable_close'].iloc[-1] / data['stable_close'].iloc[-20]) - 1
                volatile_ret = (data['volatile_close'].iloc[-1] / data['volatile_close'].iloc[-20]) - 1
                
                # Allocate more to winner
                if stable_ret > volatile_ret:
                    return {'STABLE': 0.7, 'VOLATILE': 0.3}
                else:
                    return {'STABLE': 0.3, 'VOLATILE': 0.7}
        
        backtester = Backtester(volatile_data, initial_capital=100000)
        strategy = MomentumRebalancing()
        
        result = backtester.run_backtest(strategy, asset_symbols=['STABLE', 'VOLATILE'])
        
        # Should have multiple rebalancing events (use lowercase column name)
        stable_shares = result.results_df['stable_shares']
        changes = (stable_shares.diff().abs() > 0.01).sum()
        
        assert changes >= 5  # At least 5 rebalancing events


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_positions(self):
        """Test strategy that returns empty positions dict."""
        
        class NoPositionsStrategy(Strategy):
            @property
            def name(self) -> str:
                return "No Positions"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                return {}
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': np.linspace(100, 110, 100)
        })
        
        backtester = Backtester(data, initial_capital=100000)
        strategy = NoPositionsStrategy()
        
        result = backtester.run_backtest(strategy)
        
        # Should hold all cash
        assert result.results_df['cash'].iloc[-1] == 100000
        assert result.results_df['portfolio_value'].iloc[-1] == 100000
    
    def test_extreme_allocation(self):
        """Test strategy with extreme allocations (>100%)."""
        
        class ExtremeAllocation(Strategy):
            @property
            def name(self) -> str:
                return "Extreme Allocation"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                # Try to allocate 150% (should be capped)
                return {'ASSET_A': 1.5}
        
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'ASSET_A_close': np.linspace(100, 120, 50),
            'close': np.linspace(100, 120, 50)
        })
        
        backtester = Backtester(data, initial_capital=100000)
        strategy = ExtremeAllocation()
        
        result = backtester.run_backtest(strategy, asset_symbols=['ASSET_A'])
        
        # Should be capped at available capital (use lowercase column names)
        first_allocation_row = result.results_df[result.results_df['asset_a_shares'] > 0].iloc[0]
        
        # Get the first price from the backtester's data (after DataHandler normalization)
        first_price = backtester.data.iloc[0]['asset_a_close']
        total_value = (first_allocation_row['cash'] + 
                      first_allocation_row['asset_a_shares'] * first_price)
        
        # Total value should not exceed initial capital (no leverage)
        assert total_value <= 100000 * 1.01  # Allow 1% tolerance
    
    def test_negative_positions(self):
        """Test that negative positions are handled correctly."""
        
        class NegativePositions(Strategy):
            @property
            def name(self) -> str:
                return "Negative Positions"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                # Return negative position (short)
                return {'ASSET_A': -0.5}
        
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'ASSET_A_close': np.linspace(100, 90, 50),
            'close': np.linspace(100, 90, 50)
        })
        
        backtester = Backtester(data, initial_capital=100000)
        strategy = NegativePositions()
        
        # Should either reject negative positions or handle as zero
        result = backtester.run_backtest(strategy, asset_symbols=['ASSET_A'])
        
        # Shares should not be negative (short selling not supported) - use lowercase
        assert (result.results_df['asset_a_shares'] >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__])
