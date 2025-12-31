"""
Unit tests for the Result class and compare_results function.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from backtesting_framework import Result
from backtesting_framework.result import compare_results
from examples.strategies import BuyAndHoldStrategy, MovingAverageCrossStrategy


class TestResult:
    """Tests for the Result class."""
    
    @pytest.fixture
    def sample_results_data(self):
        """Create sample results data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Simulate positive performance
        rng = np.random.default_rng(42)
        cumulative_returns = np.cumsum(rng.normal(0.001, 0.02, 100))
        portfolio_values = 100000 * (1 + cumulative_returns)
        
        # Daily returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.concatenate([[0], returns])
        
        # Benchmark (simple random walk)
        benchmark_returns = rng.normal(0.0005, 0.015, 100)
        benchmark_cumulative = np.cumsum(benchmark_returns)
        
        results_df = pd.DataFrame({
            'close': rng.uniform(95, 105, 100),
            'total_value': portfolio_values,
            'portfolio_value': portfolio_values,
            'returns': returns,
            'cumulative_returns': cumulative_returns,
            'benchmark_returns': returns * 0.8,  # Benchmark with 80% of returns
            'benchmark_cumulative': cumulative_returns * 0.8,
            'position': rng.uniform(0, 1, 100),  # Add position column
            'cash': rng.uniform(10000, 50000, 100),
            'default_shares': rng.uniform(0, 1000, 100)
        }, index=dates)
        
        return results_df
    
    @pytest.fixture
    def multi_asset_results_data(self):
        """Create multi-asset results data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        rng = np.random.default_rng(42)
        
        cumulative_returns = np.cumsum(rng.normal(0.001, 0.02, 100))
        portfolio_values = 100000 * (1 + cumulative_returns)
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.concatenate([[0], returns])
        
        results_df = pd.DataFrame({
            'close': rng.uniform(95, 105, 100),
            'total_value': portfolio_values,
            'portfolio_value': portfolio_values,
            'returns': returns,
            'cumulative_returns': cumulative_returns,
            'benchmark_returns': returns * 0.8,  # Benchmark with 80% of returns
            'benchmark_cumulative': cumulative_returns * 0.8,
            'position': rng.uniform(0, 1, 100),  # Main position column
            'position_SPY': rng.uniform(0, 1, 100),  # Position columns for each asset
            'position_AGG': rng.uniform(0, 1, 100),
            'position_GLD': rng.uniform(0, 1, 100),
            'cash': rng.uniform(10000, 50000, 100),
            'SPY_shares': rng.uniform(0, 200, 100),
            'AGG_shares': rng.uniform(0, 500, 100),
            'GLD_shares': rng.uniform(0, 100, 100),
            'SPY_value': rng.uniform(30000, 60000, 100),
            'AGG_value': rng.uniform(20000, 40000, 100),
            'GLD_value': rng.uniform(10000, 20000, 100)
        }, index=dates)
        
        return results_df
    
    def test_result_initialization(self, sample_results_data):
        """Test Result initialization."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        
        assert result.strategy == strategy
        assert len(result.results_df) == 100
        assert result.initial_capital == 100000
        assert result.asset_symbols == ['default']
    
    def test_result_multi_asset_detection(self, multi_asset_results_data, sample_results_data):
        """Test is_multi_asset property."""
        strategy = BuyAndHoldStrategy()
        
        # Single asset - use sample_results_data which has all required columns
        result_single = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        assert not result_single.is_multi_asset
        
        # Multi-asset
        result_multi = Result(
            strategy=strategy,
            results_df=multi_asset_results_data,
            initial_capital=100000,
            asset_symbols=['SPY', 'AGG', 'GLD']
        )
        assert result_multi.is_multi_asset
    
    def test_calculate_metrics_single_asset(self, sample_results_data):
        """Test metrics calculation for single asset."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        
        metrics = result.calculate_metrics()
        
        # Check all required metrics are present
        required_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, np.number))
    
    def test_calculate_metrics_multi_asset(self, multi_asset_results_data):
        """Test metrics calculation for multi-asset portfolio."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=multi_asset_results_data,
            initial_capital=100000,
            asset_symbols=['SPY', 'AGG', 'GLD']
        )
        
        metrics = result.calculate_metrics()
        
        # Same metrics should be calculated
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_sharpe_ratio_calculation(self, sample_results_data):
        """Test Sharpe ratio calculation."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        
        metrics = result.calculate_metrics()
        
        # Sharpe ratio should be a reasonable number
        assert -10 < metrics['sharpe_ratio'] < 10
    
    def test_max_drawdown_calculation(self, sample_results_data):
        """Test max drawdown calculation."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        
        metrics = result.calculate_metrics()
        
        # Max drawdown should be negative or zero
        assert metrics['max_drawdown'] <= 0
        assert metrics['max_drawdown'] >= -1  # Can't lose more than 100%
    
    def test_summary_dataframe(self, sample_results_data):
        """Test summary() returns DataFrame."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        
        summary_df = result.summary()
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) > 0
        assert 'Metric' in summary_df.columns
        assert 'Strategy' in summary_df.columns  # Summary has Strategy and Benchmark columns
    
    def test_plot_performance(self, sample_results_data):
        """Test plot_performance() method."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        
        # Should return a figure and not raise exception
        fig = result.plot_performance(backend='matplotlib')
        
        # Figure should be returned (Agg backend doesn't call plt.show())
        assert fig is not None
    
    @patch('matplotlib.pyplot.show')
    def test_plot_positions_single_asset(self, mock_show, sample_results_data):
        """Test plot_positions() for single asset."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        
        # For single asset, plot_positions prints a message instead of plotting
        result.plot_positions()
        # Should not crash - that's the main test
    
    def test_get_position_summary_single_asset(self, sample_results_data):
        """Test get_position_summary() for single asset."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        
        summary = result.get_position_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'Asset' in summary.columns
        assert 'Mean Position' in summary.columns  # Actual column returned by method
        assert len(summary) >= 1
    
    def test_get_position_summary_multi_asset(self, multi_asset_results_data):
        """Test get_position_summary() for multi-asset portfolio."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=multi_asset_results_data,
            initial_capital=100000,
            asset_symbols=['SPY', 'AGG', 'GLD']
        )
        
        summary = result.get_position_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3  # Three assets
        assert 'Asset' in summary.columns
        assert 'Mean Position' in summary.columns  # Actual columns returned
        assert '% Time Long' in summary.columns
    
    def test_plot_performance_multiple_backends(self, sample_results_data):
        """Test plot_performance() with different backends."""
        strategy = BuyAndHoldStrategy()
        
        result = Result(
            strategy=strategy,
            results_df=sample_results_data,
            initial_capital=100000,
            asset_symbols=['default']
        )
        
        # Test matplotlib - mock both matplotlib.use() and show() to avoid tkinter
        with patch('matplotlib.use'), patch('matplotlib.pyplot.show'):
            result.plot_performance(backend='matplotlib')
        
        # Test seaborn - mock both matplotlib.use() and show()
        with patch('matplotlib.use'), patch('matplotlib.pyplot.show'):
            result.plot_performance(backend='seaborn')
        
        # Test plotly
        with patch('plotly.graph_objects.Figure.show'):
            result.plot_performance(backend='plotly')


class TestCompareResults:
    """Tests for compare_results() function."""
    
    @pytest.fixture
    def multiple_results(self):
        """Create multiple Result objects for comparison."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        rng = np.random.default_rng(42)
        
        results = []
        
        for i in range(3):
            # Create varied performance
            cumulative_returns = np.cumsum(rng.normal(0.001 * (i+1), 0.02, 100))
            portfolio_values = 100000 * (1 + cumulative_returns)
            
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = np.concatenate([[0], returns])
            
            # Create benchmark returns (slightly lower performance)
            benchmark_cumulative = np.cumsum(rng.normal(0.0008, 0.015, 100))
            benchmark_rets = np.diff(benchmark_cumulative)
            benchmark_rets = np.concatenate([[0], benchmark_rets])
            
            results_df = pd.DataFrame({
                'close': rng.uniform(95, 105, 100),
                'portfolio_value': portfolio_values,
                'returns': returns,
                'cumulative_returns': cumulative_returns,
                'benchmark_returns': benchmark_rets,
                'benchmark_cumulative': benchmark_cumulative,
                'position': np.ones(100) * 1.0  # Full position
            }, index=dates)
            
            # Create a mock strategy with custom name
            class CustomStrategy(BuyAndHoldStrategy):
                def __init__(self, strategy_name):
                    super().__init__()
                    self._name = strategy_name
                
                @property
                def name(self):
                    return self._name
            
            strategy = CustomStrategy(f"Strategy {i+1}")
            
            result = Result(
                strategy=strategy,
                results_df=results_df,
                initial_capital=100000,
                asset_symbols=['default']
            )
            
            results.append(result)
        
        return results
    
    def test_compare_results_basic(self, multiple_results):
        """Test basic compare_results functionality."""
        # Unpack list to match *results signature
        with patch('matplotlib.pyplot.show'):
            result = compare_results(*multiple_results)
        
        # compare_results returns a matplotlib Figure, not a DataFrame
        assert result is not None
    
    def test_compare_results_sorting(self, multiple_results):
        """Test that compare_results works with multiple strategies."""
        # Unpack list to match *results signature
        with patch('matplotlib.pyplot.show'):
            result = compare_results(*multiple_results)
        
        # Should return a Figure
        assert result is not None
    
    def test_compare_results_empty_list(self):
        """Test compare_results with empty list."""
        # No args means len(results) == 0 < 2
        with pytest.raises(ValueError, match="At least 2 results"):
            compare_results()
    
    def test_compare_results_single_strategy(self, multiple_results):
        """Test compare_results with single strategy raises error."""
        # Function requires at least 2 results
        with pytest.raises(ValueError, match="At least 2 results"):
            compare_results(multiple_results[0])
    
    def test_compare_results_with_plot(self, multiple_results):
        """Test compare_results with matplotlib backend."""
        # Unpack list and specify backend
        fig = compare_results(*multiple_results, backend='matplotlib')
        
        # Figure should be returned (Agg backend doesn't call plt.show())
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__])
