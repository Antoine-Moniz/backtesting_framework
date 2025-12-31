"""
Unit tests for the Strategy class and concrete strategy implementations.
"""

import pytest
import pandas as pd
import numpy as np

from backtesting_framework import Strategy
from backtesting_framework.strategy import strategy_decorator
from examples.strategies import (
    BuyAndHoldStrategy,
    MovingAverageCrossStrategy,
    MeanReversionStrategy
)


class TestStrategyInterface:
    """Tests for the Strategy abstract base class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'close': 100 + np.cumsum(rng.normal(0, 2, 100)),
            'volume': rng.integers(1000000, 5000000, 100)
        })
    
    def test_strategy_is_abstract(self):
        """Test that Strategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Strategy()
    
    def test_custom_strategy_implementation(self, sample_data):
        """Test implementing a custom strategy."""
        
        class CustomStrategy(Strategy):
            """Custom test strategy."""
            
            @property
            def name(self) -> str:
                return "Custom Test Strategy"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                self.fitted = True
            
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                """Return 50% allocation to asset."""
                return {'asset': 0.5}
        
        strategy = CustomStrategy()
        assert strategy.name == "Custom Test Strategy"
        
        strategy.fit(sample_data)
        assert strategy.fitted
        
        positions = strategy.get_position(sample_data.iloc[-10:], {})
        assert isinstance(positions, dict)
        assert positions.get('asset') == 0.5


class TestBuyAndHoldStrategy:
    """Tests for Buy and Hold strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'close': 100 + np.cumsum(rng.normal(0, 2, 100)),
            'volume': rng.integers(1000000, 5000000, 100)
        })
    
    def test_buy_and_hold_initialization(self):
        """Test strategy initialization."""
        strategy = BuyAndHoldStrategy()
        
        assert strategy.name == "Buy and Hold"
        assert not hasattr(strategy, 'fitted') or not strategy.fitted
    
    def test_buy_and_hold_fit(self, sample_data):
        """Test fit method."""
        strategy = BuyAndHoldStrategy()
        strategy.fit(sample_data)
        
        # fit() doesn't need to do anything for buy and hold
        assert strategy is not None
    
    def test_buy_and_hold_position_single_asset(self, sample_data):
        """Test position signal for single asset."""
        strategy = BuyAndHoldStrategy()
        strategy.fit(sample_data)
        
        # Empty positions - should return 100% allocation
        positions = strategy.get_position(sample_data.iloc[-10:], {})
        assert isinstance(positions, dict)
        assert positions.get('asset') == 1.0
        
        # Already holding - should maintain position
        current_positions = {'asset': 1.0}
        positions = strategy.get_position(sample_data.iloc[-10:], current_positions)
        assert positions.get('asset') == 1.0
    
    def test_buy_and_hold_multi_asset(self):
        """Test buy and hold with multi-asset allocation."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        multi_asset_data = pd.DataFrame({
            'date': dates,
            'SPY_close': 300 + np.cumsum(rng.normal(0, 5, 100)),
            'AGG_close': 100 + np.cumsum(rng.normal(0, 1, 100)),
            'close': 300 + np.cumsum(rng.normal(0, 5, 100))
        })
        
        strategy = BuyAndHoldStrategy()
        strategy.fit(multi_asset_data)
        
        # Should allocate to asset
        positions = strategy.get_position(multi_asset_data.iloc[-10:], {})
        assert 'asset' in positions or len(positions) > 0


class TestMovingAverageCrossStrategy:
    """Tests for Moving Average Crossover strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create trending price data for better signal testing
        trend = np.linspace(100, 120, 100)
        noise = rng.normal(0, 2, 100)
        
        return pd.DataFrame({
            'date': dates,
            'close': trend + noise,
            'volume': rng.integers(1000000, 5000000, 100)
        })
    
    def test_ma_cross_initialization(self):
        """Test strategy initialization with parameters."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        
        assert "MA Cross" in strategy.name
        assert strategy.short_window == 5
        assert strategy.long_window == 20
    
    def test_ma_cross_initialization_invalid_windows(self):
        """Test that invalid window parameters raise error."""
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy(short_window=20, long_window=5)
    
    def test_ma_cross_fit(self, sample_data):
        """Test fit method (optional for MA Cross)."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        # fit() doesn't pre-compute for MA Cross, just verify it doesn't crash
        strategy.fit(sample_data)
        assert strategy is not None
    
    def test_ma_cross_position_single_asset(self, sample_data):
        """Test position signal generation."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        
        # Pass full data for MA calculation
        positions = strategy.get_position(sample_data, {'asset': 0.0})
        
        assert isinstance(positions, dict)
        # Position should be -1.0 (short) or 1.0 (long)
        asset_pos = positions.get('asset', 0)
        assert asset_pos in [-1.0, 1.0]
    
    def test_ma_cross_bullish_signal(self):
        """Test bullish crossover generates buy signal."""
        # Create data with clear uptrend
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = np.linspace(100, 150, 50)  # Strong uptrend
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        
        strategy = MovingAverageCrossStrategy(short_window=3, long_window=10)
        
        # Pass full data for MA calculation
        positions = strategy.get_position(data, {'asset': 0.0})
        
        # In strong uptrend, short MA > long MA → long position
        assert len(positions) > 0
        asset_pos = positions.get('asset', 0)
        assert asset_pos > 0  # Should be 1.0
    
    def test_ma_cross_bearish_signal(self):
        """Test bearish crossover generates sell signal."""
        # Create data with clear downtrend
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = np.linspace(150, 100, 50)  # Strong downtrend
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        
        strategy = MovingAverageCrossStrategy(short_window=3, long_window=10)
        strategy.fit(data)
        
        # At the end, short MA should be below long MA
        positions = strategy.get_position(data.iloc[-5:], {})
        
        # Should have no position or minimal position (bearish)
        asset_pos = positions.get('asset', 0)
        assert asset_pos == 0 or asset_pos < 0.5


class TestMeanReversionStrategy:
    """Tests for Mean Reversion strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create oscillating price data
        base = 100
        oscillation = 10 * np.sin(np.linspace(0, 4*np.pi, 100))
        noise = rng.normal(0, 1, 100)
        
        return pd.DataFrame({
            'date': dates,
            'close': base + oscillation + noise,
            'volume': rng.integers(1000000, 5000000, 100)
        })
    
    def test_mean_reversion_initialization(self):
        """Test strategy initialization."""
        strategy = MeanReversionStrategy(window=20, threshold=1.5)
        
        assert "Mean Reversion" in strategy.name
        assert strategy.window == 20
        assert strategy.threshold == 1.5
    
    def test_mean_reversion_fit(self, sample_data):
        """Test fit method."""
        strategy = MeanReversionStrategy(window=20, threshold=1.5)
        strategy.fit(sample_data)
        
        # Strategy should have calculated statistics
        assert strategy is not None
    
    def test_mean_reversion_position_single_asset(self, sample_data):
        """Test position signal generation."""
        strategy = MeanReversionStrategy(window=20, threshold=1.5)
        strategy.fit(sample_data)
        
        positions = strategy.get_position(sample_data.iloc[-10:], {})
        
        assert isinstance(positions, dict)
        # Position should be between 0.0 and 1.0
        for asset, pos in positions.items():
            assert 0.0 <= pos <= 1.0
    
    def test_mean_reversion_oversold_signal(self):
        """Test oversold condition generates buy signal."""
        # Create data where price drops significantly
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        prices = [100] * 25 + [80, 80, 80, 80, 80]  # Sharp drop at end
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        
        strategy = MeanReversionStrategy(window=20, threshold=1.0)
        
        # Pass full data for bollinger calculation
        positions = strategy.get_position(data, {'asset': 0.0})
        
        asset_pos = positions.get('asset', 0)
        assert asset_pos > 0.5  # Strong buy signal
    
    def test_mean_reversion_overbought_signal(self):
        """Test overbought condition generates sell signal."""
        # Create data where price rises significantly
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        prices = [100] * 25 + [120, 120, 120, 120, 120]  # Sharp rise at end
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        
        strategy = MeanReversionStrategy(window=20, threshold=1.0)
        strategy.fit(data)
        
        # Price is well above mean, should generate sell/no signal
        positions = strategy.get_position(data.iloc[-5:], {})
        
        asset_pos = positions.get('asset', 0)
        assert asset_pos < 0.5  # Weak or no buy signal


class TestStrategyDecorator:
    """Tests for the strategy_decorator."""
    
    def test_decorator_basic_functionality(self):
        """Test that decorator properly wraps strategy methods."""
        
        class TestStrategy(Strategy):
            @property
            def name(self) -> str:
                return "Test"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            @strategy_decorator
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                return {'asset': 1.0}
        
        strategy = TestStrategy()
        result = strategy.get_position(pd.DataFrame(), {})
        
        assert isinstance(result, dict)
        assert result.get('asset') == 1.0
    
    def test_decorator_validation(self):
        """Test that decorator validates inputs."""
        
        class TestStrategy(Strategy):
            @property
            def name(self) -> str:
                return "Test"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            @strategy_decorator
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                # Return invalid type
                return 1.0  # Should be dict
        
        strategy = TestStrategy()
        
        with pytest.raises((TypeError, ValueError)):
            strategy.get_position(pd.DataFrame(), {})
    
    def test_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        
        class TestStrategy(Strategy):
            @property
            def name(self) -> str:
                return "Test"
            
            def fit(self, historical_data: pd.DataFrame) -> None:
                pass
            
            @strategy_decorator
            def get_position(self, data: pd.DataFrame, positions: dict) -> dict:
                """Test docstring."""
                return {}
        
        strategy = TestStrategy()
        
        # Check that function name/docstring are preserved
        assert strategy.get_position.__name__ == "get_position"
        assert "Test docstring" in strategy.get_position.__doc__


if __name__ == "__main__":
    pytest.main([__file__])
