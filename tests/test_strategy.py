"""
Tests unitaires pour la classe Strategy et le décorateur strategy_decorator.
"""

import pytest
import pandas as pd
import numpy as np
from backtesting_framework.strategy import (
    Strategy, 
    strategy_decorator, 
    BuyAndHoldStrategy, 
    MovingAverageCrossStrategy,
    MeanReversionStrategy
)


class TestStrategy:
    """Tests pour la classe abstraite Strategy."""
    
    def test_strategy_initialization(self):
        """Test l'initialisation d'une stratégie."""
        
        class TestStrategy(Strategy):
            def get_position(self, historical_data, current_position):
                return 1.0
        
        strategy = TestStrategy(name="Test", rebalance_frequency="W")
        assert strategy.name == "Test"
        assert strategy.rebalance_frequency == "W"
        assert not strategy.is_fitted
    
    def test_strategy_default_name(self):
        """Test que le nom par défaut est le nom de la classe."""
        
        class CustomStrategy(Strategy):
            def get_position(self, historical_data, current_position):
                return 0.5
        
        strategy = CustomStrategy()
        assert strategy.name == "CustomStrategy"
    
    def test_strategy_fit_method(self):
        """Test la méthode fit par défaut."""
        
        class TestStrategy(Strategy):
            def get_position(self, historical_data, current_position):
                return 1.0
        
        strategy = TestStrategy()
        data = pd.DataFrame({'close': [100, 101, 102]})
        
        assert not strategy.is_fitted
        strategy.fit(data)
        assert strategy.is_fitted
    
    def test_strategy_str_repr(self):
        """Test les méthodes __str__ et __repr__."""
        
        class TestStrategy(Strategy):
            def get_position(self, historical_data, current_position):
                return 1.0
        
        strategy = TestStrategy(name="Test Strategy", rebalance_frequency="M")
        expected = "Strategy: Test Strategy (rebalance: M)"
        assert str(strategy) == expected
        assert repr(strategy) == expected


class TestStrategyDecorator:
    """Tests pour le décorateur strategy_decorator."""
    
    def test_decorator_basic_usage(self):
        """Test l'utilisation basique du décorateur."""
        
        @strategy_decorator(name="Simple Strategy")
        def simple_strategy(historical_data, current_position):
            return 1.0
        
        assert isinstance(simple_strategy, Strategy)
        assert simple_strategy.name == "Simple Strategy"
        assert simple_strategy.rebalance_frequency == "D"
    
    def test_decorator_with_frequency(self):
        """Test le décorateur avec fréquence personnalisée."""
        
        @strategy_decorator(name="Weekly Strategy", rebalance_frequency="W")
        def weekly_strategy(historical_data, current_position):
            return 0.5
        
        assert weekly_strategy.rebalance_frequency == "W"
    
    def test_decorator_default_name(self):
        """Test que le nom par défaut est le nom de la fonction."""
        
        @strategy_decorator()
        def my_custom_strategy(historical_data, current_position):
            return -1.0
        
        assert my_custom_strategy.name == "my_custom_strategy"
    
    def test_decorator_get_position(self):
        """Test que la méthode get_position fonctionne correctement."""
        
        @strategy_decorator()
        def test_strategy(historical_data, current_position):
            if len(historical_data) >= 5:
                return 1.0
            return 0.0
        
        # Test avec peu de données
        data_short = pd.DataFrame({'close': [100, 101, 102]})
        position = test_strategy.get_position(data_short, 0)
        assert abs(position - 0.0) < 1e-6
        
        # Test avec assez de données
        data_long = pd.DataFrame({'close': [100, 101, 102, 103, 104, 105]})
        position = test_strategy.get_position(data_long, 0)
        assert abs(position - 1.0) < 1e-6


class TestBuyAndHoldStrategy:
    """Tests pour la stratégie Buy and Hold."""
    
    def test_buy_and_hold_initialization(self):
        """Test l'initialisation de Buy and Hold."""
        strategy = BuyAndHoldStrategy()
        assert strategy.name == "Buy and Hold"
        assert strategy.rebalance_frequency == "D"
    
    def test_buy_and_hold_position(self):
        """Test que Buy and Hold retourne toujours 1.0."""
        strategy = BuyAndHoldStrategy()
        data = pd.DataFrame({'close': [100, 101, 102, 103, 104]})
        
        position = strategy.get_position(data, 0.5)
        assert abs(position - 1.0) < 1e-6
        
        position = strategy.get_position(data, -1.0)
        assert abs(position - 1.0) < 1e-6


class TestMovingAverageCrossStrategy:
    """Tests pour la stratégie de croisement de moyennes mobiles."""
    
    def test_ma_cross_initialization(self):
        """Test l'initialisation avec paramètres par défaut."""
        strategy = MovingAverageCrossStrategy()
        assert strategy.short_window == 5
        assert strategy.long_window == 20
        assert "MA Cross (5/20)" in strategy.name
    
    def test_ma_cross_custom_windows(self):
        """Test l'initialisation avec fenêtres personnalisées."""
        strategy = MovingAverageCrossStrategy(short_window=10, long_window=30)
        assert strategy.short_window == 10
        assert strategy.long_window == 30
        assert "MA Cross (10/30)" in strategy.name
    
    def test_ma_cross_insufficient_data(self):
        """Test avec données insuffisantes."""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        data = pd.DataFrame({'close': [100, 101, 102]})  # Seulement 3 points
        
        position = strategy.get_position(data, 0)
        assert position == 0  # Pas assez de données
    
    def test_ma_cross_long_signal(self):
        """Test signal d'achat (MA courte > MA longue)."""
        strategy = MovingAverageCrossStrategy(short_window=2, long_window=5)
        
        # Créer des données avec tendance haussière récente
        prices = [100, 100, 100, 100, 100, 105, 110]  # Prix monte à la fin
        data = pd.DataFrame({'close': prices})
        
        position = strategy.get_position(data, 0)
        assert abs(position - 1.0) < 1e-6  # Signal d'achat
    
    def test_ma_cross_short_signal(self):
        """Test signal de vente (MA courte < MA longue)."""
        strategy = MovingAverageCrossStrategy(short_window=2, long_window=5)
        
        # Créer des données avec tendance baissière récente
        prices = [110, 110, 110, 110, 110, 105, 95]  # Prix baisse à la fin
        data = pd.DataFrame({'close': prices})
        
        position = strategy.get_position(data, 0)
        assert position == -1.0  # Signal de vente


class TestMeanReversionStrategy:
    """Tests pour la stratégie de retour à la moyenne."""
    
    def test_mean_reversion_initialization(self):
        """Test l'initialisation avec paramètres par défaut."""
        strategy = MeanReversionStrategy()
        assert strategy.window == 20
        assert strategy.num_std == pytest.approx(2.0)
        assert "Mean Reversion" in strategy.name
    
    def test_mean_reversion_custom_params(self):
        """Test l'initialisation avec paramètres personnalisés."""
        strategy = MeanReversionStrategy(window=10, num_std=1.5)
        assert strategy.window == 10
        assert strategy.num_std == pytest.approx(1.5)
    
    def test_mean_reversion_insufficient_data(self):
        """Test avec données insuffisantes."""
        strategy = MeanReversionStrategy(window=20)
        data = pd.DataFrame({'close': [100, 101, 102]})  # Seulement 3 points
        
        position = strategy.get_position(data, 0)
        assert position == 0  # Pas assez de données
    
    def test_mean_reversion_buy_signal(self):
        """Test signal d'achat (prix sous la bande inférieure)."""
        strategy = MeanReversionStrategy(window=5, num_std=1.0)
        
        # Prix très bas par rapport à la moyenne
        prices = [100, 100, 100, 100, 100, 95]  # Dernier prix très bas
        data = pd.DataFrame({'close': prices})
        
        position = strategy.get_position(data, 0)
        assert abs(position - 1.0) < 1e-6  # Signal d'achat
    
    def test_mean_reversion_sell_signal(self):
        """Test signal de vente (prix au-dessus de la bande supérieure)."""
        strategy = MeanReversionStrategy(window=5, num_std=1.0)
        
        # Prix très haut par rapport à la moyenne
        prices = [100, 100, 100, 100, 100, 105]  # Dernier prix très haut
        data = pd.DataFrame({'close': prices})
        
        position = strategy.get_position(data, 0)
        assert position == -1.0  # Signal de vente
    
    def test_mean_reversion_neutral_signal(self):
        """Test signal neutre (prix dans les bandes)."""
        strategy = MeanReversionStrategy(window=5, num_std=2.0)
        
        # Prix stables autour de la moyenne
        prices = [100, 100, 100, 100, 100, 100]
        data = pd.DataFrame({'close': prices})
        
        position = strategy.get_position(data, 0)
        assert abs(position - 0.0) < 1e-6  # Signal neutre


if __name__ == "__main__":
    pytest.main([__file__])