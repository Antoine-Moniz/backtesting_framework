"""
Tests unitaires pour la classe Backtester.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from backtesting_framework.backtester import Backtester
from backtesting_framework.strategy import BuyAndHoldStrategy, MovingAverageCrossStrategy
from backtesting_framework.result import Result


class TestBacktester:
    """Tests pour la classe Backtester."""
    
    @pytest.fixture
    def sample_data(self):
        """Crée des données de test."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Génération de prix avec random walk
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
    def csv_file(self, sample_data):
        """Crée un fichier CSV temporaire avec les données de test."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_backtester_initialization_dataframe(self, sample_data):
        """Test l'initialisation avec un DataFrame."""
        backtester = Backtester(sample_data, initial_capital=50000)
        
        assert backtester.initial_capital == 50000
        assert len(backtester.data) == 100
        assert 'close' in backtester.data.columns
        assert isinstance(backtester.data.index, pd.DatetimeIndex)
    
    def test_backtester_initialization_csv(self, csv_file):
        """Test l'initialisation avec un fichier CSV."""
        backtester = Backtester(csv_file)
        
        assert backtester.initial_capital == 100000  # Valeur par défaut
        assert len(backtester.data) > 0
        assert 'close' in backtester.data.columns
    
    def test_backtester_initialization_nonexistent_file(self):
        """Test l'initialisation avec un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            Backtester("nonexistent_file.csv")
    
    def test_backtester_initialization_unsupported_format(self):
        """Test l'initialisation avec un format non supporté."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test data")
            temp_file = f.name
            
        try:
            with pytest.raises(ValueError, match="Format de fichier non supporté"):
                Backtester(temp_file)
        finally:
            try:
                os.unlink(temp_file)
            except OSError:
                pass  # Ignore les erreurs de suppression sur Windows
    
    def test_data_validation_missing_columns(self):
        """Test la validation avec colonnes manquantes."""
        # Données sans colonne 'close'
        bad_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'price': [100 + i for i in range(10)]
        })
        
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            Backtester(bad_data)
    
    def test_data_validation_auto_columns(self, sample_data):
        """Test la création automatique des colonnes manquantes."""
        # Supprimer quelques colonnes
        minimal_data = sample_data[['date', 'close']].copy()
        
        backtester = Backtester(minimal_data)
        
        # Vérifier que les colonnes ont été ajoutées
        assert 'open' in backtester.data.columns
        assert 'high' in backtester.data.columns
        assert 'low' in backtester.data.columns
        assert 'volume' in backtester.data.columns
    
    def test_run_backtest_buy_and_hold(self, sample_data):
        """Test un backtest simple avec Buy and Hold."""
        backtester = Backtester(sample_data, initial_capital=100000)
        strategy = BuyAndHoldStrategy()
        
        result = backtester.run_backtest(strategy)
        
        assert isinstance(result, Result)
        assert result.strategy.name == "Buy and Hold"
        assert len(result.results_df) == len(sample_data)
        assert 'portfolio_value' in result.results_df.columns
        assert 'returns' in result.results_df.columns
        assert 'position' in result.results_df.columns
    
    def test_run_backtest_ma_cross(self, sample_data):
        """Test un backtest avec stratégie de croisement de moyennes mobiles."""
        backtester = Backtester(sample_data, initial_capital=100000)
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        
        result = backtester.run_backtest(strategy)
        
        assert isinstance(result, Result)
        assert "MA Cross" in result.strategy.name
        assert len(result.trades) > 0  # Devrait y avoir des trades
    
    def test_run_backtest_with_dates(self, sample_data):
        """Test un backtest avec dates de début et fin."""
        backtester = Backtester(sample_data)
        strategy = BuyAndHoldStrategy()
        
        result = backtester.run_backtest(
            strategy,
            start_date='2023-01-15',
            end_date='2023-02-15'
        )
        
        # Vérifier que les résultats couvrent la période demandée
        assert result.results_df.index[0] >= pd.to_datetime('2023-01-15')
        assert result.results_df.index[-1] <= pd.to_datetime('2023-02-15')
    
    def test_run_backtest_invalid_date_range(self, sample_data):
        """Test avec une plage de dates invalide."""
        backtester = Backtester(sample_data)
        strategy = BuyAndHoldStrategy()
        
        with pytest.raises(ValueError, match="Aucune donnée disponible"):
            backtester.run_backtest(
                strategy,
                start_date='2025-01-01',  # Date dans le futur
                end_date='2025-01-31'
            )
    
    def test_transaction_costs(self, sample_data):
        """Test que les coûts de transaction sont appliqués."""
        # Backtester avec coûts élevés
        backtester_high_cost = Backtester(
            sample_data, 
            transaction_cost=0.01,  # 1% de coût
            slippage=0.001  # 0.1% de slippage
        )
        
        # Backtester sans coûts
        backtester_no_cost = Backtester(
            sample_data,
            transaction_cost=0.0,
            slippage=0.0
        )
        
        strategy = MovingAverageCrossStrategy(short_window=3, long_window=10)
        
        result_high_cost = backtester_high_cost.run_backtest(strategy)
        result_no_cost = backtester_no_cost.run_backtest(strategy)
        
        # La performance avec coûts devrait être inférieure
        high_cost_return = result_high_cost.metrics['total_return']
        no_cost_return = result_no_cost.metrics['total_return']
        
        assert high_cost_return <= no_cost_return
    
    def test_rebalance_frequencies(self, sample_data):
        """Test différentes fréquences de rééquilibrage."""
        backtester = Backtester(sample_data)
        
        # Stratégie quotidienne
        strategy_daily = MovingAverageCrossStrategy(short_window=3, long_window=10)
        strategy_daily.rebalance_frequency = 'D'
        
        # Stratégie hebdomadaire
        strategy_weekly = MovingAverageCrossStrategy(short_window=3, long_window=10)
        strategy_weekly.rebalance_frequency = 'W'
        
        result_daily = backtester.run_backtest(strategy_daily)
        result_weekly = backtester.run_backtest(strategy_weekly)
        
        # La stratégie quotidienne devrait avoir plus de trades
        assert len(result_daily.trades) >= len(result_weekly.trades)
    
    def test_get_rebalance_dates(self, sample_data):
        """Test le calcul des dates de rééquilibrage."""
        backtester = Backtester(sample_data)
        
        # Test quotidien
        daily_dates = backtester._get_rebalance_dates(backtester.data, 'D')
        assert len(daily_dates) == len(backtester.data)
        
        # Test hebdomadaire
        weekly_dates = backtester._get_rebalance_dates(backtester.data, 'W')
        assert len(weekly_dates) <= len(backtester.data)
        
        # Test mensuel
        monthly_dates = backtester._get_rebalance_dates(backtester.data, 'M')
        assert len(monthly_dates) <= len(weekly_dates)
    
    def test_execute_trade(self, sample_data):
        """Test l'exécution d'un trade."""
        backtester = Backtester(sample_data, transaction_cost=0.001, slippage=0.0001)
        
        trade_info = backtester._execute_trade(
            old_position=0.0,
            new_position=1.0,
            price=100.0,
            cash=100000.0,
            current_shares=0.0,
            date=pd.Timestamp('2023-01-01')
        )
        
        assert trade_info is not None
        assert 'transaction_cost' in trade_info
        assert 'effective_price' in trade_info
        assert trade_info['transaction_cost'] > 0
        assert abs(trade_info['effective_price'] - 100.0) > 1e-10  # À cause du slippage
    
    def test_execute_trade_no_change(self, sample_data):
        """Test qu'aucun trade n'est exécuté si pas de changement significatif."""
        backtester = Backtester(sample_data)
        
        trade_info = backtester._execute_trade(
            old_position=1.0,
            new_position=1.0001,  # Changement très petit
            price=100.0,
            cash=100000.0,
            current_shares=1000.0,
            date=pd.Timestamp('2023-01-01')
        )
        
        assert trade_info is None  # Pas de trade à cause du seuil de tolérance


if __name__ == "__main__":
    pytest.main([__file__])