"""
Tests unitaires pour la classe Result et la fonction compare_results.
"""

import pytest
import pandas as pd
import numpy as np

from backtesting_framework.result import Result, compare_results
from backtesting_framework.strategy import BuyAndHoldStrategy, MovingAverageCrossStrategy


class TestResult:
    """Tests pour la classe Result."""
    
    @pytest.fixture
    def sample_results_data(self):
        """Crée des données de résultats de test."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Simulation d'une performance positive
        rng = np.random.default_rng(42)
        cumulative_returns = np.cumsum(rng.normal(0.001, 0.02, 100))
        portfolio_values = 100000 * (1 + cumulative_returns)
        
        # Génération de positions aléatoires
        positions = rng.choice([-1, 0, 1], size=100, p=[0.2, 0.3, 0.5])
        
        # Rendements quotidiens
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.concatenate([[0], returns])
        
        # Benchmark (simple random walk)
        benchmark_returns = rng.normal(0.0005, 0.015, 100)
        benchmark_cumulative = np.cumsum(benchmark_returns)
        
        results_df = pd.DataFrame({
            'close': rng.uniform(95, 105, 100),
            'position': positions,
            'portfolio_value': portfolio_values,
            'returns': returns,
            'cumulative_returns': cumulative_returns,
            'benchmark_returns': benchmark_returns,
            'benchmark_cumulative': benchmark_cumulative
        }, index=dates)
        
        return results_df
    
    @pytest.fixture
    def sample_trades(self):
        """Crée des trades de test."""
        dates = pd.date_range('2023-01-01', periods=10, freq='10D')
        trades = []
        
        for i, date in enumerate(dates):
            trade = {
                'date': date,
                'price': 100 + i,
                'effective_price': 100 + i + 0.01,
                'shares_traded': 1000 * (-1) ** i,
                'trade_value': 10000 * (-1) ** i,
                'transaction_cost': 10,
                'cash_before': 50000,
                'cash_after': 50000 - 10000 * (-1) ** i,
                'shares_before': 1000 * i,
                'shares_after': 1000 * (i + (-1) ** i),
                'position_before': 0.5 * i,
                'position_after': 0.5 * (i + (-1) ** i)
            }
            trades.append(trade)
        
        return trades
    
    @pytest.fixture
    def sample_result(self, sample_results_data, sample_trades):
        """Crée un objet Result de test."""
        strategy = BuyAndHoldStrategy()
        return Result(
            strategy=strategy,
            results_df=sample_results_data,
            trades=sample_trades,
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.0001
        )
    
    def test_result_initialization(self, sample_result):
        """Test l'initialisation de Result."""
        assert sample_result.strategy.name == "Buy and Hold"
        assert sample_result.initial_capital == 100000
        assert sample_result.transaction_cost == pytest.approx(0.001)
        assert sample_result.slippage == pytest.approx(0.0001)
        assert len(sample_result.trades) == 10
        assert hasattr(sample_result, 'metrics')
    
    def test_metrics_calculation(self, sample_result):
        """Test le calcul des métriques."""
        metrics = sample_result.metrics
        
        # Vérifier que toutes les métriques principales sont présentes
        required_metrics = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'num_trades', 'winning_trades_pct',
            'beta', 'alpha', 'total_transaction_costs'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_total_return_calculation(self, sample_result):
        """Test le calcul du rendement total."""
        final_value = sample_result.results_df['portfolio_value'].iloc[-1]
        expected_return = (final_value / sample_result.initial_capital) - 1
        
        assert abs(sample_result.metrics['total_return'] - expected_return) < 1e-10
    
    def test_volatility_calculation(self, sample_result):
        """Test le calcul de la volatilité annualisée."""
        returns = sample_result.results_df['returns'].dropna()
        expected_volatility = returns.std() * np.sqrt(252)
        
        assert abs(sample_result.metrics['volatility'] - expected_volatility) < 1e-10
    
    def test_sharpe_ratio_calculation(self, sample_result):
        """Test le calcul du ratio de Sharpe."""
        expected_sharpe = (sample_result.metrics['annualized_return'] / 
                          sample_result.metrics['volatility'])
        
        if sample_result.metrics['volatility'] > 0:
            assert abs(sample_result.metrics['sharpe_ratio'] - expected_sharpe) < 1e-10
        else:
            assert sample_result.metrics['sharpe_ratio'] == 0
    
    def test_max_drawdown_calculation(self, sample_result):
        """Test le calcul du drawdown maximum."""
        cumulative_returns = sample_result.results_df['cumulative_returns']
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - rolling_max
        expected_max_drawdown = drawdowns.min()
        
        assert abs(sample_result.metrics['max_drawdown'] - expected_max_drawdown) < 1e-10
    
    def test_trade_statistics(self, sample_result):
        """Test les statistiques des trades."""
        assert sample_result.metrics['num_trades'] == 10
        assert 0 <= sample_result.metrics['winning_trades_pct'] <= 100
        assert sample_result.metrics['total_transaction_costs'] > 0
    
    def test_summary_dataframe(self, sample_result):
        """Test la génération du résumé."""
        summary = sample_result.summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'Métrique' in summary.columns
        assert 'Stratégie' in summary.columns
        assert 'Benchmark' in summary.columns
        
        # Vérifier que certaines métriques importantes sont présentes
        assert 'Rendement Total (%)' in summary['Métrique'].values
        assert 'Ratio de Sharpe' in summary['Métrique'].values
        assert 'Drawdown Maximum (%)' in summary['Métrique'].values
    
    def test_str_repr(self, sample_result):
        """Test les méthodes __str__ et __repr__."""
        str_repr = str(sample_result)
        assert "Buy and Hold" in str_repr
        assert "total return" in str_repr
        
        repr_result = repr(sample_result)
        assert str_repr == repr_result
    
    def test_get_available_backends(self):
        """Test la méthode get_available_backends."""
        backends = Result.get_available_backends()
        assert isinstance(backends, list)
        # Au moins matplotlib devrait être disponible
        assert len(backends) > 0


class TestCompareResults:
    """Tests pour la fonction compare_results."""
    
    def test_compare_results_insufficient_results(self):
        """Test avec un nombre insuffisant de résultats."""
        result1 = None  # Placeholder
        
        with pytest.raises(ValueError, match="Au moins 2 résultats"):
            compare_results(result1)
    
    def test_compare_results_valid_input(self):
        """Test avec des entrées valides."""
        # Création de données de test locales
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        rng = np.random.default_rng(42)
        
        # Données synthétiques
        data1 = pd.DataFrame({
            'close': rng.uniform(95, 105, 50),
            'portfolio_value': rng.uniform(90000, 110000, 50),
            'returns': rng.normal(0.001, 0.02, 50),
            'cumulative_returns': np.cumsum(rng.normal(0.001, 0.02, 50)),
            'benchmark_returns': rng.normal(0.0005, 0.015, 50),
            'benchmark_cumulative': np.cumsum(rng.normal(0.0005, 0.015, 50)),
            'position': rng.choice([-1, 0, 1], 50)
        }, index=dates)
        
        data2 = data1.copy()
        data2['portfolio_value'] *= 0.95
        
        strategy1 = BuyAndHoldStrategy()
        strategy2 = MovingAverageCrossStrategy()
        
        result1 = Result(strategy1, data1, [], 100000, 0.001, 0.0001)
        result2 = Result(strategy2, data2, [], 100000, 0.001, 0.0001)
        
        # Test que la fonction ne lève pas d'erreur
        try:
            compare_results(result1, result2, backend='matplotlib')
        except Exception as e:
            # Acceptable si matplotlib n'est pas disponible
            assert "non disponible" in str(e)
    
    def test_compare_results_basic(self):
        """Test de comparaison basique."""
        # Création de données minimales pour test
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        
        data = pd.DataFrame({
            'close': range(100, 120),
            'portfolio_value': range(100000, 120000, 1000),
            'returns': [0.01] * 20,
            'cumulative_returns': np.cumsum([0.01] * 20),
            'benchmark_returns': [0.005] * 20,
            'benchmark_cumulative': np.cumsum([0.005] * 20),
            'position': [1] * 20
        }, index=dates)
        
        strategy1 = BuyAndHoldStrategy()
        strategy2 = MovingAverageCrossStrategy()
        
        result1 = Result(strategy1, data, [], 100000, 0.001, 0.0001)
        result2 = Result(strategy2, data, [], 100000, 0.001, 0.0001)
        
        # Test que la fonction accepte 2+ résultats
        try:
            compare_results(result1, result2, backend='matplotlib')
        except Exception as e:
            # Acceptable si matplotlib n'est pas disponible
            assert "non disponible" in str(e)


class TestResultVisualization:
    """Tests pour les méthodes de visualisation."""
    
    def test_plot_performance_backend_validation(self):
        """Test la validation des backends de visualisation."""
        # Création d'un résultat de test simple
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'close': range(100, 110),
            'portfolio_value': range(100000, 110000, 1000),
            'returns': [0.01] * 10,
            'cumulative_returns': np.cumsum([0.01] * 10),
            'benchmark_returns': [0.005] * 10,
            'benchmark_cumulative': np.cumsum([0.005] * 10),
            'position': [1] * 10
        }, index=dates)
        
        strategy = BuyAndHoldStrategy()
        result = Result(strategy, data, [], 100000, 0.001, 0.0001)
        
        # Test avec backend non disponible
        plot_result = result.plot_performance(backend='nonexistent')
        assert plot_result is None
    
    def test_plot_trades_no_trades(self):
        """Test plot_trades sans trades."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'close': range(100, 110),
            'portfolio_value': range(100000, 110000, 1000),
            'returns': [0.01] * 10,
            'cumulative_returns': np.cumsum([0.01] * 10),
            'benchmark_returns': [0.005] * 10,
            'benchmark_cumulative': np.cumsum([0.005] * 10),
            'position': [1] * 10
        }, index=dates)
        
        strategy = BuyAndHoldStrategy()
        result = Result(strategy, data, [], 100000, 0.001, 0.0001)  # Pas de trades
        
        plot_result = result.plot_trades()
        assert plot_result is None


if __name__ == "__main__":
    pytest.main([__file__])