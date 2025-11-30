"""
Module contenant la classe Result pour analyser et visualiser les résultats de backtest.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import warnings

# Imports pour les visualisations
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib non disponible. Les visualisations seront limitées.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class Result:
    """
    Classe pour analyser et visualiser les résultats d'un backtest.
    
    Cette classe contient toutes les métriques de performance et les méthodes
    de visualisation pour analyser les résultats d'une stratégie.
    """
    
    def __init__(self, strategy, results_df: pd.DataFrame, trades: List[Dict],
                 initial_capital: float, transaction_cost: float, slippage: float):
        """
        Initialise l'objet Result.
        
        Args:
            strategy: La stratégie utilisée
            results_df (pd.DataFrame): DataFrame avec les résultats du backtest
            trades (List[Dict]): Liste des trades exécutés
            initial_capital (float): Capital initial
            transaction_cost (float): Coût de transaction
            slippage (float): Slippage
        """
        self.strategy = strategy
        self.results_df = results_df
        self.trades = trades
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Calcul des métriques
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calcule toutes les métriques de performance."""
        self.metrics = {}
        
        # Données de base
        returns = self.results_df['returns'].dropna()
        portfolio_values = self.results_df['portfolio_value']
        benchmark_returns = self.results_df['benchmark_returns'].dropna()
        
        # Performance totale
        self.metrics['total_return'] = (portfolio_values.iloc[-1] / self.initial_capital) - 1
        self.metrics['benchmark_total_return'] = (self.results_df['benchmark_cumulative'].iloc[-1])
        
        # Performance annualisée
        days = len(returns)
        years = days / 252  # 252 jours de trading par an
        self.metrics['annualized_return'] = (1 + self.metrics['total_return']) ** (1/years) - 1
        self.metrics['benchmark_annualized_return'] = (1 + self.metrics['benchmark_total_return']) ** (1/years) - 1
        
        # Volatilité
        self.metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualisée
        self.metrics['benchmark_volatility'] = benchmark_returns.std() * np.sqrt(252)
        
        # Ratio de Sharpe (on assume un taux sans risque de 0)
        if self.metrics['volatility'] > 0:
            self.metrics['sharpe_ratio'] = self.metrics['annualized_return'] / self.metrics['volatility']
        else:
            self.metrics['sharpe_ratio'] = 0
            
        if self.metrics['benchmark_volatility'] > 0:
            self.metrics['benchmark_sharpe_ratio'] = self.metrics['benchmark_annualized_return'] / self.metrics['benchmark_volatility']
        else:
            self.metrics['benchmark_sharpe_ratio'] = 0
        
        # Drawdown
        cumulative_returns = self.results_df['cumulative_returns']
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - rolling_max
        self.metrics['max_drawdown'] = drawdowns.min()
        
        # Calcul du drawdown du benchmark
        benchmark_cumulative = self.results_df['benchmark_cumulative']
        benchmark_rolling_max = benchmark_cumulative.expanding().max()
        benchmark_drawdowns = benchmark_cumulative - benchmark_rolling_max
        self.metrics['benchmark_max_drawdown'] = benchmark_drawdowns.min()
        
        # Ratio de Sortino (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std() * np.sqrt(252)
            if downside_std > 0:
                self.metrics['sortino_ratio'] = self.metrics['annualized_return'] / downside_std
            else:
                self.metrics['sortino_ratio'] = 0
        else:
            self.metrics['sortino_ratio'] = float('inf') if self.metrics['annualized_return'] > 0 else 0
        
        # Statistiques des trades
        if self.trades:
            trade_df = pd.DataFrame(self.trades)
            self.metrics['num_trades'] = len(self.trades)
            
            # Calcul des gains/pertes par trade
            trade_returns = []
            for i in range(1, len(self.trades)):
                prev_trade = self.trades[i-1]
                curr_trade = self.trades[i]
                if prev_trade['position_after'] != 0:
                    # Calcul du rendement entre les trades
                    price_change = (curr_trade['price'] - prev_trade['effective_price']) / prev_trade['effective_price']
                    trade_return = price_change * prev_trade['position_after']
                    trade_returns.append(trade_return)
            
            if trade_returns:
                winning_trades = [r for r in trade_returns if r > 0]
                self.metrics['winning_trades_pct'] = len(winning_trades) / len(trade_returns) * 100
                self.metrics['avg_trade_return'] = np.mean(trade_returns)
            else:
                self.metrics['winning_trades_pct'] = 0
                self.metrics['avg_trade_return'] = 0
                
            # Coûts de transaction totaux
            total_transaction_costs = sum(trade['transaction_cost'] for trade in self.trades)
            self.metrics['total_transaction_costs'] = total_transaction_costs
            self.metrics['transaction_costs_pct'] = total_transaction_costs / self.initial_capital * 100
        else:
            self.metrics['num_trades'] = 0
            self.metrics['winning_trades_pct'] = 0
            self.metrics['avg_trade_return'] = 0
            self.metrics['total_transaction_costs'] = 0
            self.metrics['transaction_costs_pct'] = 0
        
        # Informations Beta et Alpha vs benchmark
        if len(returns) > 1 and len(benchmark_returns) > 1:
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            if benchmark_variance > 0:
                self.metrics['beta'] = covariance / benchmark_variance
                self.metrics['alpha'] = self.metrics['annualized_return'] - (self.metrics['beta'] * self.metrics['benchmark_annualized_return'])
            else:
                self.metrics['beta'] = 0
                self.metrics['alpha'] = self.metrics['annualized_return']
        else:
            self.metrics['beta'] = 0
            self.metrics['alpha'] = self.metrics['annualized_return']
    
    def summary(self) -> pd.DataFrame:
        """
        Retourne un résumé des métriques de performance.
        
        Returns:
            pd.DataFrame: Table résumant les principales métriques
        """
        summary_data = {
            'Métrique': [
                'Rendement Total (%)',
                'Rendement Annualisé (%)',
                'Volatilité Annualisée (%)',
                'Ratio de Sharpe',
                'Ratio de Sortino',
                'Drawdown Maximum (%)',
                'Beta vs Benchmark',
                'Alpha vs Benchmark (%)',
                'Nombre de Trades',
                '% Trades Gagnants',
                'Coûts de Transaction (%)' 
            ],
            'Stratégie': [
                f"{self.metrics['total_return']:.2%}",
                f"{self.metrics['annualized_return']:.2%}",
                f"{self.metrics['volatility']:.2%}",
                f"{self.metrics['sharpe_ratio']:.2f}",
                f"{self.metrics['sortino_ratio']:.2f}",
                f"{self.metrics['max_drawdown']:.2%}",
                f"{self.metrics['beta']:.2f}",
                f"{self.metrics['alpha']:.2%}",
                f"{self.metrics['num_trades']}",
                f"{self.metrics['winning_trades_pct']:.1f}%",
                f"{self.metrics['transaction_costs_pct']:.2f}%"
            ],
            'Benchmark': [
                f"{self.metrics['benchmark_total_return']:.2%}",
                f"{self.metrics['benchmark_annualized_return']:.2%}",
                f"{self.metrics['benchmark_volatility']:.2%}",
                f"{self.metrics['benchmark_sharpe_ratio']:.2f}",
                '-',
                f"{self.metrics['benchmark_max_drawdown']:.2%}",
                '1.00',
                '0.00%',
                '-',
                '-',
                '0.00%'
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def plot_performance(self, backend: str = 'matplotlib', figsize: tuple = (12, 8)):
        """
        Trace la performance de la stratégie vs benchmark.
        
        Args:
            backend (str): Backend de visualisation ('matplotlib', 'seaborn', 'plotly')
            figsize (tuple): Taille de la figure pour matplotlib/seaborn
        """
        if backend == 'plotly' and PLOTLY_AVAILABLE:
            return self._plot_performance_plotly()
        elif backend in ['matplotlib', 'seaborn'] and MATPLOTLIB_AVAILABLE:
            return self._plot_performance_matplotlib(figsize, use_seaborn=(backend=='seaborn'))
        else:
            print(f"Backend '{backend}' non disponible. Backends disponibles: {self.get_available_backends()}")
            return None
    
    def _plot_performance_matplotlib(self, figsize: tuple, use_seaborn: bool = False):
        """Trace avec matplotlib/seaborn."""
        if use_seaborn and SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Performance cumulative
        ax1.plot(self.results_df.index, (1 + self.results_df['cumulative_returns']) * 100, 
                label=f'{self.strategy.name}', linewidth=2)
        ax1.plot(self.results_df.index, (1 + self.results_df['benchmark_cumulative']) * 100, 
                label='Benchmark', linewidth=2, alpha=0.7)
        ax1.set_title('Performance Cumulative')
        ax1.set_ylabel('Valeur du Portefeuille (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        cumulative_returns = self.results_df['cumulative_returns']
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) * 100
        
        ax2.fill_between(self.results_df.index, drawdowns, 0, alpha=0.5, color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Distribution des rendements
        returns = self.results_df['returns'].dropna() * 100
        ax3.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(returns.mean(), color='red', linestyle='--', label=f'Moyenne: {returns.mean():.2f}%')
        ax3.set_title('Distribution des Rendements Quotidiens')
        ax3.set_xlabel('Rendement (%)')
        ax3.set_ylabel('Fréquence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Positions dans le temps
        ax4.plot(self.results_df.index, self.results_df['position'], linewidth=1)
        ax4.set_title('Positions de la Stratégie')
        ax4.set_ylabel('Position (-1 à 1)')
        ax4.set_ylim(-1.1, 1.1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if plt.get_backend() != 'Agg':
            plt.show()
        return fig
    
    def _plot_performance_plotly(self):
        """Trace avec plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Cumulative', 'Drawdown', 
                          'Distribution des Rendements', 'Positions'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Performance cumulative
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=(1 + self.results_df['cumulative_returns']) * 100,
                name=self.strategy.name,
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=(1 + self.results_df['benchmark_cumulative']) * 100,
                name='Benchmark',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        cumulative_returns = self.results_df['cumulative_returns']
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) * 100
        
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=drawdowns,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red')
            ),
            row=1, col=2
        )
        
        # Distribution des rendements
        returns = self.results_df['returns'].dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Rendements',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Positions
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index,
                y=self.results_df['position'],
                name='Position',
                line=dict(width=1)
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Analyse de Performance")
        fig.show()
        return fig
    
    def plot_trades(self, backend: str = 'matplotlib'):
        """
        Visualise les points d'entrée et de sortie des trades.
        
        Args:
            backend (str): Backend de visualisation
        """
        if not self.trades:
            print("Aucun trade à visualiser.")
            return None
            
        if backend == 'plotly' and PLOTLY_AVAILABLE:
            return self._plot_trades_plotly()
        elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
            return self._plot_trades_matplotlib()
        else:
            print(f"Backend '{backend}' non disponible.")
            return None
    
    def _plot_trades_matplotlib(self):
        """Trace les trades avec matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prix
        ax.plot(self.results_df.index, self.results_df['close'], 
               label='Prix', linewidth=1, color='blue')
        
        # Points de trade
        trade_dates = [trade['date'] for trade in self.trades]
        trade_prices = [trade['price'] for trade in self.trades]
        trade_colors = ['green' if trade['position_after'] > trade['position_before'] 
                       else 'red' for trade in self.trades]
        
        ax.scatter(trade_dates, trade_prices, c=trade_colors, s=50, alpha=0.7, zorder=5)
        
        ax.set_title('Points d\'Entrée et de Sortie des Trades')
        ax.set_ylabel('Prix')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if plt.get_backend() != 'Agg':
            plt.show()
        return fig
    
    def _plot_trades_plotly(self):
        """Trace les trades avec plotly."""
        fig = go.Figure()
        
        # Prix
        fig.add_trace(go.Scatter(
            x=self.results_df.index,
            y=self.results_df['close'],
            name='Prix',
            line=dict(width=1, color='blue')
        ))
        
        # Trades d'achat
        buy_trades = [trade for trade in self.trades if trade['position_after'] > trade['position_before']]
        if buy_trades:
            fig.add_trace(go.Scatter(
                x=[trade['date'] for trade in buy_trades],
                y=[trade['price'] for trade in buy_trades],
                mode='markers',
                name='Achat',
                marker=dict(color='green', size=8)
            ))
        
        # Trades de vente
        sell_trades = [trade for trade in self.trades if trade['position_after'] < trade['position_before']]
        if sell_trades:
            fig.add_trace(go.Scatter(
                x=[trade['date'] for trade in sell_trades],
                y=[trade['price'] for trade in sell_trades],
                mode='markers',
                name='Vente',
                marker=dict(color='red', size=8)
            ))
        
        fig.update_layout(
            title='Points d\'Entrée et de Sortie des Trades',
            xaxis_title='Date',
            yaxis_title='Prix'
        )
        
        fig.show()
        return fig
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """Retourne la liste des backends disponibles."""
        backends = []
        if MATPLOTLIB_AVAILABLE:
            backends.append('matplotlib')
        if SEABORN_AVAILABLE:
            backends.append('seaborn')
        if PLOTLY_AVAILABLE:
            backends.append('plotly')
        return backends
    
    def __str__(self) -> str:
        return f"Result for {self.strategy.name}: {self.metrics['total_return']:.2%} total return"
    
    def __repr__(self) -> str:
        return self.__str__()


def compare_results(*results: Result, backend: str = 'matplotlib') -> Union[Figure, Any]:
    """
    Compare les résultats de plusieurs stratégies.
    
    Args:
        *results: Instances de Result à comparer
        backend (str): Backend de visualisation
        
    Returns:
        Figure ou objet plotly selon le backend
    """
    if len(results) < 2:
        raise ValueError("Au moins 2 résultats sont nécessaires pour la comparaison")
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        return _compare_results_plotly(results)
    elif backend in ['matplotlib', 'seaborn'] and MATPLOTLIB_AVAILABLE:
        return _compare_results_matplotlib(results, use_seaborn=(backend=='seaborn'))
    else:
        print(f"Backend '{backend}' non disponible.")
        return None


def _compare_results_matplotlib(results: tuple, use_seaborn: bool = False):
    """Compare avec matplotlib/seaborn."""
    if use_seaborn and SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Performance cumulative
    for result in results:
        cumulative = (1 + result.results_df['cumulative_returns']) * 100
        ax1.plot(result.results_df.index, cumulative, 
                label=result.strategy.name, linewidth=2)
    
    ax1.set_title('Comparaison des Performances Cumulatives')
    ax1.set_ylabel('Valeur du Portefeuille (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Comparaison des métriques
    metrics_comparison = []
    for result in results:
        metrics_comparison.append({
            'Stratégie': result.strategy.name,
            'Rendement Total': result.metrics['total_return'],
            'Sharpe Ratio': result.metrics['sharpe_ratio'],
            'Max Drawdown': result.metrics['max_drawdown']
        })
    
    df_metrics = pd.DataFrame(metrics_comparison)
    
    # Graphique en barres des rendements
    ax2.bar(df_metrics['Stratégie'], df_metrics['Rendement Total'] * 100)
    ax2.set_title('Rendements Totaux (%)')
    ax2.set_ylabel('Rendement (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Graphique en barres des ratios de Sharpe
    ax3.bar(df_metrics['Stratégie'], df_metrics['Sharpe Ratio'])
    ax3.set_title('Ratios de Sharpe')
    ax3.set_ylabel('Ratio de Sharpe')
    ax3.tick_params(axis='x', rotation=45)
    
    # Graphique en barres des drawdowns
    ax4.bar(df_metrics['Stratégie'], df_metrics['Max Drawdown'] * 100)
    ax4.set_title('Drawdowns Maximum (%)')
    ax4.set_ylabel('Drawdown (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if plt.get_backend() != 'Agg':
        plt.show()
    return fig


def _compare_results_plotly(results: tuple):
    """Compare avec plotly."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Cumulative', 'Rendements Totaux', 
                       'Ratios de Sharpe', 'Drawdowns Maximum'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Performance cumulative
    for result in results:
        cumulative = (1 + result.results_df['cumulative_returns']) * 100
        fig.add_trace(
            go.Scatter(
                x=result.results_df.index,
                y=cumulative,
                name=result.strategy.name,
                line=dict(width=2)
            ),
            row=1, col=1
        )
    
    # Métriques en barres
    strategy_names = [result.strategy.name for result in results]
    total_returns = [result.metrics['total_return'] * 100 for result in results]
    sharpe_ratios = [result.metrics['sharpe_ratio'] for result in results]
    max_drawdowns = [result.metrics['max_drawdown'] * 100 for result in results]
    
    fig.add_trace(
        go.Bar(x=strategy_names, y=total_returns, name='Rendement Total'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=strategy_names, y=sharpe_ratios, name='Sharpe Ratio'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=strategy_names, y=max_drawdowns, name='Max Drawdown'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Comparaison des Stratégies")
    fig.show()
    return fig