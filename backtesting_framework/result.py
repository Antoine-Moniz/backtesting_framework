"""Performance analysis and visualization toolkit.

This module provides the `Result` class for analyzing backtest outcomes and
the `compare_results` function for multi-strategy comparison. The Result class
calculates key performance metrics (returns, Sharpe ratio, drawdowns) and
generates visualizations using multiple backends (matplotlib, seaborn, plotly).

The module supports both single-strategy analysis and side-by-side comparison
of multiple strategies, making it easy to evaluate and select the best performing
approaches.

Notes
-----
Performance metrics are calculated using standard financial formulas. The Sharpe
ratio assumes 252 trading days per year for annualization. Maximum drawdown is
computed as the largest peak-to-trough decline in cumulative returns.

Visualization backends can be selected via the `backend` parameter. If a requested
backend is unavailable, the module falls back to matplotlib or raises an error if
no visualization library is installed.

Authors
-------
Mariano Benjamin
Noah Chikhi
Antoine Moniz
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import warnings

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualizations will be limited.")

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
    Class for analyzing and visualizing backtest results.
    
    This class contains all performance metrics and visualization methods
    for analyzing strategy results.
    """
    
    def __init__(self, strategy, results_df: pd.DataFrame,
                 initial_capital: float,
                 trades: Optional[List[Dict]] = None,
                 transaction_cost: float = 0.0,
                 slippage: float = 0.0,
                 asset_symbols: Optional[List[str]] = None):
        """
        Initialize the Result object.
        
        Parameters
        ----------
        strategy : Strategy
            Strategy used for the backtest.
        results_df : pd.DataFrame
            DataFrame with backtest results.
        trades : List[Dict]
            List of executed trades.
        initial_capital : float
            Initial capital.
        transaction_cost : float
            Transaction cost.
        slippage : float
            Slippage.
        asset_symbols : List[str], optional
            List of asset symbols in the portfolio.
        """
        self.strategy = strategy
        self.results_df = results_df
        self.trades = trades or []
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.asset_symbols = asset_symbols or ['asset']
        self.is_multi_asset = len(self.asset_symbols) > 1
        
        # Calculate metrics
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate all performance metrics."""
        self.metrics = {}
        
        # Base data
        returns = self.results_df['returns'].dropna()
        portfolio_values = self.results_df['portfolio_value']
        benchmark_returns = self.results_df['benchmark_returns'].dropna()
        
        # Total performance
        self.metrics['total_return'] = (portfolio_values.iloc[-1] / self.initial_capital) - 1
        self.metrics['benchmark_total_return'] = (self.results_df['benchmark_cumulative'].iloc[-1])
        
        # Annualized performance
        days = len(returns)
        years = days / 252  # 252 trading days per year
        self.metrics['annualized_return'] = (1 + self.metrics['total_return']) ** (1/years) - 1
        self.metrics['benchmark_annualized_return'] = (1 + self.metrics['benchmark_total_return']) ** (1/years) - 1
        
        # Volatility
        self.metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
        self.metrics['benchmark_volatility'] = benchmark_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming risk-free rate of 0)
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
        
        # Benchmark drawdown calculation
        benchmark_cumulative = self.results_df['benchmark_cumulative']
        benchmark_rolling_max = benchmark_cumulative.expanding().max()
        benchmark_drawdowns = benchmark_cumulative - benchmark_rolling_max
        self.metrics['benchmark_max_drawdown'] = benchmark_drawdowns.min()
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std() * np.sqrt(252)
            if downside_std > 0:
                self.metrics['sortino_ratio'] = self.metrics['annualized_return'] / downside_std
            else:
                self.metrics['sortino_ratio'] = 0
        else:
            self.metrics['sortino_ratio'] = float('inf') if self.metrics['annualized_return'] > 0 else 0
        
        # Trade statistics
        if self.trades:
            trade_df = pd.DataFrame(self.trades)
            self.metrics['num_trades'] = len(self.trades)
            
            # Calculate gains/losses per trade
            trade_returns = []
            for i in range(1, len(self.trades)):
                prev_trade = self.trades[i-1]
                curr_trade = self.trades[i]
                if prev_trade['position_after'] != 0:
                    # Calculate return between trades
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
                
            # Total transaction costs
            total_transaction_costs = sum(trade['transaction_cost'] for trade in self.trades)
            self.metrics['total_transaction_costs'] = total_transaction_costs
            self.metrics['transaction_costs_pct'] = total_transaction_costs / self.initial_capital * 100
        else:
            self.metrics['num_trades'] = 0
            self.metrics['winning_trades_pct'] = 0
            self.metrics['avg_trade_return'] = 0
            self.metrics['total_transaction_costs'] = 0
            self.metrics['transaction_costs_pct'] = 0
        
        # Beta and Alpha information vs benchmark
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
        
        return self.metrics
    
    def summary(self) -> pd.DataFrame:
        """
        Return a summary of performance metrics.
        
        Returns
        -------
        pd.DataFrame
            Table summarizing key metrics.
        """
        summary_data = {
            'Metric': [
                'Total Return (%)',
                'Annualized Return (%)',
                'Annualized Volatility (%)',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Maximum Drawdown (%)',
                'Beta vs Benchmark',
                'Alpha vs Benchmark (%)',
                'Number of Trades',
                '% Winning Trades',
                'Transaction Costs (%)' 
            ],
            'Strategy': [
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
        Plot strategy performance vs benchmark.
        
        Parameters
        ----------
        backend : str, default='matplotlib'
            Visualization backend ('matplotlib', 'seaborn', 'plotly').
        figsize : tuple, default=(12, 8)
            Figure size for matplotlib/seaborn.
        """
        if backend == 'plotly' and PLOTLY_AVAILABLE:
            return self._plot_performance_plotly()
        elif backend in ['matplotlib', 'seaborn'] and MATPLOTLIB_AVAILABLE:
            return self._plot_performance_matplotlib(figsize, use_seaborn=(backend=='seaborn'))
        else:
            print(f"Backend '{backend}' not available. Available backends: {self.get_available_backends()}")
            return None
    
    def _plot_performance_matplotlib(self, figsize: tuple, use_seaborn: bool = False):
        """Plot with matplotlib/seaborn."""
        if use_seaborn and SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Cumulative performance
        ax1.plot(self.results_df.index, (1 + self.results_df['cumulative_returns']) * 100, 
                label=f'{self.strategy.name}', linewidth=2)
        ax1.plot(self.results_df.index, (1 + self.results_df['benchmark_cumulative']) * 100, 
                label='Benchmark', linewidth=2, alpha=0.7)
        ax1.set_title('Cumulative Performance')
        ax1.set_ylabel('Portfolio Value (%)')
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
        
        # Returns distribution
        returns = self.results_df['returns'].dropna() * 100
        ax3.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Positions over time
        ax4.plot(self.results_df.index, self.results_df['position'], linewidth=1)
        ax4.set_title('Strategy Positions')
        ax4.set_ylabel('Position (-1 to 1)')
        ax4.set_ylim(-1.1, 1.1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if plt.get_backend() != 'Agg':
            plt.show()
        return fig
    
    def _plot_performance_plotly(self):
        """Plot with plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Performance', 'Drawdown', 
                          'Returns Distribution', 'Positions'),
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
        
        # Returns distribution
        returns = self.results_df['returns'].dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
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
        
        fig.update_layout(height=600, showlegend=True, title_text="Performance Analysis")
        fig.show()
        return fig
    
    def plot_positions(self, backend: str = 'matplotlib', figsize: tuple = (12, 6)):
        """
        Plot per-asset position allocation over time.
        
        Parameters
        ----------
        backend : str, default='matplotlib'
            Visualization backend ('matplotlib', 'seaborn', 'plotly').
        figsize : tuple, default=(12, 6)
            Figure size for matplotlib/seaborn.
        """
        if not self.is_multi_asset:
            print("Single-asset portfolio. Use plot_performance() to view position.")
            return None
        
        if backend == 'plotly' and PLOTLY_AVAILABLE:
            return self._plot_positions_plotly()
        elif backend in ['matplotlib', 'seaborn'] and MATPLOTLIB_AVAILABLE:
            return self._plot_positions_matplotlib(figsize, use_seaborn=(backend=='seaborn'))
        else:
            print(f"Backend '{backend}' not available.")
            return None
    
    def _plot_positions_matplotlib(self, figsize: tuple, use_seaborn: bool = False):
        """Plot per-asset positions with matplotlib/seaborn."""
        if use_seaborn and SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Stacked area chart of positions
        position_cols = [f'position_{symbol}' for symbol in self.asset_symbols]
        position_data = self.results_df[position_cols]
        
        # Rename columns for legend
        position_data.columns = self.asset_symbols
        
        # Plot stacked area
        position_data.plot.area(ax=ax1, alpha=0.7, stacked=True)
        ax1.set_title('Asset Position Allocation Over Time')
        ax1.set_ylabel('Position Allocation')
        ax1.legend(loc='upper left', title='Assets')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1.1, 1.1)
        
        # Individual position lines
        for symbol in self.asset_symbols:
            ax2.plot(self.results_df.index, self.results_df[f'position_{symbol}'], 
                    label=symbol, linewidth=2)
        
        ax2.set_title('Individual Asset Positions')
        ax2.set_ylabel('Position (-1 to 1)')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        if plt.get_backend() != 'Agg':
            plt.show()
        return fig
    
    def _plot_positions_plotly(self):
        """Plot per-asset positions with plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Asset Position Allocation', 'Individual Asset Positions'),
            vertical_spacing=0.12
        )
        
        # Stacked area chart
        for symbol in self.asset_symbols:
            fig.add_trace(
                go.Scatter(
                    x=self.results_df.index,
                    y=self.results_df[f'position_{symbol}'],
                    name=symbol,
                    stackgroup='one',
                    mode='none',
                    fillcolor=None
                ),
                row=1, col=1
            )
        
        # Individual lines
        for symbol in self.asset_symbols:
            fig.add_trace(
                go.Scatter(
                    x=self.results_df.index,
                    y=self.results_df[f'position_{symbol}'],
                    name=symbol,
                    line=dict(width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Position Allocation", row=1, col=1)
        fig.update_yaxes(title_text="Position (-1 to 1)", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, title_text="Multi-Asset Position Analysis")
        fig.show()
        return fig
    
    def get_position_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each asset's position allocation.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics per asset.
        """
        if not self.is_multi_asset:
            return pd.DataFrame({
                'Asset': [self.asset_symbols[0]],
                'Mean Position': [self.results_df['position'].mean()],
                'Min Position': [self.results_df['position'].min()],
                'Max Position': [self.results_df['position'].max()],
                'Position Std': [self.results_df['position'].std()]
            })
        
        summary_data = []
        for symbol in self.asset_symbols:
            pos_col = f'position_{symbol}'
            summary_data.append({
                'Asset': symbol,
                'Mean Position': self.results_df[pos_col].mean(),
                'Min Position': self.results_df[pos_col].min(),
                'Max Position': self.results_df[pos_col].max(),
                'Position Std': self.results_df[pos_col].std(),
                '% Time Long': (self.results_df[pos_col] > 0).sum() / len(self.results_df) * 100,
                '% Time Short': (self.results_df[pos_col] < 0).sum() / len(self.results_df) * 100,
                '% Time Neutral': (self.results_df[pos_col] == 0).sum() / len(self.results_df) * 100
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_trades(self, backend: str = 'matplotlib'):
        """
        Visualize entry and exit points of trades.
        
        Parameters
        ----------
        backend : str, default='matplotlib'
            Visualization backend.
        """
        if not self.trades:
            print("No trades to visualize.")
            return None
            
        if backend == 'plotly' and PLOTLY_AVAILABLE:
            return self._plot_trades_plotly()
        elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
            return self._plot_trades_matplotlib()
        else:
            print(f"Backend '{backend}' not available.")
            return None
    
    def _plot_trades_matplotlib(self):
        """Plot trades with matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Price
        ax.plot(self.results_df.index, self.results_df['close'], 
               label='Price', linewidth=1, color='blue')
        
        # Trade points
        trade_dates = [trade['date'] for trade in self.trades]
        trade_prices = [trade['price'] for trade in self.trades]
        trade_colors = ['green' if trade['position_after'] > trade['position_before'] 
                       else 'red' for trade in self.trades]
        
        ax.scatter(trade_dates, trade_prices, c=trade_colors, s=50, alpha=0.7, zorder=5)
        
        ax.set_title('Trade Entry and Exit Points')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if plt.get_backend() != 'Agg':
            plt.show()
        return fig
    
    def _plot_trades_plotly(self):
        """Plot trades with plotly."""
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Scatter(
            x=self.results_df.index,
            y=self.results_df['close'],
            name='Price',
            line=dict(width=1, color='blue')
        ))
        
        # Buy trades
        buy_trades = [trade for trade in self.trades if trade['position_after'] > trade['position_before']]
        if buy_trades:
            fig.add_trace(go.Scatter(
                x=[trade['date'] for trade in buy_trades],
                y=[trade['price'] for trade in buy_trades],
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=8)
            ))
        
        # Sell trades
        sell_trades = [trade for trade in self.trades if trade['position_after'] < trade['position_before']]
        if sell_trades:
            fig.add_trace(go.Scatter(
                x=[trade['date'] for trade in sell_trades],
                y=[trade['price'] for trade in sell_trades],
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=8)
            ))
        
        fig.update_layout(
            title='Trade Entry and Exit Points',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        
        fig.show()
        return fig
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """Return the list of available backends."""
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
    Compare results from multiple strategies.
    
    Parameters
    ----------
    *results : Result
        Result instances to compare.
    backend : str, default='matplotlib'
        Visualization backend.
        
    Returns
    -------
    Figure or plotly object
        Comparison visualization.
    """
    if len(results) < 2:
        raise ValueError("At least 2 results are required for comparison")
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        return _compare_results_plotly(results)
    elif backend in ['matplotlib', 'seaborn'] and MATPLOTLIB_AVAILABLE:
        return _compare_results_matplotlib(results, use_seaborn=(backend=='seaborn'))
    else:
        print(f"Backend '{backend}' not available.")
        return None


def _compare_results_matplotlib(results: tuple, use_seaborn: bool = False):
    """Compare with matplotlib/seaborn."""
    if use_seaborn and SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cumulative performance
    for result in results:
        cumulative = (1 + result.results_df['cumulative_returns']) * 100
        ax1.plot(result.results_df.index, cumulative, 
                label=result.strategy.name, linewidth=2)
    
    ax1.set_title('Cumulative Performance Comparison')
    ax1.set_ylabel('Portfolio Value (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metrics comparison
    metrics_comparison = []
    for result in results:
        metrics_comparison.append({
            'Strategy': result.strategy.name,
            'Total Return': result.metrics['total_return'],
            'Sharpe Ratio': result.metrics['sharpe_ratio'],
            'Max Drawdown': result.metrics['max_drawdown']
        })
    
    df_metrics = pd.DataFrame(metrics_comparison)
    
    # Bar chart of returns
    ax2.bar(df_metrics['Strategy'], df_metrics['Total Return'] * 100)
    ax2.set_title('Total Returns (%)')
    ax2.set_ylabel('Return (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Bar chart of Sharpe ratios
    ax3.bar(df_metrics['Strategy'], df_metrics['Sharpe Ratio'])
    ax3.set_title('Sharpe Ratios')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.tick_params(axis='x', rotation=45)
    
    # Bar chart of drawdowns
    ax4.bar(df_metrics['Strategy'], df_metrics['Max Drawdown'] * 100)
    ax4.set_title('Maximum Drawdowns (%)')
    ax4.set_ylabel('Drawdown (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if plt.get_backend() != 'Agg':
        plt.show()
    return fig


def _compare_results_plotly(results: tuple):
    """Compare with plotly."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cumulative Performance', 'Total Returns', 
                       'Sharpe Ratios', 'Maximum Drawdowns'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cumulative performance
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
    
    # Bar chart metrics
    strategy_names = [result.strategy.name for result in results]
    total_returns = [result.metrics['total_return'] * 100 for result in results]
    sharpe_ratios = [result.metrics['sharpe_ratio'] for result in results]
    max_drawdowns = [result.metrics['max_drawdown'] * 100 for result in results]
    
    fig.add_trace(
        go.Bar(x=strategy_names, y=total_returns, name='Total Return'),
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
    
    fig.update_layout(height=600, showlegend=True, title_text="Strategy Comparison")
    fig.show()
    return fig