# Investment Strategy Backtesting Framework

## Overview

This project implements a comprehensive and extensible backtesting framework designed to evaluate and compare investment strategies on historical financial data. Developed as part of the M2 Quantitative Finance program at Université Paris Dauphine-PSL, this framework provides researchers, students, and practitioners with professional-grade tools for strategy development, testing, and analysis.

The framework adopts a modular, object-oriented architecture that balances ease of use with powerful functionality. Users can create custom strategies through either class inheritance or decorator-based approaches, test them against historical data with realistic transaction costs and slippage, and analyze results through comprehensive metrics and visualizations.

## Features

**Core Capabilities:**
- Simple and intuitive API for strategy creation and testing
- Support for multiple data formats (CSV, Parquet, pandas DataFrame)
- Comprehensive performance metrics
- Multi-backend visualization support (matplotlib, seaborn, plotly)
- Realistic trading simulation with transaction costs and slippage
- Multi-asset portfolio support through dictionary-based position API
- Flexible rebalancing frequency configuration (daily, weekly, monthly)

**Advanced Features:**
- Walk-forward analysis for temporal validation
- Parameter optimization and robustness testing
- Strategy comparison and correlation analysis
- Benchmark-relative metrics (alpha, beta)
- Customizable risk-free rates and market benchmarks

## Installation & Usage

### Installation

Clone the repository and install the package:

```bash
git clone https://github.com/Antoine-Moniz/backtesting_framework.git
cd backtesting_framework
pip install -e .
```

For development with testing dependencies:

```bash
pip install -e ".[dev]"
```

### Quick Start Example

```python
import pandas as pd
from backtesting_framework import Backtester, Strategy
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.moving_average_cross import MovingAverageCrossStrategy

# Load historical data (CSV, Parquet, or DataFrame)
data = pd.read_csv('data.csv')
backtester = Backtester(
    data=data,
    initial_capital=100000,
    transaction_cost=0.001,  # 0.1% per trade
    slippage=0.0005          # 0.05% slippage
)

# Create and test a strategy
strategy = MovingAverageCrossStrategy(short_window=10, long_window=30)
result = backtester.run_backtest(strategy)

# Display results
print(result.summary())
result.plot_performance(backend='matplotlib')
```

### Creating Custom Strategies

**Method 1: Class Inheritance**

```python
from backtesting_framework import Strategy

class RSIStrategy(Strategy):
    def __init__(self, window=14, oversold=30, overbought=70):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self.is_fitted = False
        self.rebalance_frequency = 'D'
    
    @property
    def name(self):
        return f"RSI Strategy ({self.window}, {self.oversold}, {self.overbought})"
    
    def fit(self, historical_data):
        self.is_fitted = True
    
    def get_position(self, data, positions):
        if len(data) < self.window + 5:
            return {'asset': 0.0}
        
        rsi = self.calculate_rsi(data['close'], self.window)
        
        if rsi < self.oversold:
            return {'asset': 1.0}  # Buy (oversold)
        elif rsi > self.overbought:
            return {'asset': 0.0}  # Sell (overbought)
        else:
            return positions  # Keep current position
```

**Method 2: Decorator for Simple Strategies**

```python
from backtesting_framework import strategy_decorator

@strategy_decorator(name="Simple MA Strategy")
def simple_strategy(historical_data, current_position):
    if len(historical_data) < 10:
        return 0
    
    # Simple moving average logic
    short_ma = historical_data['close'].rolling(5).mean().iloc[-1]
    long_ma = historical_data['close'].rolling(10).mean().iloc[-1]
    
    return 1 if short_ma > long_ma else -1
```

### Comparing Multiple Strategies

```python
from backtesting_framework import compare_results

# Create multiple strategies
buy_hold = BuyAndHoldStrategy()
ma_cross = MovingAverageCrossStrategy(short_window=5, long_window=20)
rsi = RSIStrategy()

# Run backtests
result1 = backtester.run_backtest(buy_hold)
result2 = backtester.run_backtest(ma_cross)
result3 = backtester.run_backtest(rsi)

# Compare visually
compare_results(result1, result2, result3, backend='matplotlib')
```

For detailed examples including walk-forward analysis, parameter optimization, and advanced visualizations, see `examples/example_usage.ipynb`.

## Project Structure

```
backtesting_framework/
├── backtesting_framework/
│   ├── __init__.py              # Package exports and initialization
│   ├── strategy.py              # Abstract Strategy class and decorators
│   ├── backtester.py            # Core backtesting engine
│   ├── result.py                # Results analysis and visualization
│   └── data_handler.py          # Data loading and validation utilities
│
├── examples/
│   ├── strategies/              # Built-in strategy implementations
│   │   ├── buy_and_hold.py
│   │   ├── moving_average_cross.py
│   │   └── mean_reversion.py
│   └── example_usage.ipynb      # Comprehensive usage demonstration
│
├── tests/
│   ├── test_backtester.py       # Backtester unit tests
│   ├── test_strategy.py         # Strategy tests
│   ├── test_result.py           # Results analysis tests
│   └── test_multi_asset.py      # Multi-asset support tests
│
├── pyproject.toml               # Package configuration and dependencies
├── README.md                    # Project documentation
└── LICENSE                      # MIT License
```

## Methodology Summary

### Object-Oriented Architecture

The framework implements core object-oriented programming principles:

- **Abstraction**: The abstract `Strategy` class defines the interface all strategies must implement, hiding implementation complexity behind a simple API.
- **Inheritance**: Custom strategies inherit from `Strategy`, gaining access to common functionality while implementing specific trading logic.
- **Polymorphism**: All strategies share the same interface (`get_position`, `fit`), allowing the `Backtester` to work with any strategy implementation.
- **Encapsulation**: Each class encapsulates its data and methods, exposing only necessary interfaces.

### Backtesting Process

1. **Data Preparation**: Load and validate historical data (OHLCV format)
2. **Strategy Initialization**: Configure strategy parameters and training
3. **Simulation Loop**: Iterate through historical data, generating positions
4. **Position Management**: Apply transaction costs, slippage, and rebalancing
5. **Metrics Calculation**: Compute comprehensive performance statistics
6. **Visualization**: Generate interactive charts and comparison plots

### Performance Metrics

The framework calculates 15+ performance metrics:

- **Returns**: Total, annualized, cumulative
- **Risk Measures**: Volatility, Value at Risk (VaR), maximum drawdown
- **Risk-Adjusted Ratios**: Sharpe, Sortino, Calmar
- **Benchmark Analysis**: Alpha, Beta, correlation
- **Trading Statistics**: Number of trades, win rate, transaction costs

### Testing Strategy

Comprehensive test coverage includes:

- **Unit Tests**: Individual component validation
- **Integration Tests**: Cross-component interaction verification
- **Multi-Asset Tests**: Portfolio strategy validation
- **Edge Cases**: Boundary condition handling

## Reports

### Key Deliverables

- **Technical Implementation**: Complete source code with modular architecture
- **Documentation**: Comprehensive README, docstrings, and inline comments
- **Example Notebook**: Detailed Jupyter notebook (`example_usage.ipynb`) demonstrating:
  - Data loading from Yahoo Finance
  - Built-in strategy testing (Buy & Hold, MA Cross, Mean Reversion)
  - Custom strategy creation (RSI, Momentum, Simple ML)
  - Performance comparison and visualization
  - Walk-forward analysis and parameter optimization
  - Professional conclusions and insights

- **Test Suite**: Unit and integration tests covering all components
- **Package Configuration**: Professional `pyproject.toml` enabling pip installation

### Performance Results

Testing on Apple (AAPL) stock from 2020-2023 yielded:

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Trades |
|----------|--------------|--------------|--------------|--------|
| Momentum | 151.55% | 1.21 | -51.35% | 1005 |
| Simple ML | 19.06% | 0.29 | -32.37% | 1005 |
| RSI | 2.47% | 0.02 | -34.82% | 1005 |

Results demonstrate the framework's capability to differentiate strategy performance and provide actionable insights for investment decision-making.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors / Contact

**M2 Quantitative Finance - Université Paris Dauphine-PSL**

* [Mariano BENJAMIN](mailto:mariano.benjamin@dauphine.eu)
* [Noah CHIKHI](mailto:noah.chikhi@dauphine.eu)
* [Antoine Moniz](mailto:antoine.moniz@dauphine.eu)

For issues, discussions, or contributions, please open an issue or pull request on the project's GitHub page