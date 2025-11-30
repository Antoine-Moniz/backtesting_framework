# Framework de Backtesting de Stratégies d'Investissement

Un framework Python flexible et extensible pour évaluer et comparer différentes stratégies d'investissement sur des données historiques.

## Caractéristiques

- 🚀 **Interface simple et intuitive** : Créez des stratégies en quelques lignes de code
- 📊 **Métriques complètes** : Plus de 15 métriques de performance incluant Sharpe, Sortino, drawdown, etc.
- 📈 **Visualisations avancées** : Support de matplotlib, seaborn et plotly
- 🔧 **Extensible** : Classe abstraite Strategy ou décorateur pour les stratégies simples
- 💰 **Réaliste** : Prise en compte des coûts de transaction et du slippage
- 📦 **Multi-actifs** : Support des stratégies sur un ou plusieurs actifs
- ⚡ **Performance** : Optimisé pour les gros volumes de données

## Installation

```bash
pip install -e .
```

Pour installer avec les dépendances de développement :

```bash
pip install -e ".[dev]"
```

## Utilisation rapide

```python
import pandas as pd
from backtesting_framework import Backtester, BuyAndHoldStrategy, MovingAverageCrossStrategy

# Chargement des données (CSV, Parquet ou DataFrame)
backtester = Backtester('data.csv', initial_capital=100000)

# Création d'une stratégie
strategy = MovingAverageCrossStrategy(short_window=10, long_window=30)

# Exécution du backtest
result = backtester.run_backtest(strategy)

# Affichage des résultats
print(result.summary())
result.plot_performance()
```

## Création de stratégies personnalisées

### Méthode 1 : Héritage de la classe Strategy

```python
from backtesting_framework import Strategy

class CustomStrategy(Strategy):
    def __init__(self):
        super().__init__("Ma Stratégie Custom")
    
    def get_position(self, historical_data, current_position):
        # Votre logique ici
        if len(historical_data) < 20:
            return 0
        
        # Exemple : stratégie RSI
        rsi = calculate_rsi(historical_data['close'])
        if rsi < 30:
            return 1.0  # Achat
        elif rsi > 70:
            return -1.0  # Vente
        else:
            return 0.0  # Neutre
```

### Méthode 2 : Décorateur pour stratégies simples

```python
from backtesting_framework import strategy_decorator

@strategy_decorator(name="Ma Stratégie Simple")
def simple_strategy(historical_data, current_position):
    if len(historical_data) < 10:
        return 0
    
    # Logique simple de moyenne mobile
    short_ma = historical_data['close'].rolling(5).mean().iloc[-1]
    long_ma = historical_data['close'].rolling(10).mean().iloc[-1]
    
    return 1 if short_ma > long_ma else -1
```

## Comparaison de stratégies

```python
from backtesting_framework import compare_results

# Création de plusieurs stratégies
buy_hold = BuyAndHoldStrategy()
ma_cross = MovingAverageCrossStrategy(5, 20)
custom = CustomStrategy()

# Exécution des backtests
result1 = backtester.run_backtest(buy_hold)
result2 = backtester.run_backtest(ma_cross)
result3 = backtester.run_backtest(custom)

# Comparaison
compare_results(result1, result2, result3, backend='plotly')
```

## Stratégies intégrées

### Stratégies prêtes à l'emploi
- **BuyAndHoldStrategy** : Stratégie passive d'achat-conservation
- **MovingAverageCrossStrategy** : Croisement de moyennes mobiles
- **MeanReversionStrategy** : Retour à la moyenne avec bandes de Bollinger

### Exemples de stratégies personnalisées
- **RSIStrategy** : Basée sur l'indicateur RSI
- **MomentumStrategy** : Stratégie de momentum avec décorateur
- **SimpleMLStrategy** : Stratégie avec features techniques

## Métriques disponibles

- **Performance** : Rendement total, annualisé
- **Risque** : Volatilité, VaR, drawdown maximum
- **Ratios** : Sharpe, Sortino, Calmar
- **Analyse vs benchmark** : Alpha, Beta, corrélation
- **Trading** : Nombre de trades, % trades gagnants, coûts de transaction

## Technologies utilisées

- **Python 3.8+** : Langage principal
- **pandas** : Manipulation de données financières
- **numpy** : Calculs numériques optimisés
- **matplotlib/seaborn/plotly** : Visualisations interactives
- **pytest** : Tests unitaires (51 tests couvrant tous les composants)
- **setuptools** : Packaging professionnel

## Architecture orientée objet

- **Polymorphisme** : Interface Strategy commune pour toutes les stratégies
- **Encapsulation** : Données et méthodes groupées logiquement
- **Héritage** : Stratégies héritent de la classe abstraite Strategy
- **Abstraction** : Complexité cachée derrière une API simple

## Structure du projet

```
backtesting_framework/
├── __init__.py          # Point d'entrée du package
├── strategy.py          # Classes Strategy et décorateurs
├── backtester.py        # Moteur de backtesting
└── result.py           # Analyse et visualisation des résultats

tests/                   # Tests unitaires
examples/               # Notebooks d'exemple
pyproject.toml          # Configuration du package
```

## Formats de données supportés

Le framework accepte :
- **DataFrames pandas** avec colonnes : date (index), close (obligatoire), open, high, low, volume (optionnelles)
- **Fichiers CSV** avec les mêmes colonnes
- **Fichiers Parquet** avec les mêmes colonnes

Exemple de format attendu :
```
date,open,high,low,close,volume
2023-01-01,100.0,102.0,99.0,101.0,1000000
2023-01-02,101.0,103.0,100.5,102.5,1200000
...
```

## Configuration avancée

```python
# Configuration personnalisée du backtester
backtester = Backtester(
    data='data.csv',
    initial_capital=100000,
    transaction_cost=0.001,  # 0.1% par trade
    slippage=0.0001         # 0.01% de slippage
)

# Stratégie avec fréquence de rééquilibrage
strategy = MovingAverageCrossStrategy(
    short_window=10, 
    long_window=30,
    rebalance_frequency='W'  # Hebdomadaire
)

# Backtest sur période spécifique
result = backtester.run_backtest(
    strategy,
    start_date='2023-01-01',
    end_date='2023-12-31',
    benchmark='SPY'  # Colonne benchmark
)
```

## Tests

```bash
pytest tests/
```

Avec couverture :
```bash
pytest tests/ --cov=backtesting_framework
```

## Exemple complet

Voir le notebook `examples/example_usage.ipynb` pour un exemple complet d'utilisation du framework.