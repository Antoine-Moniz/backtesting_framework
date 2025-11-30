"""
Module contenant la classe Backtester pour exécuter les backtests de stratégies.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
from pathlib import Path
import warnings

from .strategy import Strategy
from .result import Result


class Backtester:
    """
    Classe principale pour exécuter des backtests de stratégies d'investissement.
    
    Cette classe prend des données historiques et permet d'exécuter des stratégies
    pour évaluer leur performance.
    """
    
    def __init__(self, data: Union[pd.DataFrame, str, Path], 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0001):
        """
        Initialise le backtester avec des données historiques.
        
        Args:
            data (Union[pd.DataFrame, str, Path]): Données historiques ou chemin vers fichier CSV/Parquet
            initial_capital (float): Capital initial pour le backtest
            transaction_cost (float): Coût de transaction en pourcentage (0.001 = 0.1%)
            slippage (float): Slippage en pourcentage (0.0001 = 0.01%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Chargement des données
        self.data = self._load_data(data)
        self._validate_data()
        
    def _load_data(self, data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        Charge les données depuis différents formats.
        
        Args:
            data: Données à charger
            
        Returns:
            pd.DataFrame: Données chargées et formatées
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        file_path = Path(data)
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
            
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Format de fichier non supporté: {file_path.suffix}")
            
        return df
    
    def _validate_data(self):
        """
        Valide que les données contiennent les colonnes requises.
        """
        required_cols = ['close']
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        elif not isinstance(self.data.index, pd.DatetimeIndex):
            # Si pas de colonne date et index pas datetime, on crée un index temporel
            warnings.warn("Aucune colonne 'date' trouvée et index pas datetime. "
                         "Création d'un index temporel artificiel.")
            date_range = pd.date_range(start='2020-01-01', periods=len(self.data), freq='D')
            self.data.index = date_range
            
        # Vérification des colonnes requises
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans les données: {missing_cols}")
            
        # Ajout des colonnes optionnelles si manquantes
        if 'open' not in self.data.columns:
            self.data['open'] = self.data['close'].shift(1).fillna(self.data['close'])
        if 'high' not in self.data.columns:
            self.data['high'] = self.data['close']
        if 'low' not in self.data.columns:
            self.data['low'] = self.data['close']
        if 'volume' not in self.data.columns:
            self.data['volume'] = 1000000  # Volume artificiel
            
        # Tri par date
        self.data.sort_index(inplace=True)
        
    def run_backtest(self, strategy: Strategy, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    benchmark: Optional[str] = None) -> Result:
        """
        Exécute un backtest avec la stratégie donnée.
        
        Args:
            strategy (Strategy): Stratégie à tester
            start_date (str, optional): Date de début (format YYYY-MM-DD)
            end_date (str, optional): Date de fin (format YYYY-MM-DD)
            benchmark (str, optional): Nom de la colonne à utiliser comme benchmark
            
        Returns:
            Result: Résultats du backtest
        """
        # Préparation des données
        data_subset = self._prepare_data_subset(start_date, end_date)
        
        # Entraînement de la stratégie si nécessaire
        if hasattr(strategy, 'fit') and not strategy.is_fitted:
            # Utilise les premières 60% des données pour l'entraînement
            train_size = int(len(data_subset) * 0.6)
            train_data = data_subset.iloc[:train_size]
            strategy.fit(train_data)
            
        # Variables pour le backtest
        positions = []
        portfolio_values = [self.initial_capital]
        trades = []
        current_position = 0.0
        current_shares = 0.0
        cash = self.initial_capital
        
        # Calcul des dates de rééquilibrage
        rebalance_dates = self._get_rebalance_dates(data_subset, strategy.rebalance_frequency)
        
        # Boucle principale du backtest
        for i, (date, row) in enumerate(data_subset.iterrows()):
            # Données historiques disponibles jusqu'à cette date
            historical_data = data_subset.iloc[:i+1]
            
            # Vérification si c'est une date de rééquilibrage
            should_rebalance = date in rebalance_dates or i == 0
            
            if should_rebalance and len(historical_data) > 1:
                # Demande à la stratégie la nouvelle position
                try:
                    new_position = strategy.get_position(historical_data, current_position)
                    new_position = np.clip(new_position, -1.0, 1.0)  # Limite entre -1 et 1
                except Exception as e:
                    warnings.warn(f"Erreur dans la stratégie à la date {date}: {e}")
                    new_position = current_position
                    
                # Exécution du trade si changement de position
                if abs(new_position - current_position) > 0.001:  # Seuil de tolérance
                    trade_info = self._execute_trade(
                        current_position, new_position, row['close'], 
                        cash, current_shares, date
                    )
                    if trade_info:
                        trades.append(trade_info)
                        cash = trade_info['cash_after']
                        current_shares = trade_info['shares_after']
                        current_position = new_position
            
            # Calcul de la valeur du portefeuille
            portfolio_value = cash + current_shares * row['close']
            portfolio_values.append(portfolio_value)
            positions.append(current_position)
            
        # Création du DataFrame des résultats
        results_df = pd.DataFrame({
            'date': data_subset.index,
            'close': data_subset['close'].values,
            'position': positions,
            'portfolio_value': portfolio_values[1:]  # Exclut la valeur initiale
        })
        results_df.set_index('date', inplace=True)
        
        # Calcul des rendements
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_returns'] = (results_df['portfolio_value'] / self.initial_capital) - 1
        
        # Benchmark (Buy and Hold par défaut)
        if benchmark and benchmark in data_subset.columns:
            benchmark_data = data_subset[benchmark]
        else:
            benchmark_data = data_subset['close']
            
        benchmark_returns = benchmark_data.pct_change()
        benchmark_cumulative = (benchmark_data / benchmark_data.iloc[0]) - 1
        
        results_df['benchmark_returns'] = benchmark_returns
        results_df['benchmark_cumulative'] = benchmark_cumulative
        
        # Création de l'objet Result
        return Result(
            strategy=strategy,
            results_df=results_df,
            trades=trades,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage
        )
    
    def _prepare_data_subset(self, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """
        Prépare un sous-ensemble des données pour le backtest.
        """
        data_subset = self.data.copy()
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            data_subset = data_subset[data_subset.index >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            data_subset = data_subset[data_subset.index <= end_date]
            
        if len(data_subset) == 0:
            raise ValueError("Aucune donnée disponible pour la période spécifiée")
            
        return data_subset
    
    def _get_rebalance_dates(self, data: pd.DataFrame, frequency: str) -> pd.DatetimeIndex:
        """
        Calcule les dates de rééquilibrage selon la fréquence.
        """
        if frequency == 'D':
            return data.index  # Tous les jours
        elif frequency == 'W':
            return pd.date_range(start=data.index[0], end=data.index[-1], freq='W')
        elif frequency == 'M':
            return pd.date_range(start=data.index[0], end=data.index[-1], freq='ME')
        elif frequency == 'Q':
            return pd.date_range(start=data.index[0], end=data.index[-1], freq='Q')
        elif frequency == 'Y':
            return pd.date_range(start=data.index[0], end=data.index[-1], freq='Y')
        else:
            return data.index  # Par défaut, quotidien
    
    def _execute_trade(self, old_position: float, new_position: float, price: float,
                      cash: float, current_shares: float, date: pd.Timestamp) -> Optional[Dict]:
        """
        Exécute un trade et calcule les coûts.
        """
        position_change = new_position - old_position
        
        if abs(position_change) < 0.001:
            return None
            
        # Calcul du nombre d'actions à acheter/vendre
        current_portfolio_value = cash + current_shares * price
        target_value = new_position * current_portfolio_value
        current_value = current_shares * price
        trade_value = target_value - current_value
        
        # Application du slippage
        effective_price = price * (1 + self.slippage * np.sign(trade_value))
        
        # Calcul des actions à trader
        shares_to_trade = trade_value / effective_price
        
        # Coûts de transaction
        transaction_cost_amount = abs(trade_value) * self.transaction_cost
        
        # Mise à jour du cash et des actions
        new_cash = cash - trade_value - transaction_cost_amount
        new_shares = current_shares + shares_to_trade
        
        return {
            'date': date,
            'price': price,
            'effective_price': effective_price,
            'shares_traded': shares_to_trade,
            'trade_value': trade_value,
            'transaction_cost': transaction_cost_amount,
            'cash_before': cash,
            'cash_after': new_cash,
            'shares_before': current_shares,
            'shares_after': new_shares,
            'position_before': old_position,
            'position_after': new_position
        }