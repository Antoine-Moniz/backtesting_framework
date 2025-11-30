"""
Module contenant la classe abstraite Strategy et le décorateur pour les stratégies simples.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Callable
import pandas as pd
import numpy as np


class Strategy(ABC):
    """
    Classe abstraite pour définir une stratégie d'investissement.
    
    Cette classe doit être héritée pour créer des stratégies personnalisées.
    Elle définit l'interface requise pour toutes les stratégies.
    """
    
    def __init__(self, name: str = None, rebalance_frequency: str = "D"):
        """
        Initialise la stratégie.
        
        Args:
            name (str): Nom de la stratégie
            rebalance_frequency (str): Fréquence de rééquilibrage ('D', 'W', 'M', 'Q', 'Y')
        """
        self.name = name if name else self.__class__.__name__
        self.rebalance_frequency = rebalance_frequency
        self.is_fitted = False
        
    @abstractmethod
    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Méthode abstraite obligatoire pour déterminer la position à prendre.
        
        Args:
            historical_data (pd.DataFrame): Données historiques disponibles jusqu'au moment actuel
            current_position (float): Position actuelle (entre -1 et 1, où 1 = 100% long, -1 = 100% short, 0 = neutre)
            
        Returns:
            float: Nouvelle position à prendre (entre -1 et 1)
        """
        pass
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Méthode optionnelle pour entraîner/calibrer la stratégie sur des données historiques.
        
        Par défaut, cette méthode ne fait rien. Elle peut être surchargée par les stratégies
        qui ont besoin d'un entraînement préalable.
        
        Args:
            data (pd.DataFrame): Données d'entraînement
        """
        self.is_fitted = True
    
    def __str__(self) -> str:
        return f"Strategy: {self.name} (rebalance: {self.rebalance_frequency})"
    
    def __repr__(self) -> str:
        return self.__str__()


def strategy_decorator(name: str = None, rebalance_frequency: str = "D"):
    """
    Décorateur pour créer des stratégies simples sans avoir besoin d'hériter de Strategy.
    
    Ce décorateur permet de transformer une fonction simple en stratégie complète.
    La fonction décorée doit accepter (historical_data, current_position) et retourner une position.
    
    Args:
        name (str): Nom de la stratégie
        rebalance_frequency (str): Fréquence de rééquilibrage
        
    Returns:
        Strategy: Instance de stratégie créée à partir de la fonction
        
    Example:
        @strategy_decorator(name="Simple MA Cross", rebalance_frequency="D")
        def ma_cross_strategy(historical_data, current_position):
            if len(historical_data) < 20:
                return 0
            short_ma = historical_data['close'].rolling(5).mean().iloc[-1]
            long_ma = historical_data['close'].rolling(20).mean().iloc[-1]
            return 1 if short_ma > long_ma else -1
    """
    def decorator(func: Callable):
        class DecoratedStrategy(Strategy):
            def __init__(self):
                strategy_name = name if name else func.__name__
                super().__init__(strategy_name, rebalance_frequency)
                self._strategy_func = func
                
            def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
                return self._strategy_func(historical_data, current_position)
                
        return DecoratedStrategy()
    
    return decorator


# Stratégies d'exemple pour les tests et démonstrations

class BuyAndHoldStrategy(Strategy):
    """
    Stratégie simple Buy and Hold - achète au début et ne vend jamais.
    """
    
    def __init__(self):
        super().__init__("Buy and Hold", "D")
        
    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """Toujours long à 100%"""
        return 1.0


class MovingAverageCrossStrategy(Strategy):
    """
    Stratégie de croisement de moyennes mobiles.
    Achète quand la MA courte passe au-dessus de la MA longue, vend dans le cas contraire.
    """
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(f"MA Cross ({short_window}/{long_window})", "D")
        self.short_window = short_window
        self.long_window = long_window
        
    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Détermine la position basée sur le croisement des moyennes mobiles.
        """
        if len(historical_data) < self.long_window:
            return 0  # Pas assez de données
            
        # Calcul des moyennes mobiles
        short_ma = historical_data['close'].rolling(self.short_window).mean().iloc[-1]
        long_ma = historical_data['close'].rolling(self.long_window).mean().iloc[-1]
        
        # Signal de position
        if short_ma > long_ma:
            return 1.0  # Long
        else:
            return -1.0  # Short


class MeanReversionStrategy(Strategy):
    """
    Stratégie de retour à la moyenne utilisant les bandes de Bollinger.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(f"Mean Reversion (BB {window}, {num_std}σ)", "D")
        self.window = window
        self.num_std = num_std
        
    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Position basée sur les bandes de Bollinger pour le retour à la moyenne.
        """
        if len(historical_data) < self.window:
            return 0
            
        closes = historical_data['close']
        ma = closes.rolling(self.window).mean().iloc[-1]
        std = closes.rolling(self.window).std().iloc[-1]
        current_price = closes.iloc[-1]
        
        upper_band = ma + (self.num_std * std)
        lower_band = ma - (self.num_std * std)
        
        if current_price > upper_band:
            return -1.0  # Prix trop haut, vendre
        elif current_price < lower_band:
            return 1.0   # Prix trop bas, acheter
        else:
            return 0.0   # Neutre