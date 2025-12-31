"""Financial data loading and validation utilities.

This module contains the `DataHandler` class, which handles loading market data
from various sources (CSV, Parquet, DataFrame) and performs validation and
normalization. It implements the Single Responsibility Principle by separating
data I/O concerns from backtesting logic.

The DataHandler ensures that loaded data has the required columns (open, high,
low, close, volume), a properly formatted datetime index, and consistent column
naming (lowercase). Missing optional columns are added with appropriate default
values.

Notes
-----
Column names are automatically normalized to lowercase to ensure consistency
across different data sources. The module supports both file-based inputs
(CSV, Parquet) and in-memory DataFrames.

If volume data is missing, it is filled with zeros. The datetime index is
automatically inferred and validated during the loading process.

Authors
-------
Mariano Benjamin
Noah Chikhi
Antoine Moniz
"""

import pandas as pd
import numpy as np
from typing import Union, List
from pathlib import Path
import warnings


class DataHandler:
    """
    Handles loading and validation of financial market data.
    
    Supports loading from CSV files, Parquet files, and Pandas DataFrames.
    Ensures data contains required columns and proper formatting.
    """
    
    REQUIRED_COLUMNS = ['close']
    OPTIONAL_COLUMNS = ['open', 'high', 'low', 'volume']
    
    def __init__(self):
        """Initialize the DataHandler."""
        pass
    
    def load(self, data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """
        Load financial data from various sources.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, str, Path]
            Data source. Can be:
            - pd.DataFrame: Direct dataframe input
            - str or Path: Path to CSV or Parquet file
        
        Returns
        -------
        pd.DataFrame
            Loaded and validated dataframe with datetime index and required columns.
        
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file format is unsupported or data is invalid.
        """
        # Load data based on input type
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = self._load_from_file(data)
        
        # Validate and process the data
        df = self._validate_and_process(df)
        
        return df
    
    def _load_from_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from a file (CSV or Parquet).
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the data file.
        
        Returns
        -------
        pd.DataFrame
            Loaded dataframe.
        
        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load accordingly
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: .csv, .parquet, .pq"
            )
        
        return df
    
    def _validate_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and process the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe to validate.
        
        Returns
        -------
        pd.DataFrame
            Validated and processed dataframe.
        
        Raises
        ------
        ValueError
            If required columns are missing.
        """
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Handle datetime index
        df = self._setup_datetime_index(df)
        
        # Check for required columns
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add optional columns if missing
        df = self._add_missing_columns(df)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
    
    def _setup_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set up a proper datetime index for the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to process.
        
        Returns
        -------
        pd.DataFrame
            Dataframe with datetime index.
        """
        # Check if 'date' column exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # If no date column and index is not datetime, create artificial date range
            warnings.warn(
                "No 'date' column found and index is not datetime. "
                "Creating an artificial datetime index starting from 2020-01-01."
            )
            date_range = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            df.index = date_range
        
        return df
    
    def _add_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add missing optional columns with default values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to process.
        
        Returns
        -------
        pd.DataFrame
            Dataframe with all required and optional columns.
        """
        # Add 'open' if missing (use previous close or current close)
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        
        # Add 'high' if missing (use close as proxy)
        if 'high' not in df.columns:
            df['high'] = df['close']
        
        # Add 'low' if missing (use close as proxy)
        if 'low' not in df.columns:
            df['low'] = df['close']
        
        # Add 'volume' if missing (use artificial volume)
        if 'volume' not in df.columns:
            df['volume'] = 1_000_000  # Default volume
        
        return df
    
    def validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that a dataframe contains specific columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to validate.
        required_columns : List[str]
            List of required column names.
        
        Returns
        -------
        bool
            True if all required columns exist, False otherwise.
        """
        return all(col in df.columns for col in required_columns)
