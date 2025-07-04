"""
Data Loader Factory for the backtesting framework.
Automatically detects and creates the appropriate data loader based on available data.
"""

from pathlib import Path
from typing import Union
import logging

from core.data_structures import BacktestConfig
from data.dukascopy_loader import DukascopyDataLoader
from data.csv_data_loader import CSVDataLoader


logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory class for creating appropriate data loaders.
    Automatically detects data format and creates the right loader.
    """
    
    @staticmethod
    def create_loader(data_path: str, config: BacktestConfig, 
                     force_format: str = None) -> Union[DukascopyDataLoader, CSVDataLoader]:
        """
        Create appropriate data loader based on available data.
        
        Args:
            data_path: Path to data directory
            config: Backtesting configuration
            force_format: Force specific format ('bi5' or 'csv')
            
        Returns:
            Appropriate data loader instance
        """
        data_path_obj = Path(data_path)
        
        if not data_path_obj.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # If format is forced, use it
        if force_format:
            if force_format.lower() == 'bi5':
                logger.info("Using Dukascopy .bi5 data loader (forced)")
                return DukascopyDataLoader(data_path, config)
            elif force_format.lower() == 'csv':
                logger.info("Using CSV data loader (forced)")
                return CSVDataLoader(data_path, config)
            else:
                raise ValueError(f"Unknown format: {force_format}. Use 'bi5' or 'csv'")
        
        # Auto-detect format based on available data
        has_bi5_data = DataLoaderFactory._detect_bi5_data(data_path_obj, config.symbols)
        has_csv_data = DataLoaderFactory._detect_csv_data(data_path_obj, config.symbols)
        
        if has_bi5_data and has_csv_data:
            logger.info("Both .bi5 and CSV data found, preferring .bi5 format")
            return DukascopyDataLoader(data_path, config)
        elif has_bi5_data:
            logger.info("Using Dukascopy .bi5 data loader (auto-detected)")
            return DukascopyDataLoader(data_path, config)
        elif has_csv_data:
            logger.info("Using CSV data loader (auto-detected)")
            return CSVDataLoader(data_path, config)
        else:
            # Default to CSV loader and let user know
            logger.warning(f"No data found for symbols {config.symbols}. Creating CSV loader - you may need to download data first.")
            return CSVDataLoader(data_path, config)
    
    @staticmethod
    def _detect_bi5_data(data_path: Path, symbols: list) -> bool:
        """
        Detect if .bi5 data is available.
        
        Args:
            data_path: Path to data directory
            symbols: List of symbols to check
            
        Returns:
            True if .bi5 data is found
        """
        for symbol in symbols:
            symbol_dir = data_path / symbol.upper()
            if symbol_dir.exists():
                # Look for year directories and .bi5 files
                year_dirs = [d for d in symbol_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                for year_dir in year_dirs:
                    # Look for .bi5 files in subdirectories
                    bi5_files = list(year_dir.rglob('*.bi5'))
                    if bi5_files:
                        return True
        return False
    
    @staticmethod
    def _detect_csv_data(data_path: Path, symbols: list) -> bool:
        """
        Detect if CSV data is available.
        
        Args:
            data_path: Path to data directory
            symbols: List of symbols to check
            
        Returns:
            True if CSV data is found
        """
        for symbol in symbols:
            symbol_dir = data_path / symbol.upper()
            if symbol_dir.exists():
                # Look for CSV files
                csv_files = list(symbol_dir.glob('*.csv'))
                if csv_files:
                    return True
        return False
    
    @staticmethod
    def get_data_format_info(data_path: str, symbols: list) -> dict:
        """
        Get information about available data formats.
        
        Args:
            data_path: Path to data directory
            symbols: List of symbols to check
            
        Returns:
            Dictionary with format information
        """
        data_path_obj = Path(data_path)
        
        has_bi5 = DataLoaderFactory._detect_bi5_data(data_path_obj, symbols)
        has_csv = DataLoaderFactory._detect_csv_data(data_path_obj, symbols)
        
        info = {
            'data_path': str(data_path_obj),
            'has_bi5_data': has_bi5,
            'has_csv_data': has_csv,
            'recommended_format': 'bi5' if has_bi5 else 'csv' if has_csv else None,
            'symbol_details': {}
        }
        
        # Get details for each symbol
        for symbol in symbols:
            symbol_dir = data_path_obj / symbol.upper()
            symbol_info = {
                'directory_exists': symbol_dir.exists(),
                'bi5_files': 0,
                'csv_files': 0
            }
            
            if symbol_dir.exists():
                # Count .bi5 files
                bi5_files = list(symbol_dir.rglob('*.bi5'))
                symbol_info['bi5_files'] = len(bi5_files)
                
                # Count CSV files
                csv_files = list(symbol_dir.glob('*.csv'))
                symbol_info['csv_files'] = len(csv_files)
            
            info['symbol_details'][symbol.upper()] = symbol_info
        
        return info