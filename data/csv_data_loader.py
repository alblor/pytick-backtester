"""
CSV Data Loader for the backtesting framework.
Loads tick data from CSV files downloaded via dukascopy-node.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Generator, Optional, Dict, Tuple
from pathlib import Path
import logging
import glob
import json

from core.data_structures import Tick, BacktestConfig


logger = logging.getLogger(__name__)


class CSVDataLoader:
    """
    Professional CSV data loader for Dukascopy data.
    Handles CSV files from dukascopy-node with tick interpolation and gap filling.
    """
    
    def __init__(self, data_path: str, config: BacktestConfig):
        """
        Initialize the CSV data loader.
        
        Args:
            data_path: Path to CSV data directory
            config: Backtesting configuration
        """
        self.data_path = Path(data_path)
        self.config = config
        self.pip_values = self._get_pip_values()
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Cache for loaded data
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_max_size = 100  # Maximum cached files
        
        logger.info(f"CSV data loader initialized for path: {data_path}")
    
    def _get_pip_values(self) -> dict:
        """
        Get pip values for different currency pairs.
        Essential for accurate P&L calculations.
        """
        pip_values = {}
        
        for symbol in self.config.symbols:
            if 'JPY' in symbol:
                pip_values[symbol] = 0.01
            else:
                pip_values[symbol] = 0.0001
        
        return pip_values
    
    def _find_csv_files(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Path]:
        """
        Find CSV files for a symbol within the date range.
        
        Args:
            symbol: Currency pair symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of CSV file paths
        """
        symbol_dir = self.data_path / symbol.upper()
        if not symbol_dir.exists():
            logger.warning(f"Symbol directory not found: {symbol_dir}")
            return []
        
        # Look for CSV files matching the symbol
        pattern = f"{symbol.upper()}_*.csv"
        csv_files = list(symbol_dir.glob(pattern))
        
        if not csv_files:
            logger.warning(f"No CSV files found for {symbol} in {symbol_dir}")
            return []
        
        # Filter files by date range based on filename
        filtered_files = []
        for file_path in csv_files:
            if self._file_in_date_range(file_path, start_date, end_date):
                filtered_files.append(file_path)
        
        # Sort by filename (which should include dates)
        filtered_files.sort()
        
        logger.info(f"Found {len(filtered_files)} CSV files for {symbol}")
        return filtered_files
    
    def _file_in_date_range(self, file_path: Path, start_date: datetime, end_date: datetime) -> bool:
        """
        Check if a CSV file contains data within the specified date range.
        
        Args:
            file_path: Path to CSV file
            start_date: Start date
            end_date: End date
            
        Returns:
            True if file is in range
        """
        filename = file_path.stem
        
        # Try to extract dates from filename
        # Expected format: SYMBOL_YYYY-MM-DD_YYYY-MM-DD_timeframe.csv
        parts = filename.split('_')
        if len(parts) >= 3:
            try:
                file_start = datetime.strptime(parts[1], '%Y-%m-%d')
                file_end = datetime.strptime(parts[2], '%Y-%m-%d')
                
                # Check if there's any overlap
                return not (file_end < start_date or file_start > end_date)
            except ValueError:
                pass
        
        # If we can't parse the filename, include the file
        return True
    
    def _load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single CSV file into a DataFrame.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with tick data
        """
        cache_key = str(file_path)
        
        # Check cache first
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        try:
            # Load CSV with proper data types
            df = pd.read_csv(
                file_path,
                parse_dates=['timestamp'],
                dtype={
                    'symbol': 'string',
                    'bid': 'float64',
                    'ask': 'float64',
                    'bid_volume': 'float64',
                    'ask_volume': 'float64'
                }
            )
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'symbol', 'bid', 'ask']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Add default volume columns if missing
            if 'bid_volume' not in df.columns:
                df['bid_volume'] = 0.0
            if 'ask_volume' not in df.columns:
                df['ask_volume'] = 0.0
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Cache the data (with size limit)
            if len(self._data_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._data_cache))
                del self._data_cache[oldest_key]
            
            self._data_cache[cache_key] = df
            
            logger.debug(f"Loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return pd.DataFrame()
    
    def _validate_tick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean tick data.
        
        Args:
            df: Raw tick data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        initial_count = len(df)
        
        # Remove rows with invalid prices
        df = df[(df['bid'] > 0) & (df['ask'] > 0)]
        
        # Remove rows with inverted spreads
        df = df[df['bid'] < df['ask']]
        
        # Remove extreme spreads (> 100 pips)
        symbol = df['symbol'].iloc[0] if not df.empty else 'UNKNOWN'
        pip_value = self.pip_values.get(symbol, 0.0001)
        
        df['spread_pips'] = (df['ask'] - df['bid']) / pip_value
        df = df[df['spread_pips'] <= 100]
        df = df.drop('spread_pips', axis=1)
        
        # Remove duplicate timestamps
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        cleaned_count = len(df)
        if cleaned_count < initial_count:
            logger.info(f"Cleaned {initial_count - cleaned_count} invalid ticks, {cleaned_count} remaining")
        
        return df
    
    def _interpolate_missing_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing ticks to fill gaps in data.
        
        Args:
            df: DataFrame with tick data (may have gaps)
            
        Returns:
            DataFrame with interpolated data
        """
        if df.empty or not self.config.interpolate_missing_ticks:
            return df
        
        interpolated_rows = []
        
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            # Add current row
            interpolated_rows.append(current_row)
            
            # Calculate time gap
            time_gap = (next_row['timestamp'] - current_row['timestamp']).total_seconds()
            
            # If gap is significant, interpolate
            if time_gap > self.config.max_gap_seconds:
                num_interpolations = int(time_gap // self.config.max_gap_seconds)
                
                for j in range(1, num_interpolations + 1):
                    # Linear interpolation
                    ratio = j / (num_interpolations + 1)
                    
                    interpolated_timestamp = current_row['timestamp'] + timedelta(
                        seconds=ratio * time_gap
                    )
                    
                    interpolated_bid = current_row['bid'] + ratio * (next_row['bid'] - current_row['bid'])
                    interpolated_ask = current_row['ask'] + ratio * (next_row['ask'] - current_row['ask'])
                    
                    interpolated_row = pd.Series({
                        'timestamp': interpolated_timestamp,
                        'symbol': current_row['symbol'],
                        'bid': interpolated_bid,
                        'ask': interpolated_ask,
                        'bid_volume': current_row['bid_volume'],
                        'ask_volume': current_row['ask_volume'],
                        'is_interpolated': True
                    })
                    
                    interpolated_rows.append(interpolated_row)
        
        # Add the last row
        if not df.empty:
            interpolated_rows.append(df.iloc[-1])
        
        # Create new DataFrame
        if interpolated_rows:
            result_df = pd.DataFrame(interpolated_rows).reset_index(drop=True)
            
            # Add is_interpolated column if not present
            if 'is_interpolated' not in result_df.columns:
                result_df['is_interpolated'] = False
            
            original_count = len(df)
            interpolated_count = len(result_df) - original_count
            
            if interpolated_count > 0:
                logger.info(f"Interpolated {interpolated_count} ticks for better continuity")
            
            return result_df
        
        return df
    
    def _dataframe_to_ticks(self, df: pd.DataFrame) -> Generator[Tick, None, None]:
        """
        Convert DataFrame rows to Tick objects.
        
        Args:
            df: DataFrame with tick data
            
        Yields:
            Tick objects
        """
        for _, row in df.iterrows():
            tick = Tick(
                timestamp=row['timestamp'].to_pydatetime(),
                symbol=row['symbol'],
                bid=float(row['bid']),
                ask=float(row['ask']),
                bid_volume=float(row['bid_volume']),
                ask_volume=float(row['ask_volume']),
                is_interpolated=bool(row.get('is_interpolated', False))
            )
            yield tick
    
    def load_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Generator[Tick, None, None]:
        """
        Load tick data for a specific symbol and date range.
        
        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Yields:
            Tick objects in chronological order
        """
        logger.info(f"Loading CSV data for {symbol} from {start_date} to {end_date}")
        
        # Find relevant CSV files
        csv_files = self._find_csv_files(symbol, start_date, end_date)
        
        if not csv_files:
            logger.warning(f"No CSV files found for {symbol}")
            return
        
        # Load and combine all files
        all_dataframes = []
        
        for csv_file in csv_files:
            df = self._load_csv_file(csv_file)
            if not df.empty:
                all_dataframes.append(df)
        
        if not all_dataframes:
            logger.warning(f"No valid data loaded for {symbol}")
            return
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Filter by exact date range
        mask = (combined_df['timestamp'] >= start_date) & (combined_df['timestamp'] <= end_date)
        filtered_df = combined_df[mask]
        
        if filtered_df.empty:
            logger.warning(f"No data in specified date range for {symbol}")
            return
        
        # Sort by timestamp
        filtered_df = filtered_df.sort_values('timestamp').reset_index(drop=True)
        
        # Validate and clean data
        filtered_df = self._validate_tick_data(filtered_df)
        
        # Interpolate missing ticks
        filtered_df = self._interpolate_missing_ticks(filtered_df)
        
        logger.info(f"Loaded {len(filtered_df)} ticks for {symbol}")
        
        # Convert to Tick objects and yield
        for tick in self._dataframe_to_ticks(filtered_df):
            yield tick
    
    def load_multiple_symbols(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Generator[Tick, None, None]:
        """
        Load tick data for multiple symbols and merge chronologically.
        
        Args:
            symbols: List of currency pair symbols
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Yields:
            Tick objects in chronological order across all symbols
        """
        logger.info(f"Loading multiple symbols: {symbols}")
        
        # Load all ticks from all symbols
        all_ticks = []
        
        for symbol in symbols:
            symbol_ticks = list(self.load_symbol_data(symbol, start_date, end_date))
            all_ticks.extend(symbol_ticks)
            logger.info(f"Loaded {len(symbol_ticks)} ticks for {symbol}")
        
        # Sort all ticks by timestamp
        all_ticks.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Total ticks loaded: {len(all_ticks)}")
        
        # Yield ticks in chronological order
        for tick in all_ticks:
            yield tick
    
    def get_data_summary(self, symbol: str, start_date: datetime, end_date: datetime) -> dict:
        """
        Get summary statistics for available data.
        
        Args:
            symbol: Currency pair symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with data summary statistics
        """
        ticks = list(self.load_symbol_data(symbol, start_date, end_date))
        
        if not ticks:
            return {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'tick_count': 0,
                'interpolated_count': 0,
                'interpolation_rate': 0,
                'data_completeness': 0
            }
        
        tick_count = len(ticks)
        interpolated_count = sum(1 for tick in ticks if tick.is_interpolated)
        
        spreads = [(tick.ask - tick.bid) / self.pip_values[symbol] for tick in ticks]
        
        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'tick_count': tick_count,
            'interpolated_count': interpolated_count,
            'interpolation_rate': interpolated_count / tick_count if tick_count > 0 else 0,
            'min_spread_pips': min(spreads) if spreads else 0,
            'max_spread_pips': max(spreads) if spreads else 0,
            'avg_spread_pips': sum(spreads) / len(spreads) if spreads else 0,
            'first_tick_time': ticks[0].timestamp if ticks else None,
            'last_tick_time': ticks[-1].timestamp if ticks else None,
            'data_completeness': 1.0 - (interpolated_count / tick_count) if tick_count > 0 else 0
        }
    
    def verify_data_integrity(self, symbol: str, start_date: datetime, end_date: datetime) -> bool:
        """
        Verify data integrity for a symbol and date range.
        
        Args:
            symbol: Currency pair symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            True if data integrity is acceptable
        """
        summary = self.get_data_summary(symbol, start_date, end_date)
        
        # Check minimum requirements
        if summary['tick_count'] < 100:  # Reduced threshold for CSV data
            logger.warning(f"Insufficient tick data for {symbol}: {summary['tick_count']} ticks")
            return False
        
        if summary['interpolation_rate'] > 0.5:  # Allow higher interpolation for CSV
            logger.warning(f"High interpolation rate for {symbol}: {summary['interpolation_rate']:.2%}")
            return False
        
        if summary['avg_spread_pips'] > 20:  # Allow wider spreads for CSV data
            logger.warning(f"Unusually high average spread for {symbol}: {summary['avg_spread_pips']:.1f} pips")
            return False
        
        return True
    
    def list_available_data(self) -> Dict[str, List[str]]:
        """
        List all available data files.
        
        Returns:
            Dictionary mapping symbols to list of available files
        """
        available_data = {}
        
        for symbol in self.config.symbols:
            symbol_dir = self.data_path / symbol.upper()
            if symbol_dir.exists():
                csv_files = list(symbol_dir.glob('*.csv'))
                available_data[symbol.upper()] = [f.name for f in csv_files]
            else:
                available_data[symbol.upper()] = []
        
        return available_data
    
    def get_download_summary(self) -> Optional[dict]:
        """
        Get download summary if available.
        
        Returns:
            Download summary data or None
        """
        summary_path = self.data_path / 'download_summary.json'
        
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load download summary: {e}")
        
        return None