"""
Dukascopy data loader with tick interpolation and gap filling.
Handles loading historical tick data and ensures data quality.
"""

import os
import struct
import gzip
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Generator, Optional, Tuple
from pathlib import Path
import logging

from core.data_structures import Tick, BacktestConfig


logger = logging.getLogger(__name__)


class DukascopyDataLoader:
    """
    Professional Dukascopy data loader with tick interpolation.
    Handles .bi5 files and provides clean, gapless tick data.
    """
    
    def __init__(self, data_path: str, config: BacktestConfig):
        """
        Initialize the Dukascopy data loader.
        
        Args:
            data_path: Path to Dukascopy data directory
            config: Backtesting configuration
        """
        self.data_path = Path(data_path)
        self.config = config
        self.pip_values = self._get_pip_values()
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    def _get_pip_values(self) -> dict:
        """
        Get pip values for different currency pairs.
        Essential for accurate P&L calculations.
        """
        # Standard pip values (0.0001 for most pairs, 0.01 for JPY pairs)
        pip_values = {}
        
        for symbol in self.config.symbols:
            if 'JPY' in symbol:
                pip_values[symbol] = 0.01
            else:
                pip_values[symbol] = 0.0001
        
        return pip_values
    
    def _parse_bi5_file(self, file_path: Path) -> List[Tick]:
        """
        Parse a single .bi5 file into tick data.
        
        Args:
            file_path: Path to .bi5 file
            
        Returns:
            List of Tick objects
        """
        ticks = []
        
        try:
            with gzip.open(file_path, 'rb') as f:
                # Read the entire file content
                content = f.read()
                
                # Each tick is 20 bytes in .bi5 format
                tick_size = 20
                num_ticks = len(content) // tick_size
                
                for i in range(num_ticks):
                    # Extract 20 bytes for this tick
                    tick_data = content[i * tick_size:(i + 1) * tick_size]
                    
                    # Unpack the binary data
                    # Format: timestamp (4 bytes), ask (4 bytes), bid (4 bytes), 
                    #         ask_volume (4 bytes), bid_volume (4 bytes)
                    timestamp_ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack('>LLLLL', tick_data)
                    
                    # Convert timestamp (milliseconds since hour start)
                    hour_start = self._get_hour_start_from_path(file_path)
                    timestamp = hour_start + timedelta(milliseconds=timestamp_ms)
                    
                    # Convert prices (multiply by point value)
                    symbol = self._extract_symbol_from_path(file_path)
                    point_value = self.pip_values[symbol]
                    
                    ask = ask_raw * point_value
                    bid = bid_raw * point_value
                    
                    # Create tick object
                    tick = Tick(
                        timestamp=timestamp,
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        bid_volume=bid_vol,
                        ask_volume=ask_vol,
                        is_interpolated=False
                    )
                    
                    ticks.append(tick)
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
        
        return ticks
    
    def _get_hour_start_from_path(self, file_path: Path) -> datetime:
        """
        Extract hour start timestamp from file path.
        
        Args:
            file_path: Path to .bi5 file
            
        Returns:
            Datetime object representing the hour start
        """
        # Example path: /EURUSD/2023/01/15/12h_ticks.bi5
        parts = file_path.parts
        
        # Find year, month, day, hour from path
        year = int(parts[-4])
        month = int(parts[-3])
        day = int(parts[-2])
        hour = int(parts[-1].split('h')[0])
        
        return datetime(year, month, day, hour)
    
    def _extract_symbol_from_path(self, file_path: Path) -> str:
        """
        Extract symbol from file path.
        
        Args:
            file_path: Path to .bi5 file
            
        Returns:
            Symbol string (e.g., 'EURUSD')
        """
        # Example path: /EURUSD/2023/01/15/12h_ticks.bi5
        parts = file_path.parts
        
        # Find symbol in path (usually the first directory)
        for part in parts:
            if part in self.config.symbols:
                return part
        
        # If not found, try to guess from path structure
        return parts[-5] if len(parts) >= 5 else "UNKNOWN"
    
    def _interpolate_missing_ticks(self, ticks: List[Tick]) -> List[Tick]:
        """
        Interpolate missing ticks to fill gaps in data.
        
        Args:
            ticks: List of tick data (may have gaps)
            
        Returns:
            List of ticks with interpolated data
        """
        if not ticks or not self.config.interpolate_missing_ticks:
            return ticks
        
        interpolated_ticks = []
        
        for i in range(len(ticks) - 1):
            current_tick = ticks[i]
            next_tick = ticks[i + 1]
            
            # Add current tick
            interpolated_ticks.append(current_tick)
            
            # Calculate time gap
            time_gap = (next_tick.timestamp - current_tick.timestamp).total_seconds()
            
            # If gap is significant, interpolate
            if time_gap > self.config.max_gap_seconds:
                num_interpolations = int(time_gap // self.config.max_gap_seconds)
                
                for j in range(1, num_interpolations + 1):
                    # Linear interpolation
                    ratio = j / (num_interpolations + 1)
                    
                    interpolated_timestamp = current_tick.timestamp + timedelta(
                        seconds=ratio * time_gap
                    )
                    
                    interpolated_bid = current_tick.bid + ratio * (next_tick.bid - current_tick.bid)
                    interpolated_ask = current_tick.ask + ratio * (next_tick.ask - current_tick.ask)
                    
                    interpolated_tick = Tick(
                        timestamp=interpolated_timestamp,
                        symbol=current_tick.symbol,
                        bid=interpolated_bid,
                        ask=interpolated_ask,
                        bid_volume=current_tick.bid_volume,
                        ask_volume=current_tick.ask_volume,
                        is_interpolated=True
                    )
                    
                    interpolated_ticks.append(interpolated_tick)
        
        # Add the last tick
        if ticks:
            interpolated_ticks.append(ticks[-1])
        
        return interpolated_ticks
    
    def _validate_tick_data(self, ticks: List[Tick]) -> List[Tick]:
        """
        Validate and clean tick data.
        
        Args:
            ticks: Raw tick data
            
        Returns:
            Cleaned tick data
        """
        valid_ticks = []
        
        for tick in ticks:
            # Check for valid prices
            if tick.bid <= 0 or tick.ask <= 0:
                logger.warning(f"Invalid prices in tick: {tick}")
                continue
            
            # Check for inverted spread (bid > ask)
            if tick.bid >= tick.ask:
                logger.warning(f"Inverted spread in tick: {tick}")
                continue
            
            # Check for extreme spreads (> 100 pips)
            spread_pips = tick.spread / self.pip_values[tick.symbol]
            if spread_pips > 100:
                logger.warning(f"Extreme spread ({spread_pips:.1f} pips) in tick: {tick}")
                continue
            
            valid_ticks.append(tick)
        
        return valid_ticks
    
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
        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        
        current_date = start_date
        
        while current_date <= end_date:
            # Build file path for this date
            year = current_date.year
            month = current_date.month
            day = current_date.day
            
            date_path = self.data_path / symbol / str(year) / f"{month:02d}" / f"{day:02d}"
            
            if not date_path.exists():
                logger.warning(f"No data found for {symbol} on {current_date.date()}")
                current_date += timedelta(days=1)
                continue
            
            # Load all hourly files for this date
            daily_ticks = []
            
            for hour in range(24):
                hour_file = date_path / f"{hour:02d}h_ticks.bi5"
                
                if hour_file.exists():
                    hour_ticks = self._parse_bi5_file(hour_file)
                    daily_ticks.extend(hour_ticks)
            
            # Sort ticks by timestamp
            daily_ticks.sort(key=lambda x: x.timestamp)
            
            # Validate and clean data
            daily_ticks = self._validate_tick_data(daily_ticks)
            
            # Interpolate missing ticks
            daily_ticks = self._interpolate_missing_ticks(daily_ticks)
            
            # Yield ticks in chronological order
            for tick in daily_ticks:
                if start_date <= tick.timestamp <= end_date:
                    yield tick
            
            current_date += timedelta(days=1)
    
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
        # Load all ticks from all symbols
        all_ticks = []
        
        for symbol in symbols:
            symbol_ticks = list(self.load_symbol_data(symbol, start_date, end_date))
            all_ticks.extend(symbol_ticks)
        
        # Sort all ticks by timestamp
        all_ticks.sort(key=lambda x: x.timestamp)
        
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
        tick_count = 0
        interpolated_count = 0
        min_spread = float('inf')
        max_spread = 0.0
        total_spread = 0.0
        
        first_tick = None
        last_tick = None
        
        for tick in self.load_symbol_data(symbol, start_date, end_date):
            if first_tick is None:
                first_tick = tick
            last_tick = tick
            
            tick_count += 1
            if tick.is_interpolated:
                interpolated_count += 1
            
            spread_pips = tick.spread / self.pip_values[symbol]
            min_spread = min(min_spread, spread_pips)
            max_spread = max(max_spread, spread_pips)
            total_spread += spread_pips
        
        avg_spread = total_spread / tick_count if tick_count > 0 else 0
        
        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'tick_count': tick_count,
            'interpolated_count': interpolated_count,
            'interpolation_rate': interpolated_count / tick_count if tick_count > 0 else 0,
            'min_spread_pips': min_spread if min_spread != float('inf') else 0,
            'max_spread_pips': max_spread,
            'avg_spread_pips': avg_spread,
            'first_tick_time': first_tick.timestamp if first_tick else None,
            'last_tick_time': last_tick.timestamp if last_tick else None,
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
        if summary['tick_count'] < 1000:
            logger.warning(f"Insufficient tick data for {symbol}: {summary['tick_count']} ticks")
            return False
        
        if summary['interpolation_rate'] > 0.1:
            logger.warning(f"High interpolation rate for {symbol}: {summary['interpolation_rate']:.2%}")
            return False
        
        if summary['avg_spread_pips'] > 10:
            logger.warning(f"Unusually high average spread for {symbol}: {summary['avg_spread_pips']:.1f} pips")
            return False
        
        return True