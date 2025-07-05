#!/usr/bin/env python3

"""
Test Data Utilities
Ensures test data is available for all testing scenarios.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add backtester to path
backtester_path = str(Path(__file__).parent.parent.parent)
sys.path.append(backtester_path)

# Import from the correct path
import importlib.util
spec = importlib.util.spec_from_file_location("data_manager", Path(backtester_path) / "scripts" / "data_manager.py")
data_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_manager_module)
DataManager = data_manager_module.DataManager

logger = logging.getLogger(__name__)


class TestDataManager:
    """Manages test data download and validation for the testing suite."""
    
    def __init__(self):
        try:
            self.data_manager = DataManager()
        except ImportError:
            self.data_manager = None
            logger.warning("Data manager not available, using fallback data download")
        self.data_path = Path('./data')
        
    def ensure_test_data_available(self, symbols: list = None, days: int = 2) -> bool:
        """
        Ensure required test data is available, download if missing.
        
        Args:
            symbols: List of symbols needed for testing
            days: Number of days of data needed
            
        Returns:
            True if data is available or successfully downloaded
        """
        if symbols is None:
            symbols = ['EURUSD', 'EURJPY', 'GBPNZD']  # Default test symbols
        
        missing_symbols = []
        
        # Check each symbol for data availability
        for symbol in symbols:
            if not self._check_symbol_data_exists(symbol):
                missing_symbols.append(symbol)
        
        if missing_symbols:
            logger.info(f"🔄 Missing test data for symbols: {missing_symbols}")
            logger.info("📥 Downloading required test data...")
            
            if self.data_manager is None:
                logger.error("❌ Data manager not available and data is missing")
                logger.error("Please run: python scripts/data_manager.py quick --symbol eurusd --days 2")
                return False
            
            # Download missing data
            for symbol in missing_symbols:
                logger.info(f"📥 Downloading {symbol} data...")
                success = self.data_manager.quick_download(symbol.lower(), days)
                if not success:
                    logger.error(f"❌ Failed to download data for {symbol}")
                    return False
                else:
                    logger.info(f"✅ Successfully downloaded {symbol} data")
        
        logger.info("✅ All required test data is available")
        return True
    
    def _check_symbol_data_exists(self, symbol: str) -> bool:
        """
        Check if data exists for a symbol.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            True if data files exist
        """
        symbol_path = self.data_path / symbol
        
        if not symbol_path.exists():
            return False
        
        # Check for CSV files
        csv_files = list(symbol_path.glob('*.csv'))
        if csv_files:
            # Check if CSV files have content
            for csv_file in csv_files:
                if csv_file.stat().st_size > 1000:  # At least 1KB of data
                    return True
        
        # Check for .bi5 files
        bi5_files = list(symbol_path.rglob('*.bi5'))
        if bi5_files:
            return True
        
        return False
    
    def download_specific_test_data(self) -> bool:
        """
        Download specific data sets used by the test suite.
        
        Returns:
            True if all downloads successful
        """
        test_scenarios = [
            {
                'name': 'Simple Backtest Data',
                'symbols': ['EURUSD'],
                'days': 2
            },
            {
                'name': 'Execution Validation Data',
                'symbols': ['EURUSD'],
                'days': 2
            },
            {
                'name': 'Multi-Symbol Data',
                'symbols': ['EURUSD', 'EURJPY', 'GBPNZD'],
                'days': 2
            },
            {
                'name': 'Performance Analysis Data',
                'symbols': ['EURUSD'],
                'days': 2
            }
        ]
        
        logger.info("📥 Downloading test data for all scenarios...")
        
        all_symbols = set()
        for scenario in test_scenarios:
            all_symbols.update(scenario['symbols'])
        
        # Download all required symbols
        for symbol in all_symbols:
            logger.info(f"📥 Downloading {symbol}...")
            success = self.data_manager.quick_download(symbol.lower(), 2)
            if not success:
                logger.error(f"❌ Failed to download {symbol}")
                return False
        
        logger.info("✅ All test data downloaded successfully")
        return True


def main():
    """CLI interface for test data management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage test data for the backtesting framework')
    parser.add_argument('--download', action='store_true', help='Download all required test data')
    parser.add_argument('--check', action='store_true', help='Check data availability')
    parser.add_argument('--symbols', nargs='+', default=['EURUSD', 'EURJPY', 'GBPNZD'], 
                       help='Symbols to check/download')
    
    args = parser.parse_args()
    
    test_manager = TestDataManager()
    
    if args.download:
        success = test_manager.download_specific_test_data()
        sys.exit(0 if success else 1)
    elif args.check:
        available = test_manager.ensure_test_data_available(args.symbols, days=0)  # Just check
        print(f"✅ Data available: {available}")
        sys.exit(0 if available else 1)
    else:
        # Default: ensure data is available
        success = test_manager.ensure_test_data_available(args.symbols)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()