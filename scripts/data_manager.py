#!/usr/bin/env python3

"""
Data Management Utilities for the Backtesting Framework.
Provides tools for downloading, validating, and managing tick data.
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import json

# Add backtester to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_structures import BacktestConfig
from data.data_loader_factory import DataLoaderFactory
from analysis.performance_analyzer import PerformanceAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataManager:
    """Comprehensive data management for the backtesting framework."""
    
    def __init__(self, data_path: str = './data'):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
    
    def download_data(self, symbols: list, start_date: str, end_date: str, 
                     timeframe: str = 'tick', batch_size: int = 7) -> bool:
        """
        Download data using dukascopy-node.
        
        Args:
            symbols: List of currency pairs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            batch_size: Batch size in days
            
        Returns:
            True if download successful
        """
        logger.info(f"Downloading data for {symbols} from {start_date} to {end_date}")
        
        try:
            # Build command
            cmd = [
                'node', 'scripts/fetch_data.js',
                '--symbols', ','.join(symbols),
                '--from', start_date,
                '--to', end_date,
                '--output', str(self.data_path),
                '--timeframe', timeframe,
                '--batch-size', str(batch_size),
                '--verbose'
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            # Run download
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            if result.returncode == 0:
                logger.info("Download completed successfully")
                print(result.stdout)
                return True
            else:
                logger.error(f"Download failed: {result.stderr}")
                print(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"Error running download: {e}")
            return False
    
    def quick_download(self, symbol: str = 'eurusd', days: int = 2) -> bool:
        """
        Quick download for testing (single pair, short period).
        
        Args:
            symbol: Currency pair
            days: Number of days to download
            
        Returns:
            True if successful
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Avoid weekends
        if start_date.weekday() >= 5:  # Saturday or Sunday
            start_date = start_date - timedelta(days=start_date.weekday()-4)
        
        return self.download_data(
            symbols=[symbol],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            timeframe='tick'
        )
    
    def validate_data(self, symbols: list, start_date: str, end_date: str) -> dict:
        """
        Validate downloaded data quality.
        
        Args:
            symbols: List of symbols to validate
            start_date: Start date
            end_date: End date
            
        Returns:
            Validation results
        """
        logger.info("Validating data quality...")
        
        config = BacktestConfig(
            start_date=datetime.strptime(start_date, '%Y-%m-%d'),
            end_date=datetime.strptime(end_date, '%Y-%m-%d'),
            symbols=symbols,
            initial_balance=100000
        )
        
        try:
            data_loader = DataLoaderFactory.create_loader(str(self.data_path), config)
            
            validation_results = {
                'overall_status': 'PASS',
                'symbols': {},
                'data_format': type(data_loader).__name__,
                'total_issues': 0
            }
            
            for symbol in symbols:
                logger.info(f"Validating {symbol}...")
                
                # Get data summary
                summary = data_loader.get_data_summary(
                    symbol, config.start_date, config.end_date
                )
                
                # Check data integrity
                integrity_ok = data_loader.verify_data_integrity(
                    symbol, config.start_date, config.end_date
                )
                
                issues = []
                
                if summary['tick_count'] == 0:
                    issues.append("NO_DATA")
                elif summary['tick_count'] < 100:
                    issues.append("INSUFFICIENT_DATA")
                
                if summary['interpolation_rate'] > 0.5:
                    issues.append("HIGH_INTERPOLATION")
                
                if summary['avg_spread_pips'] > 20:
                    issues.append("WIDE_SPREADS")
                
                if not integrity_ok:
                    issues.append("INTEGRITY_FAILED")
                
                symbol_status = 'FAIL' if issues else 'PASS'
                
                validation_results['symbols'][symbol] = {
                    'status': symbol_status,
                    'tick_count': summary['tick_count'],
                    'interpolation_rate': summary['interpolation_rate'],
                    'avg_spread_pips': summary['avg_spread_pips'],
                    'data_completeness': summary['data_completeness'],
                    'issues': issues
                }
                
                validation_results['total_issues'] += len(issues)
                
                if symbol_status == 'FAIL':
                    validation_results['overall_status'] = 'FAIL'
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'overall_status': 'ERROR',
                'error': str(e),
                'symbols': {},
                'total_issues': 1
            }
    
    def list_available_data(self) -> dict:
        """
        List all available data.
        
        Returns:
            Dictionary with available data information
        """
        logger.info("Scanning available data...")
        
        available_data = {
            'data_path': str(self.data_path),
            'symbols': {},
            'summary': {
                'total_symbols': 0,
                'total_files': 0,
                'data_formats': []
            }
        }
        
        # Scan for symbol directories
        if self.data_path.exists():
            for item in self.data_path.iterdir():
                if item.is_dir() and item.name.isupper():
                    symbol = item.name
                    
                    # Count files
                    csv_files = list(item.glob('*.csv'))
                    bi5_files = list(item.rglob('*.bi5'))
                    
                    available_data['symbols'][symbol] = {
                        'csv_files': len(csv_files),
                        'bi5_files': len(bi5_files),
                        'csv_file_list': [f.name for f in csv_files],
                        'has_data': len(csv_files) > 0 or len(bi5_files) > 0
                    }
                    
                    available_data['summary']['total_files'] += len(csv_files) + len(bi5_files)
                    
                    if len(csv_files) > 0:
                        available_data['summary']['data_formats'].append('CSV')
                    if len(bi5_files) > 0:
                        available_data['summary']['data_formats'].append('BI5')
            
            available_data['summary']['total_symbols'] = len(available_data['symbols'])
            available_data['summary']['data_formats'] = list(set(available_data['summary']['data_formats']))
        
        # Check for download summary
        summary_file = self.data_path / 'download_summary.json'
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    download_summary = json.load(f)
                available_data['last_download'] = download_summary
            except Exception as e:
                logger.warning(f"Could not read download summary: {e}")
        
        return available_data
    
    def cleanup_data(self, older_than_days: int = 30) -> dict:
        """
        Clean up old data files.
        
        Args:
            older_than_days: Remove files older than this many days
            
        Returns:
            Cleanup results
        """
        logger.info(f"Cleaning up data older than {older_than_days} days...")
        
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        removed_files = []
        total_size_removed = 0
        
        if self.data_path.exists():
            for file_path in self.data_path.rglob('*'):
                if file_path.is_file():
                    file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_modified < cutoff_date:
                        file_size = file_path.stat().st_size
                        removed_files.append({
                            'file': str(file_path.relative_to(self.data_path)),
                            'size': file_size,
                            'modified': file_modified.isoformat()
                        })
                        total_size_removed += file_size
                        file_path.unlink()
        
        return {
            'removed_files': len(removed_files),
            'total_size_removed': total_size_removed,
            'cutoff_date': cutoff_date.isoformat(),
            'details': removed_files
        }
    
    def export_data_info(self, output_file: str) -> None:
        """
        Export data information to file.
        
        Args:
            output_file: Output file path
        """
        data_info = {
            'scan_date': datetime.now().isoformat(),
            'available_data': self.list_available_data()
        }
        
        output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(data_info, f, indent=2, default=str)
        else:
            # Plain text format
            with open(output_path, 'w') as f:
                f.write("BACKTESTING FRAMEWORK - DATA INVENTORY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Scan Date: {data_info['scan_date']}\n")
                f.write(f"Data Path: {data_info['available_data']['data_path']}\n\n")
                
                f.write("SUMMARY:\n")
                summary = data_info['available_data']['summary']
                f.write(f"  Total Symbols: {summary['total_symbols']}\n")
                f.write(f"  Total Files: {summary['total_files']}\n")
                f.write(f"  Data Formats: {', '.join(summary['data_formats'])}\n\n")
                
                f.write("SYMBOLS:\n")
                for symbol, info in data_info['available_data']['symbols'].items():
                    f.write(f"  {symbol}:\n")
                    f.write(f"    CSV Files: {info['csv_files']}\n")
                    f.write(f"    BI5 Files: {info['bi5_files']}\n")
                    f.write(f"    Has Data: {info['has_data']}\n\n")
        
        logger.info(f"Data information exported to {output_path}")


def main():
    """Command line interface for data management."""
    parser = argparse.ArgumentParser(description='Backtesting Framework Data Manager')
    parser.add_argument('--data-path', default='./data', help='Data directory path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download data')
    download_parser.add_argument('--symbols', default='eurusd', help='Comma-separated symbols')
    download_parser.add_argument('--from', dest='start_date', required=True, help='Start date (YYYY-MM-DD)')
    download_parser.add_argument('--to', dest='end_date', required=True, help='End date (YYYY-MM-DD)')
    download_parser.add_argument('--timeframe', default='tick', help='Timeframe (tick, m1, etc.)')
    download_parser.add_argument('--batch-size', type=int, default=7, help='Batch size in days')
    
    # Quick download command
    quick_parser = subparsers.add_parser('quick', help='Quick download for testing')
    quick_parser.add_argument('--symbol', default='eurusd', help='Symbol to download')
    quick_parser.add_argument('--days', type=int, default=2, help='Number of days')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data quality')
    validate_parser.add_argument('--symbols', default='eurusd', help='Comma-separated symbols')
    validate_parser.add_argument('--from', dest='start_date', required=True, help='Start date')
    validate_parser.add_argument('--to', dest='end_date', required=True, help='End date')
    
    # List command
    subparsers.add_parser('list', help='List available data')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Remove files older than N days')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data information')
    export_parser.add_argument('output_file', help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create data manager
    manager = DataManager(args.data_path)
    
    try:
        if args.command == 'download':
            symbols = [s.strip().lower() for s in args.symbols.split(',')]
            success = manager.download_data(
                symbols, args.start_date, args.end_date, 
                args.timeframe, args.batch_size
            )
            if not success:
                sys.exit(1)
        
        elif args.command == 'quick':
            success = manager.quick_download(args.symbol.lower(), args.days)
            if not success:
                sys.exit(1)
        
        elif args.command == 'validate':
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            results = manager.validate_data(symbols, args.start_date, args.end_date)
            
            print("\nDATA VALIDATION RESULTS")
            print("="*50)
            print(f"Overall Status: {results['overall_status']}")
            print(f"Data Format: {results.get('data_format', 'Unknown')}")
            print(f"Total Issues: {results['total_issues']}")
            
            if 'error' in results:
                print(f"Error: {results['error']}")
            else:
                print("\nPer-Symbol Results:")
                for symbol, info in results['symbols'].items():
                    print(f"\n{symbol}: {info['status']}")
                    print(f"  Ticks: {info['tick_count']:,}")
                    print(f"  Interpolation: {info['interpolation_rate']:.1%}")
                    print(f"  Avg Spread: {info['avg_spread_pips']:.1f} pips")
                    print(f"  Completeness: {info['data_completeness']:.1%}")
                    if info['issues']:
                        print(f"  Issues: {', '.join(info['issues'])}")
        
        elif args.command == 'list':
            data_info = manager.list_available_data()
            
            print("\nAVAILABLE DATA")
            print("="*50)
            print(f"Data Path: {data_info['data_path']}")
            print(f"Total Symbols: {data_info['summary']['total_symbols']}")
            print(f"Total Files: {data_info['summary']['total_files']}")
            print(f"Formats: {', '.join(data_info['summary']['data_formats'])}")
            
            print("\nSymbol Details:")
            for symbol, info in data_info['symbols'].items():
                if info['has_data']:
                    print(f"  {symbol}: {info['csv_files']} CSV, {info['bi5_files']} BI5 files")
        
        elif args.command == 'cleanup':
            results = manager.cleanup_data(args.days)
            
            print(f"\nCLEANUP RESULTS")
            print("="*50)
            print(f"Files Removed: {results['removed_files']}")
            print(f"Space Freed: {results['total_size_removed'] / (1024*1024):.1f} MB")
            print(f"Cutoff Date: {results['cutoff_date']}")
        
        elif args.command == 'export':
            manager.export_data_info(args.output_file)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()