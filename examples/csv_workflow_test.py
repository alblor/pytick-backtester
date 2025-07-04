#!/usr/bin/env python3

"""
Complete CSV Workflow Test for the Backtesting Framework.
Tests data download, loading, backtesting, and analysis with CSV data.
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import time

# Add backtester to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_structures import BacktestConfig
from engine.backtest_engine import BacktestEngine
from strategy.strategy_interface import StrategyConfig
from examples.moving_average_strategy import MovingAverageCrossoverStrategy
from analysis.performance_analyzer import PerformanceAnalyzer
from data.data_loader_factory import DataLoaderFactory


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_download() -> bool:
    """Test data download using dukascopy-node."""
    logger.info("🚀 Testing data download...")
    
    try:
        # Use fetch_single_pair.js for quick testing
        result = subprocess.run(
            ['node', 'scripts/fetch_single_pair.js'],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ Data download successful")
            print(result.stdout)
            return True
        else:
            logger.error("❌ Data download failed")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Data download timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Data download error: {e}")
        return False


def test_data_loading() -> bool:
    """Test CSV data loading."""
    logger.info("📊 Testing CSV data loading...")
    
    try:
        # Create minimal config for testing (use sample data dates)
        config = BacktestConfig(
            start_date=datetime(2024, 1, 2, 0, 0, 0),
            end_date=datetime(2024, 1, 2, 23, 59, 59),
            symbols=['EURUSD'],
            initial_balance=10000
        )
        
        # Test data loader factory
        data_loader = DataLoaderFactory.create_loader('./data', config)
        logger.info(f"✅ Created data loader: {type(data_loader).__name__}")
        
        # Test data loading
        ticks = list(data_loader.load_symbol_data('EURUSD', config.start_date, config.end_date))
        
        if len(ticks) > 0:
            logger.info(f"✅ Loaded {len(ticks)} ticks for EURUSD")
            
            # Show sample data
            sample_tick = ticks[0]
            logger.info(f"📋 Sample tick: {sample_tick.timestamp} - Bid: {sample_tick.bid}, Ask: {sample_tick.ask}")
            
            return True
        else:
            logger.warning("⚠️ No ticks loaded - this may be expected if no data is available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return False


def test_data_validation() -> bool:
    """Test data validation."""
    logger.info("🔍 Testing data validation...")
    
    try:
        config = BacktestConfig(
            start_date=datetime(2024, 1, 2, 0, 0, 0),
            end_date=datetime(2024, 1, 2, 23, 59, 59),
            symbols=['EURUSD'],
            initial_balance=10000
        )
        
        data_loader = DataLoaderFactory.create_loader('./data', config)
        
        # Get data summary
        summary = data_loader.get_data_summary('EURUSD', config.start_date, config.end_date)
        
        logger.info("📊 Data Summary:")
        logger.info(f"  Tick Count: {summary['tick_count']}")
        logger.info(f"  Interpolation Rate: {summary['interpolation_rate']:.2%}")
        logger.info(f"  Avg Spread: {summary['avg_spread_pips']:.2f} pips")
        logger.info(f"  Data Completeness: {summary['data_completeness']:.2%}")
        
        # Test integrity
        integrity_ok = data_loader.verify_data_integrity('EURUSD', config.start_date, config.end_date)
        
        if integrity_ok:
            logger.info("✅ Data integrity check passed")
            return True
        else:
            logger.warning("⚠️ Data integrity check failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data validation failed: {e}")
        return False


def test_backtesting() -> bool:
    """Test complete backtesting workflow."""
    logger.info("🎯 Testing backtesting workflow...")
    
    try:
        # Create configuration
        config = BacktestConfig(
            start_date=datetime(2024, 1, 2, 0, 0, 0),
            end_date=datetime(2024, 1, 2, 23, 59, 59),
            symbols=['EURUSD'],
            initial_balance=10000,
            spread_markup=0.5,
            max_slippage=1.0,
            commission_per_lot=5.0
        )
        
        # Create strategy
        strategy_config = StrategyConfig(
            name="Test_MA_Strategy",
            description="Moving Average test strategy",
            parameters={
                'fast_period': 5,
                'slow_period': 10,
                'min_spread_pips': 1,
                'max_spread_pips': 5
            },
            risk_management={
                'max_position_size': 0.1,
                'stop_loss_pips': 20,
                'take_profit_pips': 40
            }
        )
        
        strategy = MovingAverageCrossoverStrategy(strategy_config, config)
        
        # Create engine
        engine = BacktestEngine(config, './data')
        engine.add_strategy(strategy)
        
        # Add progress callback
        def progress_callback(progress, status):
            if progress > 0:
                logger.info(f"Progress: {progress:.1%} - Processed {status['processed_ticks']} ticks")
        
        engine.on_progress_update = progress_callback
        
        # Run backtest
        logger.info("🚀 Running backtest...")
        result = engine.run_backtest()
        
        if result:
            logger.info("✅ Backtest completed successfully")
            logger.info(f"📊 Total trades: {result.total_trades}")
            logger.info(f"💰 Total return: {result.total_return:.2%}")
            logger.info(f"📈 Win rate: {result.win_rate:.2%}")
            
            return True
        else:
            logger.error("❌ Backtest failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_analysis() -> bool:
    """Test performance analysis."""
    logger.info("📈 Testing performance analysis...")
    
    try:
        # Run a quick backtest to get results
        config = BacktestConfig(
            start_date=datetime(2024, 1, 2, 0, 0, 0),
            end_date=datetime(2024, 1, 2, 23, 59, 59),
            symbols=['EURUSD'],
            initial_balance=10000
        )
        
        strategy_config = StrategyConfig(
            name="Analysis_Test",
            description="Test strategy for analysis",
            parameters={'fast_period': 5, 'slow_period': 10},
            risk_management={'max_position_size': 0.1}
        )
        
        strategy = MovingAverageCrossoverStrategy(strategy_config, config)
        engine = BacktestEngine(config, './data')
        engine.add_strategy(strategy)
        
        result = engine.run_backtest()
        
        if result:
            # Test performance analyzer
            analyzer = PerformanceAnalyzer(result)
            
            # Generate summary
            summary = analyzer.generate_summary_report()
            
            logger.info("✅ Performance analysis completed")
            logger.info(f"📊 Analysis includes {len(summary)} sections")
            
            # Test export
            analyzer.export_detailed_report('test_analysis.json', 'json')
            logger.info("✅ Analysis export completed")
            
            return True
        else:
            logger.warning("⚠️ No backtest results to analyze")
            return False
            
    except Exception as e:
        logger.error(f"❌ Performance analysis failed: {e}")
        return False


def test_data_formats() -> bool:
    """Test data format detection."""
    logger.info("🔍 Testing data format detection...")
    
    try:
        symbols = ['EURUSD']
        format_info = DataLoaderFactory.get_data_format_info('./data', symbols)
        
        logger.info("📋 Data Format Information:")
        logger.info(f"  Data Path: {format_info['data_path']}")
        logger.info(f"  Has BI5 Data: {format_info['has_bi5_data']}")
        logger.info(f"  Has CSV Data: {format_info['has_csv_data']}")
        logger.info(f"  Recommended Format: {format_info['recommended_format']}")
        
        for symbol, details in format_info['symbol_details'].items():
            logger.info(f"  {symbol}: {details['csv_files']} CSV files, {details['bi5_files']} BI5 files")
        
        logger.info("✅ Data format detection completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data format detection failed: {e}")
        return False


def run_complete_workflow_test() -> None:
    """Run complete CSV workflow test."""
    logger.info("🚀 Starting Complete CSV Workflow Test")
    logger.info("="*60)
    
    tests = [
        ("Data Download", test_data_download),
        ("Data Loading", test_data_loading),
        ("Data Validation", test_data_validation),
        ("Data Format Detection", test_data_formats),
        ("Backtesting", test_backtesting),
        ("Performance Analysis", test_performance_analysis),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        logger.info("-" * 40)
        
        test_start = time.time()
        
        try:
            success = test_func()
            test_duration = time.time() - test_start
            
            results.append({
                'name': test_name,
                'success': success,
                'duration': test_duration
            })
            
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{status} - {test_name} ({test_duration:.1f}s)")
            
        except Exception as e:
            test_duration = time.time() - test_start
            results.append({
                'name': test_name,
                'success': False,
                'duration': test_duration,
                'error': str(e)
            })
            logger.error(f"❌ FAILED - {test_name}: {e}")
    
    # Summary
    total_duration = time.time() - start_time
    passed_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {passed_tests/total_tests:.1%}")
    logger.info(f"Total Duration: {total_duration:.1f}s")
    
    logger.info("\nDetailed Results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        duration = result['duration']
        logger.info(f"  {status} {result['name']} ({duration:.1f}s)")
        
        if not result['success'] and 'error' in result:
            logger.info(f"      Error: {result['error']}")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 All tests passed! CSV workflow is working correctly.")
    else:
        logger.info(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Check the logs above for details.")
    
    logger.info("="*60)


if __name__ == '__main__':
    try:
        run_complete_workflow_test()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()