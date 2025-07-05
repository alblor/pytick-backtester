#!/usr/bin/env python3

"""
Test performance analysis and reporting functionality
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
import json

# Add backtester to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_structures import BacktestConfig
from engine.backtest_engine import BacktestEngine
from strategy.strategy_interface import StrategyConfig
from examples.moving_average_strategy import MovingAverageCrossoverStrategy
from analysis.performance_analyzer import PerformanceAnalyzer
from test_data_utils import TestDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_performance_analysis():
    """Test performance analysis and reporting."""
    logger.info("🧪 Testing performance analysis and reporting...")
    
    # Ensure test data is available
    test_data_manager = TestDataManager()
    if not test_data_manager.ensure_test_data_available(['EURUSD'], days=2):
        logger.error("❌ Failed to ensure test data availability")
        return False
    
    # Create configuration that should generate multiple trades
    config = BacktestConfig(
        start_date=datetime(2025, 7, 3),
        end_date=datetime(2025, 7, 4),
        symbols=['EURUSD'],
        spread_markup=0.5,
        slippage_model='linear',
        max_slippage=1.0,
        execution_delay_min=0.0,
        execution_delay_max=0.1,
        initial_balance=100000.0,
        leverage=100.0,
        commission_per_lot=7.0,
        max_position_size=10.0,
        margin_requirement=0.01,
        interpolate_missing_ticks=False,
        max_gap_seconds=60.0
    )
    
    # Create aggressive strategy to generate more trades
    strategy_config = StrategyConfig(
        name="Performance_Test",
        description="Strategy for performance analysis testing",
        parameters={
            'fast_period': 3,   # Very fast to get more signals
            'slow_period': 8,
            'position_size': 0.2
        },
        risk_management={
            'max_position_size': 1.0,
            'stop_loss_pips': 25,
            'take_profit_pips': 50
        }
    )
    
    # Run backtest
    engine = BacktestEngine(config, './data')
    strategy = MovingAverageCrossoverStrategy(strategy_config, config)
    engine.add_strategy(strategy)
    
    result = engine.run_backtest()
    
    logger.info(f"📊 Backtest generated {result.total_trades} trades")
    
    # Test performance analyzer
    analyzer = PerformanceAnalyzer(result)
    
    # Test summary report
    try:
        analyzer.print_summary()
        logger.info("✅ Summary report generated successfully")
        summary_passed = True
    except Exception as e:
        logger.error(f"❌ Summary report failed: {e}")
        summary_passed = False
    
    # Test detailed metrics
    try:
        summary_data = analyzer.generate_summary_report()
        
        # Check required sections
        required_sections = [
            'basic_metrics',
            'trade_analysis', 
            'risk_analysis',
            'execution_analysis',
            'symbol_analysis',
            'time_analysis'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in summary_data:
                missing_sections.append(section)
        
        if missing_sections:
            logger.error(f"❌ Missing report sections: {missing_sections}")
            metrics_passed = False
        else:
            logger.info("✅ All required report sections present")
            
            # Check specific metrics
            basic_metrics = summary_data['basic_metrics']
            logger.info(f"  Total Return: {basic_metrics.get('total_return', 0):.2%}")
            logger.info(f"  Final Balance: ${basic_metrics.get('final_balance', 0):,.2f}")
            
            trade_analysis = summary_data['trade_analysis']
            logger.info(f"  Total Trades: {trade_analysis.get('total_trades', 0)}")
            logger.info(f"  Win Rate: {trade_analysis.get('win_rate', 0):.1%}")
            
            risk_analysis = summary_data['risk_analysis']
            logger.info(f"  Max Drawdown: {risk_analysis.get('max_drawdown', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {risk_analysis.get('sharpe_ratio', 0):.2f}")
            
            execution_analysis = summary_data['execution_analysis']
            logger.info(f"  Avg Slippage: {execution_analysis.get('avg_slippage', 0):.2f} pips")
            logger.info(f"  Total Commission: ${execution_analysis.get('total_commission', 0):.2f}")
            
            metrics_passed = True
            
    except Exception as e:
        logger.error(f"❌ Detailed metrics failed: {e}")
        metrics_passed = False
    
    # Test export functionality
    export_passed = True
    try:
        # Test JSON export
        analyzer.export_detailed_report('test_results/test_report.json', 'json')
        
        # Verify JSON file was created and is valid
        with open('test_results/test_report.json', 'r') as f:
            json_data = json.load(f)
        
        # Check for the actual sections that should be present
        required_json_sections = ['basic_metrics', 'trade_analysis', 'risk_analysis', 'execution_analysis']
        missing_json_sections = [section for section in required_json_sections if section not in json_data]
        
        if missing_json_sections:
            logger.error(f"❌ JSON export missing required sections: {missing_json_sections}")
            export_passed = False
        else:
            logger.info("✅ JSON export successful")
            
    except Exception as e:
        logger.error(f"❌ JSON export failed: {e}")
        export_passed = False
    
    try:
        # Test Excel export (if available)
        analyzer.export_detailed_report('test_results/test_report.xlsx', 'excel')
        logger.info("✅ Excel export successful")
    except Exception as e:
        logger.warning(f"⚠️  Excel export not available: {e}")
        # Don't fail the test for Excel since it's optional
    
    # Test trade-by-trade analysis
    trade_analysis_passed = True
    try:
        if result.trades:
            for i, trade in enumerate(result.trades[:3]):  # Check first 3 trades
                logger.info(f"  Trade {i+1}: {trade.symbol} {trade.side.value} "
                          f"Entry={trade.entry_price:.5f} Exit={trade.exit_price:.5f} "
                          f"P&L={trade.net_pnl:.2f} Duration={trade.duration:.0f}s")
            
            # Calculate some basic trade statistics
            winning_trades = [t for t in result.trades if t.net_pnl > 0]
            losing_trades = [t for t in result.trades if t.net_pnl <= 0]
            
            logger.info(f"  Winning trades: {len(winning_trades)}")
            logger.info(f"  Losing trades: {len(losing_trades)}")
            
            if winning_trades:
                avg_win = sum(t.net_pnl for t in winning_trades) / len(winning_trades)
                logger.info(f"  Average win: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = sum(t.net_pnl for t in losing_trades) / len(losing_trades)
                logger.info(f"  Average loss: ${avg_loss:.2f}")
            
            logger.info("✅ Trade-by-trade analysis successful")
            
        else:
            logger.warning("⚠️  No trades to analyze")
            
    except Exception as e:
        logger.error(f"❌ Trade analysis failed: {e}")
        trade_analysis_passed = False
    
    # Overall validation
    all_passed = all([
        summary_passed,
        metrics_passed,
        export_passed,
        trade_analysis_passed
    ])
    
    if all_passed:
        logger.info("🎉 All performance analysis tests PASSED!")
    else:
        logger.error("💥 Some performance analysis tests FAILED!")
    
    # Clean up test files
    try:
        Path('test_results/test_report.json').unlink(missing_ok=True)
        Path('test_results/test_report.xlsx').unlink(missing_ok=True)
    except:
        pass
    
    return all_passed

if __name__ == '__main__':
    try:
        success = test_performance_analysis()
        if success:
            print("✅ Performance analysis test PASSED")
        else:
            print("❌ Performance analysis test FAILED")
    except Exception as e:
        print(f"❌ Performance analysis test ERROR: {e}")
        import traceback
        traceback.print_exc()