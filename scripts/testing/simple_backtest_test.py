#!/usr/bin/env python3

"""
Simple backtest test to debug timezone issues
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add backtester to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_structures import BacktestConfig
from engine.backtest_engine import BacktestEngine
from strategy.strategy_interface import StrategyConfig
from examples.moving_average_strategy import MovingAverageCrossoverStrategy
from test_data_utils import TestDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_backtest():
    """Test a simple backtest to debug issues."""
    logger.info("🧪 Testing simple backtest...")
    
    # Ensure test data is available
    test_data_manager = TestDataManager()
    if not test_data_manager.ensure_test_data_available(['EURUSD'], days=2):
        logger.error("❌ Failed to ensure test data availability")
        return False
    
    # Create simple configuration
    config = BacktestConfig(
        start_date=datetime(2025, 7, 3),
        end_date=datetime(2025, 7, 3, 1, 0, 0),  # Just 1 hour
        symbols=['EURUSD'],
        spread_markup=0.0,
        slippage_model='linear',
        max_slippage=1.0,
        execution_delay_min=0.0,
        execution_delay_max=0.0,
        initial_balance=100000.0,
        leverage=100.0,
        commission_per_lot=7.0,
        max_position_size=10.0,
        margin_requirement=0.01,
        interpolate_missing_ticks=False,  # Disable interpolation to avoid timestamp issues
        max_gap_seconds=60.0
    )
    
    # Create strategy
    strategy_config = StrategyConfig(
        name="Simple_Test",
        description="Simple Test Strategy",
        parameters={
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.1  # Small position
        },
        risk_management={
            'max_position_size': 1.0,
            'stop_loss_pips': 20,
            'take_profit_pips': 40
        }
    )
    
    # Create engine
    engine = BacktestEngine(config, './data')
    strategy = MovingAverageCrossoverStrategy(strategy_config, config)
    engine.add_strategy(strategy)
    
    # Run backtest
    result = engine.run_backtest()
    
    logger.info(f"✅ Simple backtest completed successfully!")
    logger.info(f"Total trades: {result.total_trades}")
    logger.info(f"Total return: {result.total_return:.2%}")
    
    return True

if __name__ == '__main__':
    try:
        test_simple_backtest()
        print("✅ Simple test PASSED")
    except Exception as e:
        print(f"❌ Simple test FAILED: {e}")
        import traceback
        traceback.print_exc()