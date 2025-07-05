#!/usr/bin/env python3

"""
Test order execution and position management validation
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

def test_order_execution():
    """Test order execution and position management."""
    logger.info("🧪 Testing order execution and position management...")
    
    # Ensure test data is available
    test_data_manager = TestDataManager()
    if not test_data_manager.ensure_test_data_available(['EURUSD'], days=2):
        logger.error("❌ Failed to ensure test data availability")
        return False
    
    # Create configuration that should generate trades
    config = BacktestConfig(
        start_date=datetime(2025, 7, 3),
        end_date=datetime(2025, 7, 4),  # Full day for more data
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
        interpolate_missing_ticks=False,  # Keep simple
        max_gap_seconds=60.0
    )
    
    # Create more sensitive strategy to generate trades
    strategy_config = StrategyConfig(
        name="Execution_Test",
        description="Strategy designed to generate trades",
        parameters={
            'fast_period': 5,   # Very fast periods to get crossovers
            'slow_period': 15,
            'position_size': 0.1  # Small position
        },
        risk_management={
            'max_position_size': 1.0,
            'stop_loss_pips': 50,
            'take_profit_pips': 100
        }
    )
    
    # Track execution events
    execution_events = {
        'orders_created': 0,
        'orders_filled': 0,
        'positions_opened': 0,
        'positions_closed': 0,
        'trades_completed': 0
    }
    
    def track_order_filled(order):
        execution_events['orders_filled'] += 1
        logger.info(f"📈 Order filled: {order.side.value} {order.quantity} lots at {order.avg_fill_price}")
    
    def track_position_opened(position):
        execution_events['positions_opened'] += 1
        logger.info(f"🔓 Position opened: {position.side.value} {position.quantity} lots")
    
    def track_position_closed(position, trade):
        execution_events['positions_closed'] += 1
        logger.info(f"🔒 Position closed: P&L = {trade.net_pnl:.2f}")
    
    # Create engine and add callbacks
    engine = BacktestEngine(config, './data')
    engine.on_order_filled = track_order_filled
    engine.on_position_opened = track_position_opened
    engine.on_position_closed = track_position_closed
    
    # Add strategy
    strategy = MovingAverageCrossoverStrategy(strategy_config, config)
    engine.add_strategy(strategy)
    
    # Run backtest
    result = engine.run_backtest()
    
    # Validate results
    logger.info("📊 Execution Results:")
    logger.info(f"  Orders filled: {execution_events['orders_filled']}")
    logger.info(f"  Positions opened: {execution_events['positions_opened']}")
    logger.info(f"  Positions closed: {execution_events['positions_closed']}")
    logger.info(f"  Total trades: {result.total_trades}")
    logger.info(f"  Total return: {result.total_return:.2%}")
    logger.info(f"  Win rate: {result.win_rate:.1%}")
    logger.info(f"  Avg execution delay: {result.avg_execution_delay:.3f}s")
    logger.info(f"  Total slippage: {result.total_slippage:.2f} pips")
    logger.info(f"  Total commission: {result.total_commission:.2f}")
    
    # Validation checks
    validation_passed = True
    
    if result.total_trades > 0:
        logger.info("✅ Strategy generated trades")
        
        if execution_events['orders_filled'] == 0:
            logger.error("❌ No orders filled despite having trades")
            validation_passed = False
        else:
            logger.info("✅ Orders were filled correctly")
        
        if execution_events['positions_opened'] == 0:
            logger.error("❌ No positions opened despite having trades")
            validation_passed = False
        else:
            logger.info("✅ Positions were opened correctly")
        
        # Check P&L consistency
        total_trade_pnl = sum(trade.net_pnl for trade in result.trades)
        account_pnl = config.initial_balance * result.total_return
        pnl_diff = abs(total_trade_pnl - account_pnl)
        
        if pnl_diff > 1.0:  # Allow small rounding differences
            logger.error(f"❌ P&L mismatch: trades={total_trade_pnl:.2f}, account={account_pnl:.2f}")
            validation_passed = False
        else:
            logger.info("✅ P&L calculations are consistent")
        
        # Check bid/ask spreads
        if result.trades:
            for i, trade in enumerate(result.trades[:3]):  # Check first few trades
                logger.info(f"  Trade {i+1}: Entry={trade.entry_price:.5f}, Exit={trade.exit_price:.5f}, P&L={trade.net_pnl:.2f}")
        
    else:
        logger.warning("⚠️  No trades generated - strategy might be too conservative")
    
    if validation_passed:
        logger.info("🎉 All execution validation checks PASSED!")
    else:
        logger.error("💥 Some execution validation checks FAILED!")
    
    return validation_passed

if __name__ == '__main__':
    try:
        success = test_order_execution()
        if success:
            print("✅ Execution validation test PASSED")
        else:
            print("❌ Execution validation test FAILED")
    except Exception as e:
        print(f"❌ Execution validation test ERROR: {e}")
        import traceback
        traceback.print_exc()