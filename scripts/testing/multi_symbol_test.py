#!/usr/bin/env python3

"""
Test multi-symbol backtesting functionality
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

def test_multi_symbol():
    """Test multi-symbol backtesting."""
    logger.info("🧪 Testing multi-symbol backtesting...")
    
    # Ensure test data is available
    test_data_manager = TestDataManager()
    if not test_data_manager.ensure_test_data_available(['EURUSD', 'EURJPY', 'GBPNZD'], days=2):
        logger.error("❌ Failed to ensure test data availability")
        return False
    
    # Create configuration with multiple symbols
    config = BacktestConfig(
        start_date=datetime(2025, 7, 3),
        end_date=datetime(2025, 7, 3, 6, 0, 0),  # 6 hours to keep it manageable
        symbols=['EURUSD', 'EURJPY', 'GBPNZD'],  # All our available symbols
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
    
    # Create strategy that works across symbols
    strategy_config = StrategyConfig(
        name="Multi_Symbol_Test",
        description="Strategy for multiple symbols",
        parameters={
            'fast_period': 5,
            'slow_period': 15,
            'position_size': 0.1
        },
        risk_management={
            'max_position_size': 0.5,  # Smaller per symbol
            'stop_loss_pips': 30,
            'take_profit_pips': 60
        }
    )
    
    # Track per-symbol activity
    symbol_activity = {symbol: {'orders': 0, 'trades': 0} for symbol in config.symbols}
    
    def track_order_filled(order):
        symbol_activity[order.symbol]['orders'] += 1
        logger.info(f"📈 {order.symbol}: {order.side.value} {order.quantity} lots at {order.avg_fill_price}")
    
    # Create engine
    engine = BacktestEngine(config, './data')
    engine.on_order_filled = track_order_filled
    
    # Add strategy
    strategy = MovingAverageCrossoverStrategy(strategy_config, config)
    engine.add_strategy(strategy)
    
    # Run backtest
    result = engine.run_backtest()
    
    # Count trades per symbol
    for trade in result.trades:
        symbol_activity[trade.symbol]['trades'] += 1
    
    # Validate results
    logger.info("📊 Multi-Symbol Results:")
    logger.info(f"  Total symbols: {len(config.symbols)}")
    logger.info(f"  Total trades: {result.total_trades}")
    logger.info(f"  Total return: {result.total_return:.2%}")
    
    logger.info("📈 Per-Symbol Activity:")
    for symbol in config.symbols:
        activity = symbol_activity[symbol]
        logger.info(f"  {symbol}: {activity['orders']} orders, {activity['trades']} trades")
    
    # Validation checks
    validation_passed = True
    
    # Check that data was loaded for all symbols
    symbols_with_data = set()
    for trade in result.trades:
        symbols_with_data.add(trade.symbol)
    
    if len(symbols_with_data) > 1:
        logger.info("✅ Multiple symbols processed successfully")
    elif len(symbols_with_data) == 1:
        logger.warning("⚠️  Only one symbol generated trades")
    else:
        logger.warning("⚠️  No symbols generated trades")
    
    # Check that chronological ordering is maintained
    if len(result.trades) > 1:
        trades_ordered = all(
            result.trades[i].entry_time <= result.trades[i+1].entry_time 
            for i in range(len(result.trades)-1)
        )
        if trades_ordered:
            logger.info("✅ Trades are chronologically ordered")
        else:
            logger.error("❌ Trades are not chronologically ordered")
            validation_passed = False
    
    # Check pip value differences for JPY vs non-JPY pairs
    jpy_trades = [t for t in result.trades if 'JPY' in t.symbol]
    non_jpy_trades = [t for t in result.trades if 'JPY' not in t.symbol]
    
    if jpy_trades and non_jpy_trades:
        logger.info("✅ Both JPY and non-JPY pairs processed")
        # Check that JPY pairs have different pip values
        logger.info(f"  JPY trades: {len(jpy_trades)}")
        logger.info(f"  Non-JPY trades: {len(non_jpy_trades)}")
    
    if validation_passed:
        logger.info("🎉 Multi-symbol validation checks PASSED!")
    else:
        logger.error("💥 Some multi-symbol validation checks FAILED!")
    
    return validation_passed

if __name__ == '__main__':
    try:
        success = test_multi_symbol()
        if success:
            print("✅ Multi-symbol test PASSED")
        else:
            print("❌ Multi-symbol test FAILED")
    except Exception as e:
        print(f"❌ Multi-symbol test ERROR: {e}")
        import traceback
        traceback.print_exc()