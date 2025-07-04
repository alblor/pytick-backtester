"""
Complete example demonstrating how to use the backtesting framework.
Shows how to configure, run, and analyze a backtest with multiple strategies.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the backtester to the path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_structures import BacktestConfig
from engine.backtest_engine import BacktestEngine
from strategy.strategy_interface import StrategyConfig
from examples.moving_average_strategy import MovingAverageCrossoverStrategy, MovingAverageRSIStrategy
from analysis.performance_analyzer import PerformanceAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def create_backtest_config() -> BacktestConfig:
    """
    Create backtesting configuration.
    
    Returns:
        Configured BacktestConfig object
    """
    config = BacktestConfig(
        # Data settings
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),  # One month for example
        symbols=['EURUSD', 'GBPUSD'],
        
        # Execution settings
        spread_markup=0.5,  # Additional 0.5 pips spread
        slippage_model='linear',
        max_slippage=2.0,
        execution_delay_min=0.1,
        execution_delay_max=0.5,
        
        # Account settings
        initial_balance=100000.0,
        currency='USD',
        leverage=100.0,
        commission_per_lot=7.0,
        
        # Risk settings
        max_position_size=10.0,
        margin_requirement=0.01,
        margin_call_level=0.5,
        
        # Data quality settings
        interpolate_missing_ticks=True,
        max_gap_seconds=60.0
    )
    
    return config


def create_ma_strategy() -> MovingAverageCrossoverStrategy:
    """
    Create Moving Average Crossover strategy.
    
    Returns:
        Configured strategy
    """
    strategy_config = StrategyConfig(
        name="MA_Crossover_10_20",
        description="Moving Average Crossover Strategy with 10/20 periods",
        parameters={
            'fast_period': 10,
            'slow_period': 20,
            'min_spread_pips': 1,
            'max_spread_pips': 5,
            'risk_per_trade': 500  # Risk $500 per trade
        },
        risk_management={
            'max_position_size': 2.0,
            'stop_loss_pips': 30,
            'take_profit_pips': 60,
            'max_daily_loss': 2000,
            'max_drawdown': 0.15
        }
    )
    
    backtest_config = create_backtest_config()
    return MovingAverageCrossoverStrategy(strategy_config, backtest_config)


def create_ma_rsi_strategy() -> MovingAverageRSIStrategy:
    """
    Create Moving Average + RSI strategy.
    
    Returns:
        Configured strategy
    """
    strategy_config = StrategyConfig(
        name="MA_RSI_Combo",
        description="Moving Average Crossover with RSI confirmation",
        parameters={
            'fast_period': 8,
            'slow_period': 21,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'risk_per_trade': 750  # Risk $750 per trade
        },
        risk_management={
            'max_position_size': 1.5,
            'stop_loss_pips': 40,
            'take_profit_pips': 80,
            'max_daily_loss': 3000,
            'max_drawdown': 0.12
        }
    )
    
    backtest_config = create_backtest_config()
    return MovingAverageRSIStrategy(strategy_config, backtest_config)


def setup_progress_callback(engine: BacktestEngine) -> None:
    """
    Setup progress callback for monitoring backtest progress.
    
    Args:
        engine: Backtest engine
    """
    def progress_callback(progress: float, status_info: dict) -> None:
        """Progress callback function."""
        print(f"\rProgress: {progress:.1%} | "
              f"Ticks: {status_info['processed_ticks']:,} | "
              f"Balance: ${status_info['account_balance']:,.2f} | "
              f"Equity: ${status_info['account_equity']:,.2f} | "
              f"Positions: {status_info['active_positions']} | "
              f"Trades: {status_info['total_trades']}", end='')
    
    engine.on_progress_update = progress_callback


def run_single_strategy_backtest(data_path: str) -> None:
    """
    Run a backtest with a single strategy.
    
    Args:
        data_path: Path to data directory (supports both .bi5 and CSV)
    """
    logger.info("Starting single strategy backtest...")
    
    # Create configuration
    config = create_backtest_config()
    
    # Create engine
    engine = BacktestEngine(config, data_path)
    
    # Add strategy
    strategy = create_ma_strategy()
    engine.add_strategy(strategy)
    
    # Setup progress monitoring
    setup_progress_callback(engine)
    
    # Run backtest
    result = engine.run_backtest()
    
    print("\n")  # New line after progress
    
    # Analyze results
    analyzer = PerformanceAnalyzer(result)
    analyzer.print_summary()
    
    # Export detailed report
    analyzer.export_detailed_report('single_strategy_report.json', 'json')
    analyzer.export_detailed_report('single_strategy_report.xlsx', 'excel')
    
    logger.info("Single strategy backtest completed")


def run_multi_strategy_backtest(data_path: str) -> None:
    """
    Run a backtest with multiple strategies.
    
    Args:
        data_path: Path to data directory (supports both .bi5 and CSV)
    """
    logger.info("Starting multi-strategy backtest...")
    
    # Create configuration
    config = create_backtest_config()
    
    # Create engine
    engine = BacktestEngine(config, data_path)
    
    # Add multiple strategies
    ma_strategy = create_ma_strategy()
    ma_rsi_strategy = create_ma_rsi_strategy()
    
    engine.add_strategy(ma_strategy)
    engine.add_strategy(ma_rsi_strategy)
    
    # Setup progress monitoring
    setup_progress_callback(engine)
    
    # Run backtest
    result = engine.run_backtest()
    
    print("\n")  # New line after progress
    
    # Analyze results
    analyzer = PerformanceAnalyzer(result)
    analyzer.print_summary()
    
    # Export detailed report
    analyzer.export_detailed_report('multi_strategy_report.json', 'json')
    analyzer.export_detailed_report('multi_strategy_report.xlsx', 'excel')
    
    logger.info("Multi-strategy backtest completed")


def run_parameter_optimization(data_path: str) -> None:
    """
    Run parameter optimization for a strategy.
    
    Args:
        data_path: Path to data directory (supports both .bi5 and CSV)
    """
    logger.info("Starting parameter optimization...")
    
    # Parameter combinations to test
    fast_periods = [5, 8, 10, 12]
    slow_periods = [15, 20, 25, 30]
    
    results = []
    
    for fast_period in fast_periods:
        for slow_period in slow_periods:
            if fast_period >= slow_period:
                continue
            
            logger.info(f"Testing MA({fast_period},{slow_period})")
            
            # Create configuration
            config = create_backtest_config()
            
            # Create engine
            engine = BacktestEngine(config, data_path)
            
            # Create strategy with current parameters
            strategy_config = StrategyConfig(
                name=f"MA_{fast_period}_{slow_period}",
                description=f"MA Crossover {fast_period}/{slow_period}",
                parameters={
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'min_spread_pips': 1,
                    'max_spread_pips': 5,
                    'risk_per_trade': 500
                },
                risk_management={
                    'max_position_size': 2.0,
                    'stop_loss_pips': 30,
                    'take_profit_pips': 60,
                    'max_daily_loss': 2000,
                    'max_drawdown': 0.15
                }
            )
            
            strategy = MovingAverageCrossoverStrategy(strategy_config, config)
            engine.add_strategy(strategy)
            
            # Run backtest
            result = engine.run_backtest()
            
            # Store results
            results.append({
                'fast_period': fast_period,
                'slow_period': slow_period,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'profit_factor': result.profit_factor
            })
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['sharpe_ratio'])
    
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best parameters: MA({best_result['fast_period']},{best_result['slow_period']})")
    print(f"Total Return: {best_result['total_return']:.2%}")
    print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {best_result['max_drawdown']:.2%}")
    print(f"Win Rate: {best_result['win_rate']:.2%}")
    print(f"Total Trades: {best_result['total_trades']}")
    print(f"Profit Factor: {best_result['profit_factor']:.2f}")
    
    # Export optimization results
    import json
    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Parameter optimization completed")


def main():
    """Main function to run examples."""
    # Set data path (adjust this to your data location)
    data_path = "./data"  # Default to local data directory
    
    # Check if data path exists
    if not Path(data_path).exists():
        logger.error(f"Data path does not exist: {data_path}")
        logger.info("Please download data first using:")
        logger.info("  python scripts/data_manager.py quick")
        logger.info("  or")
        logger.info("  npm run fetch-single")
        return
    
    try:
        # Run examples
        print("="*60)
        print("BACKTESTING FRAMEWORK EXAMPLES")
        print("="*60)
        
        print("\n1. Single Strategy Backtest:")
        run_single_strategy_backtest(data_path)
        
        print("\n2. Multi-Strategy Backtest:")
        run_multi_strategy_backtest(data_path)
        
        print("\n3. Parameter Optimization:")
        run_parameter_optimization(data_path)
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()