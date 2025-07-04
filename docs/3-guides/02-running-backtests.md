# Running Backtests

## Overview

This guide covers everything you need to know about executing backtests, from basic single-strategy tests to complex multi-strategy portfolio simulations.

## Basic Backtest Setup

### Simple Single Strategy Backtest
```python
from datetime import datetime
from backtester import BacktestEngine, BacktestConfig
from examples.moving_average_strategy import MovingAverageCrossoverStrategy
from strategy.strategy_interface import StrategyConfig

# 1. Configure the backtest
config = BacktestConfig(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    symbols=['EURUSD'],
    initial_balance=100000,
    spread_markup=0.5,
    max_slippage=2.0,
    commission_per_lot=7.0
)

# 2. Configure the strategy
strategy_config = StrategyConfig(
    name="MA_Cross_10_30",
    description="10/30 Moving Average Crossover",
    parameters={
        'fast_period': 10,
        'slow_period': 30,
        'min_spread_pips': 1,
        'max_spread_pips': 5
    },
    risk_management={
        'max_position_size': 1.0,
        'stop_loss_pips': 30,
        'take_profit_pips': 60,
        'risk_per_trade': 1000
    }
)

# 3. Create strategy and engine
strategy = MovingAverageCrossoverStrategy(strategy_config, config)
engine = BacktestEngine(config, './data')
engine.add_strategy(strategy)

# 4. Run the backtest
print("Starting backtest...")
result = engine.run_backtest()

# 5. Display results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

### Multi-Symbol Backtest
```python
# Test strategy across multiple currency pairs
config = BacktestConfig(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    symbols=['EURUSD', 'GBPUSD', 'USDJPY'],  # Multiple symbols
    initial_balance=100000,
    spread_markup=0.5,
    max_slippage=2.0,
    commission_per_lot=7.0
)

# Create strategy for each symbol
strategies = []
for symbol in config.symbols:
    strategy_config = StrategyConfig(
        name=f"MA_Cross_{symbol}",
        description=f"MA Crossover for {symbol}",
        parameters={
            'fast_period': 10,
            'slow_period': 30,
            'symbol_specific': True
        },
        risk_management={
            'max_position_size': 0.5,  # Smaller size per symbol
            'stop_loss_pips': 30,
            'take_profit_pips': 60,
            'risk_per_trade': 500      # Reduced risk per symbol
        }
    )
    
    strategy = MovingAverageCrossoverStrategy(strategy_config, config)
    strategies.append(strategy)

# Add all strategies to engine
engine = BacktestEngine(config, './data')
for strategy in strategies:
    engine.add_strategy(strategy)

result = engine.run_backtest()
```

## Advanced Backtest Configuration

### Comprehensive Configuration
```python
config = BacktestConfig(
    # Time period
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    
    # Symbols to trade
    symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    
    # Initial capital
    initial_balance=100000,
    
    # Trading costs
    spread_markup=0.5,           # Additional spread markup (pips)
    commission_per_lot=7.0,      # Commission per standard lot
    commission_type='per_lot',   # 'per_lot' or 'percentage'
    
    # Slippage modeling
    max_slippage=2.0,           # Maximum slippage (pips)
    slippage_type='random',     # 'fixed', 'random', or 'linear'
    
    # Execution delays
    min_execution_delay=0.1,    # Minimum delay (seconds)
    max_execution_delay=0.5,    # Maximum delay (seconds)
    
    # Risk management
    max_leverage=30,            # Maximum leverage
    margin_requirement=0.02,    # Margin requirement (2%)
    margin_call_level=0.5,      # Margin call at 50%
    
    # Data processing
    interpolate_missing_ticks=True,  # Fill missing data
    max_gap_seconds=60,             # Maximum gap to fill
    
    # Performance
    enable_progress_bar=True,    # Show progress during backtest
    cache_data=True,            # Cache loaded data
    parallel_processing=False    # Use parallel processing
)
```

### Risk Management Configuration
```python
# Advanced risk management settings
risk_config = {
    'max_daily_loss': 2000,           # Maximum daily loss
    'max_weekly_loss': 5000,          # Maximum weekly loss
    'max_monthly_loss': 10000,        # Maximum monthly loss
    'max_drawdown_pct': 0.15,         # Maximum drawdown (15%)
    'max_open_positions': 5,          # Maximum concurrent positions
    'max_symbol_exposure': 2.0,       # Maximum lots per symbol
    'position_sizing_method': 'risk_based',  # Position sizing method
    'risk_per_trade_pct': 0.02,       # Risk 2% per trade
    'stop_loss_method': 'trailing',    # Stop loss method
    'take_profit_method': 'fixed'      # Take profit method
}

# Apply risk management to strategy
strategy_config.risk_management.update(risk_config)
```

## Multi-Strategy Backtesting

### Portfolio of Strategies
```python
from examples.moving_average_strategy import MovingAverageCrossoverStrategy
from examples.rsi_strategy import RSIStrategy
from examples.bollinger_bands_strategy import BollingerBandsStrategy

# Create multiple strategies
strategies = []

# Strategy 1: Moving Average Crossover
ma_config = StrategyConfig(
    name="MA_Crossover",
    description="Moving Average Crossover Strategy",
    parameters={'fast_period': 10, 'slow_period': 30},
    risk_management={'max_position_size': 0.5, 'risk_per_trade': 500}
)
strategies.append(MovingAverageCrossoverStrategy(ma_config, config))

# Strategy 2: RSI Mean Reversion
rsi_config = StrategyConfig(
    name="RSI_MeanReversion",
    description="RSI Mean Reversion Strategy",
    parameters={'rsi_period': 14, 'oversold': 30, 'overbought': 70},
    risk_management={'max_position_size': 0.5, 'risk_per_trade': 500}
)
strategies.append(RSIStrategy(rsi_config, config))

# Strategy 3: Bollinger Bands
bb_config = StrategyConfig(
    name="BollingerBands",
    description="Bollinger Bands Breakout Strategy",
    parameters={'bb_period': 20, 'bb_std': 2.0},
    risk_management={'max_position_size': 0.5, 'risk_per_trade': 500}
)
strategies.append(BollingerBandsStrategy(bb_config, config))

# Run portfolio backtest
engine = BacktestEngine(config, './data')
for strategy in strategies:
    engine.add_strategy(strategy)

portfolio_result = engine.run_backtest()

# Analyze individual strategy performance
for strategy in strategies:
    individual_result = engine.get_strategy_result(strategy.config.name)
    print(f"{strategy.config.name}: {individual_result.total_return:.2%}")
```

## Parameter Optimization

### Grid Search Optimization
```python
from itertools import product

def optimize_strategy_parameters(base_config, parameter_grid):
    """Optimize strategy parameters using grid search"""
    
    best_result = None
    best_params = None
    best_sharpe = -999
    
    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())
    
    results = []
    
    for params in product(*param_values):
        # Create parameter combination
        param_dict = dict(zip(param_names, params))
        
        # Update strategy configuration
        test_config = base_config.copy()
        test_config.parameters.update(param_dict)
        
        # Run backtest
        strategy = MovingAverageCrossoverStrategy(test_config, config)
        engine = BacktestEngine(config, './data')
        engine.add_strategy(strategy)
        
        result = engine.run_backtest()
        
        # Store result
        results.append({
            'parameters': param_dict,
            'result': result,
            'sharpe_ratio': result.sharpe_ratio,
            'total_return': result.total_return,
            'max_drawdown': result.max_drawdown
        })
        
        # Check if this is the best result
        if result.sharpe_ratio > best_sharpe:
            best_sharpe = result.sharpe_ratio
            best_params = param_dict
            best_result = result
    
    return best_params, best_result, results

# Define parameter grid
parameter_grid = {
    'fast_period': [5, 10, 15, 20],
    'slow_period': [25, 30, 35, 40],
    'stop_loss_pips': [20, 30, 40],
    'take_profit_pips': [40, 60, 80]
}

# Run optimization
best_params, best_result, all_results = optimize_strategy_parameters(
    strategy_config, parameter_grid
)

print(f"Best parameters: {best_params}")
print(f"Best Sharpe ratio: {best_result.sharpe_ratio:.2f}")
```

### Walk-Forward Analysis
```python
from datetime import datetime, timedelta

def walk_forward_analysis(strategy_class, config, optimization_window_days=90, 
                         test_window_days=30, parameter_grid=None):
    """Perform walk-forward analysis"""
    
    results = []
    current_date = config.start_date
    
    while current_date < config.end_date:
        # Define optimization period
        opt_start = current_date
        opt_end = current_date + timedelta(days=optimization_window_days)
        
        # Define test period
        test_start = opt_end
        test_end = opt_end + timedelta(days=test_window_days)
        
        # Skip if test period exceeds end date
        if test_end > config.end_date:
            break
        
        # Optimize parameters on training data
        opt_config = config.copy()
        opt_config.start_date = opt_start
        opt_config.end_date = opt_end
        
        if parameter_grid:
            best_params, _, _ = optimize_strategy_parameters(
                strategy_config, parameter_grid
            )
        else:
            best_params = strategy_config.parameters
        
        # Test on out-of-sample data
        test_config = config.copy()
        test_config.start_date = test_start
        test_config.end_date = test_end
        
        # Update strategy with optimized parameters
        test_strategy_config = strategy_config.copy()
        test_strategy_config.parameters.update(best_params)
        
        # Run test
        strategy = strategy_class(test_strategy_config, test_config)
        engine = BacktestEngine(test_config, './data')
        engine.add_strategy(strategy)
        
        result = engine.run_backtest()
        
        # Store result
        results.append({
            'optimization_period': (opt_start, opt_end),
            'test_period': (test_start, test_end),
            'optimized_parameters': best_params,
            'result': result
        })
        
        # Move to next period
        current_date = test_end
    
    return results

# Run walk-forward analysis
wf_results = walk_forward_analysis(
    MovingAverageCrossoverStrategy,
    config,
    optimization_window_days=90,
    test_window_days=30,
    parameter_grid=parameter_grid
)

# Analyze walk-forward results
total_returns = [r['result'].total_return for r in wf_results]
avg_return = sum(total_returns) / len(total_returns)
print(f"Walk-forward average return: {avg_return:.2%}")
```

## Performance Monitoring

### Real-Time Progress Monitoring
```python
class ProgressMonitor:
    def __init__(self, total_ticks):
        self.total_ticks = total_ticks
        self.processed_ticks = 0
        self.start_time = time.time()
        self.last_update = time.time()
    
    def update(self, processed_ticks):
        self.processed_ticks = processed_ticks
        current_time = time.time()
        
        # Update every 5 seconds
        if current_time - self.last_update > 5:
            self.print_progress()
            self.last_update = current_time
    
    def print_progress(self):
        progress = self.processed_ticks / self.total_ticks
        elapsed = time.time() - self.start_time
        eta = (elapsed / progress) - elapsed if progress > 0 else 0
        
        print(f"Progress: {progress:.1%} | "
              f"Processed: {self.processed_ticks:,} | "
              f"ETA: {eta:.0f}s")

# Use with backtest engine
def run_backtest_with_monitoring(engine):
    # Estimate total ticks
    total_ticks = engine.estimate_total_ticks()
    monitor = ProgressMonitor(total_ticks)
    
    # Add progress callback
    engine.set_progress_callback(monitor.update)
    
    # Run backtest
    result = engine.run_backtest()
    
    print("Backtest completed!")
    return result
```

### Memory Usage Monitoring
```python
import psutil
import os

def monitor_memory_usage():
    """Monitor memory usage during backtest"""
    process = psutil.Process(os.getpid())
    
    def log_memory():
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.1f} MB")
    
    return log_memory

# Use during backtest
memory_logger = monitor_memory_usage()

# Log memory before backtest
memory_logger()

# Run backtest
result = engine.run_backtest()

# Log memory after backtest
memory_logger()
```

## Batch Processing

### Multiple Backtest Execution
```python
def run_multiple_backtests(strategies, configs, data_path):
    """Run multiple backtests efficiently"""
    
    results = {}
    
    for i, (strategy_class, strategy_config) in enumerate(strategies):
        for j, backtest_config in enumerate(configs):
            
            test_name = f"Test_{i}_{j}_{strategy_config.name}"
            print(f"Running {test_name}...")
            
            # Create strategy and engine
            strategy = strategy_class(strategy_config, backtest_config)
            engine = BacktestEngine(backtest_config, data_path)
            engine.add_strategy(strategy)
            
            # Run backtest
            result = engine.run_backtest()
            
            # Store result
            results[test_name] = {
                'strategy_config': strategy_config,
                'backtest_config': backtest_config,
                'result': result
            }
    
    return results

# Define test scenarios
strategies = [
    (MovingAverageCrossoverStrategy, ma_config),
    (RSIStrategy, rsi_config),
    (BollingerBandsStrategy, bb_config)
]

configs = [
    config_2024_q1,  # Q1 2024
    config_2024_q2,  # Q2 2024
    config_2024_q3,  # Q3 2024
    config_2024_q4   # Q4 2024
]

# Run all tests
all_results = run_multiple_backtests(strategies, configs, './data')

# Analyze results
for test_name, test_result in all_results.items():
    result = test_result['result']
    print(f"{test_name}: Return={result.total_return:.2%}, "
          f"Sharpe={result.sharpe_ratio:.2f}")
```

## Error Handling and Debugging

### Robust Error Handling
```python
def safe_backtest_execution(engine, max_retries=3):
    """Execute backtest with error handling and retries"""
    
    for attempt in range(max_retries):
        try:
            print(f"Backtest attempt {attempt + 1}")
            
            # Run backtest
            result = engine.run_backtest()
            
            # Validate result
            if result.total_trades == 0:
                print("Warning: No trades generated")
            
            return result
            
        except MemoryError:
            print("Memory error - trying with reduced dataset")
            # Reduce dataset size or use streaming
            engine.config.max_chunk_size = 5000
            
        except Exception as e:
            print(f"Backtest failed: {e}")
            if attempt == max_retries - 1:
                raise
            
            # Wait before retry
            time.sleep(5)
    
    raise Exception("Backtest failed after all retries")

# Use safe execution
try:
    result = safe_backtest_execution(engine)
    print("Backtest completed successfully!")
except Exception as e:
    print(f"Backtest failed: {e}")
```

### Debug Mode
```python
# Enable debug mode for detailed logging
config.debug_mode = True
config.log_level = 'DEBUG'
config.save_debug_data = True

# Run with debug information
engine = BacktestEngine(config, './data')
engine.add_strategy(strategy)

# This will generate detailed logs and save debug data
result = engine.run_backtest()

# Access debug information
debug_info = engine.get_debug_info()
print(f"Total tick processing time: {debug_info['total_processing_time']:.2f}s")
print(f"Average processing time per tick: {debug_info['avg_time_per_tick']:.6f}s")
```

## Best Practices

### Performance Optimization
1. **Use appropriate data ranges**: Don't load unnecessary data
2. **Optimize indicator calculations**: Cache expensive computations
3. **Monitor memory usage**: Use streaming for large datasets
4. **Profile your code**: Identify bottlenecks
5. **Use parallel processing**: When dealing with multiple strategies

### Testing Strategy
1. **Start with small datasets**: Test logic with limited data first
2. **Use multiple time periods**: Test across different market conditions
3. **Validate assumptions**: Check that your strategy logic is correct
4. **Monitor edge cases**: Handle market gaps, holidays, etc.
5. **Use walk-forward analysis**: Test parameter stability

### Risk Management
1. **Set realistic parameters**: Don't over-optimize
2. **Use proper position sizing**: Risk-based sizing
3. **Monitor drawdowns**: Set maximum acceptable drawdown
4. **Diversify strategies**: Don't rely on single approach
5. **Test robustness**: Ensure strategies work in various conditions

---

*Running effective backtests requires careful configuration, proper monitoring, and thorough analysis. Always validate your results and test across multiple market conditions.*