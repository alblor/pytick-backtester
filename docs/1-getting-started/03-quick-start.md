# Quick Start Guide

## Your First Backtest in 5 Minutes

This guide will walk you through running your first backtest, from data setup to analyzing results.

## Step 1: Setup Sample Data (30 seconds)

First, let's generate realistic sample data for testing:

```bash
# Generate sample tick data for EURUSD, GBPUSD, USDJPY
python scripts/create_sample_data.py
```

**Output:**
```
INFO:__main__:Creating sample data for testing...
INFO:__main__:✅ Saved 15755 records to data/EURUSD/EURUSD_2024-01-02_2024-01-02_tick.csv
INFO:__main__:✅ Saved 15651 records to data/GBPUSD/GBPUSD_2024-01-02_2024-01-02_tick.csv
INFO:__main__:✅ Saved 15657 records to data/USDJPY/USDJPY_2024-01-02_2024-01-02_tick.csv
🎉 Sample data creation completed!
```

## Step 2: Run Your First Backtest (2 minutes)

Create a simple backtest script:

```python
# save as: my_first_backtest.py
from datetime import datetime
from backtester import BacktestEngine, BacktestConfig
from examples.moving_average_strategy import MovingAverageCrossoverStrategy
from strategy.strategy_interface import StrategyConfig

# 1. Configure the backtest
config = BacktestConfig(
    start_date=datetime(2024, 1, 2, 0, 0, 0),
    end_date=datetime(2024, 1, 2, 23, 59, 59),
    symbols=['EURUSD'],
    initial_balance=100000,       # $100,000 starting capital
    spread_markup=0.5,           # Additional 0.5 pips spread
    max_slippage=2.0,           # Maximum 2 pips slippage
    commission_per_lot=7.0       # $7 commission per standard lot
)

# 2. Configure the strategy
strategy_config = StrategyConfig(
    name="MA_Cross_5_20",
    description="5/20 Moving Average Crossover",
    parameters={
        'fast_period': 5,        # Fast moving average period
        'slow_period': 20,       # Slow moving average period
        'min_spread_pips': 1,    # Minimum spread to trade
        'max_spread_pips': 5     # Maximum spread to trade
    },
    risk_management={
        'max_position_size': 1.0,  # Maximum 1 standard lot
        'stop_loss_pips': 30,      # 30 pip stop loss
        'take_profit_pips': 60,    # 60 pip take profit
        'risk_per_trade': 500      # Risk $500 per trade
    }
)

# 3. Create strategy and engine
strategy = MovingAverageCrossoverStrategy(strategy_config, config)
engine = BacktestEngine(config, './data')
engine.add_strategy(strategy)

# 4. Run the backtest
print("🚀 Starting backtest...")
result = engine.run_backtest()

# 5. Display basic results
print(f"\n📊 BACKTEST RESULTS")
print(f"═══════════════════")
print(f"Total Trades: {result.total_trades}")
print(f"Win Rate: {result.win_rate:.1%}")
print(f"Total Return: {result.total_return:.2%}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
```

**Run it:**
```bash
python my_first_backtest.py
```

**Expected Output:**
```
🚀 Starting backtest...
📊 BACKTEST RESULTS
═══════════════════
Total Trades: 15
Win Rate: 53.3%
Total Return: 0.85%
Max Drawdown: 2.1%
Sharpe Ratio: 1.34
```

## Step 3: Analyze Results (2 minutes)

Let's get comprehensive analysis:

```python
# Add this to the end of your script
from analysis.performance_analyzer import PerformanceAnalyzer

# Create analyzer
analyzer = PerformanceAnalyzer(result)

# Print detailed summary
analyzer.print_summary()

# Export detailed reports
analyzer.export_detailed_report('my_backtest_results.json', 'json')
analyzer.export_detailed_report('my_backtest_results.xlsx', 'excel')

print("\n📋 Reports exported:")
print("- my_backtest_results.json (detailed JSON)")
print("- my_backtest_results.xlsx (Excel spreadsheet)")
```

**Sample Analysis Output:**
```
============================================================
BACKTESTING RESULTS SUMMARY
============================================================

PERFORMANCE OVERVIEW:
  Period: 2024-01-02 to 2024-01-02
  Initial Balance: $100,000.00
  Final Balance: $100,847.50
  Total Return: 0.85%
  Max Drawdown: 2.10%

TRADE STATISTICS:
  Total Trades: 15
  Winning Trades: 8
  Losing Trades: 7
  Win Rate: 53.33%
  Profit Factor: 1.34

RISK METRICS:
  Sharpe Ratio: 1.34
  Sortino Ratio: 1.89

EXECUTION ANALYSIS:
  Total Commission: $105.00
  Total Slippage: 18.5 pips
  Avg Execution Delay: 0.247 seconds
============================================================
```

## Step 4: Download Real Data (Optional)

To use real market data instead of sample data:

### Option A: Quick Real Data
```bash
# Download 1 day of EURUSD tick data
node scripts/fetch_single_pair.js
```

### Option B: Custom Data Range
```bash
# Download multiple symbols for a week
node scripts/fetch_data.js \
  --symbols eurusd,gbpusd \
  --from 2024-01-01 \
  --to 2024-01-07 \
  --timeframe tick
```

### Option C: Python Data Manager
```bash
# Using Python interface
python scripts/data_manager.py download \
  --symbols eurusd,gbpusd \
  --from 2024-01-01 \
  --to 2024-01-07
```

## Understanding the Results

### Key Metrics Explained

**Total Return**: Overall percentage gain/loss on your initial capital
- **0.85%** means you made $847.50 on $100,000

**Win Rate**: Percentage of profitable trades
- **53.33%** means 8 out of 15 trades were profitable

**Sharpe Ratio**: Risk-adjusted return metric
- **> 1.0** is generally considered good
- **> 2.0** is excellent

**Max Drawdown**: Largest peak-to-trough decline
- **2.1%** means you never lost more than $2,100 from a peak

**Profit Factor**: Ratio of gross profit to gross loss
- **> 1.0** means profitable overall
- **1.34** means you made $1.34 for every $1.00 lost

### Trade Analysis

The framework tracks every detail:
- **Entry/exit prices** with exact timestamps
- **Slippage costs** on each trade
- **Commission charges** per transaction
- **Maximum favorable/adverse excursion** for each position

## Next Steps

### 1. Experiment with Parameters
Try different moving average periods:

```python
# Test different combinations
parameters_to_test = [
    {'fast_period': 5, 'slow_period': 15},
    {'fast_period': 10, 'slow_period': 30},
    {'fast_period': 8, 'slow_period': 21},
]

for params in parameters_to_test:
    strategy_config.parameters.update(params)
    # ... run backtest and compare results
```

### 2. Add More Symbols
```python
config = BacktestConfig(
    # ... other settings
    symbols=['EURUSD', 'GBPUSD', 'USDJPY'],  # Multiple symbols
)
```

### 3. Extend the Time Period
```python
config = BacktestConfig(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),  # Full month
    # ... other settings
)
```

### 4. Create Your Own Strategy
```python
from strategy.strategy_interface import TradingStrategy

class MyCustomStrategy(TradingStrategy):
    def on_tick(self, tick):
        # Your trading logic here
        signals = []
        
        # Example: Buy when price drops 0.1%
        if tick.bid < self.last_price * 0.999:
            signal = self.create_signal(tick, 'BUY', strength=0.8)
            signals.append(signal)
        
        return signals
```

## Common Patterns

### Multi-Strategy Backtesting
```python
# Test multiple strategies simultaneously
strategy1 = MovingAverageCrossoverStrategy(config1, backtest_config)
strategy2 = RSIStrategy(config2, backtest_config)

engine.add_strategy(strategy1)
engine.add_strategy(strategy2)

result = engine.run_backtest()
```

### Parameter Optimization
```python
best_return = -999
best_params = None

for fast in range(5, 15):
    for slow in range(15, 35):
        # Test each combination
        # Keep track of best performing parameters
```

### Risk Management Testing
```python
# Test different risk levels
for risk_per_trade in [100, 250, 500, 1000]:
    strategy_config.risk_management['risk_per_trade'] = risk_per_trade
    # Run backtest and analyze risk/return profile
```

## Troubleshooting

### No Trades Generated
```python
# Check if strategy conditions are too restrictive
strategy_config.parameters.update({
    'min_spread_pips': 0.5,  # Lower minimum spread
    'max_spread_pips': 10    # Higher maximum spread
})
```

### Memory Issues
```python
# Use smaller datasets or batch processing
config = BacktestConfig(
    # ... other settings
    start_date=datetime(2024, 1, 2, 0, 0, 0),
    end_date=datetime(2024, 1, 2, 12, 0, 0),  # Half day only
)
```

### Data Not Found
```bash
# Verify data exists
python scripts/data_manager.py list

# Generate fresh sample data
python scripts/create_sample_data.py
```

## What's Next?

Now that you've run your first backtest, explore these guides:

1. **[Creating Custom Strategies](../3-guides/01-creating-a-strategy.md)** - Build your own trading logic
2. **[Data Management](../2-data-management/01-fetching-data.md)** - Master data downloading and management
3. **[Performance Analysis](../3-guides/03-analyzing-results.md)** - Deep dive into result analysis
4. **[Core Concepts](../4-core-concepts/01-architecture.md)** - Understand the framework architecture

---

*Congratulations! You've successfully run your first professional-grade backtest. The journey into quantitative trading starts here.*