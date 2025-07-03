# Professional Algorithmic Trading Backtesting Suite

A comprehensive, professional-grade backtesting framework designed for algorithmic trading with tick-by-tick precision simulation. Built specifically for Dukascopy forex data with realistic execution modeling.

## Features

### 🎯 **Tick-by-Tick Precision**
- Every real tick simulation with precise timestamp handling
- Missing tick interpolation and gap filling
- Data quality validation and integrity checks

### 📊 **Realistic Execution Modeling**
- Configurable spread markup (positive and negative values supported)
- Advanced slippage simulation (linear, random, fixed models)
- Execution delay simulation (random or fixed)
- Commission and margin calculations

### 🎮 **Advanced Order Management**
- Market, Limit, and Stop orders
- Stop Loss and Take Profit management
- Order modification and cancellation
- Realistic fill simulation

### 💼 **Professional Position Management**
- Precise lot sizing (from standard lots to 0.01)
- Real-time P&L calculation
- Margin requirement enforcement
- Risk management controls

### 🧠 **Strategy Framework**
- Template-based strategy development
- Technical indicator library
- Signal generation and validation
- Multiple strategy support

### 📈 **Comprehensive Analysis**
- Detailed performance metrics
- Risk analysis (VaR, Expected Shortfall, Drawdown)
- Trade-by-trade analysis
- Export capabilities (JSON, Excel, CSV)

## Installation

### Prerequisites
```bash
pip install pandas numpy openpyxl
```

### Setup
1. Clone or download the backtesting suite
2. Ensure you have Dukascopy tick data in .bi5 format
3. Update the data path in examples

## Quick Start

### Basic Usage

```python
from datetime import datetime
from backtester import BacktestEngine, BacktestConfig, StrategyConfig
from backtester.examples import MovingAverageCrossoverStrategy

# Create backtesting configuration
config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 1, 31),
    symbols=['EURUSD'],
    initial_balance=100000.0,
    spread_markup=0.5,  # Additional 0.5 pips
    max_slippage=2.0,
    commission_per_lot=7.0
)

# Create strategy
strategy_config = StrategyConfig(
    name="MA_Cross_10_20",
    description="Moving Average Crossover Strategy",
    parameters={'fast_period': 10, 'slow_period': 20},
    risk_management={
        'max_position_size': 2.0,
        'stop_loss_pips': 30,
        'take_profit_pips': 60
    }
)

strategy = MovingAverageCrossoverStrategy(strategy_config, config)

# Run backtest
engine = BacktestEngine(config, "/path/to/dukascopy/data")
engine.add_strategy(strategy)
result = engine.run_backtest()

# Analyze results
from backtester import PerformanceAnalyzer
analyzer = PerformanceAnalyzer(result)
analyzer.print_summary()
```

### Creating Custom Strategies

```python
from backtester.strategy import TradingStrategy, StrategySignal
from backtester.core import Tick, OrderSide

class MyCustomStrategy(TradingStrategy):
    def initialize(self):
        # Initialize indicators and state
        pass
    
    def on_tick(self, tick: Tick) -> List[StrategySignal]:
        signals = []
        
        # Your trading logic here
        if self.should_buy(tick):
            signal = self.create_signal(
                tick=tick,
                signal_type='BUY',
                strength=0.8,
                quantity=1.0
            )
            signals.append(signal)
        
        return signals
    
    def on_order_filled(self, order):
        # Handle order fills
        pass
    
    def on_position_update(self, position):
        # Handle position updates
        pass
```

## Architecture

### Core Components

1. **Data Layer** (`data/`)
   - `DukascopyDataLoader`: Loads and processes .bi5 files
   - Tick interpolation and gap filling
   - Data quality validation

2. **Execution Engine** (`execution/`)
   - `OrderManager`: Handles order lifecycle
   - `PositionManager`: Manages positions and P&L
   - Realistic execution simulation

3. **Strategy Framework** (`strategy/`)
   - `TradingStrategy`: Base strategy class
   - Technical indicators (MA, RSI, etc.)
   - Signal generation framework

4. **Backtesting Engine** (`engine/`)
   - `BacktestEngine`: Coordinates all components
   - Event-driven simulation
   - Progress monitoring

5. **Analysis Tools** (`analysis/`)
   - `PerformanceAnalyzer`: Comprehensive analysis
   - Risk metrics calculation
   - Report generation

### Data Structures

```python
# Core tick data
@dataclass
class Tick:
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    is_interpolated: bool = False

# Order representation
@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # ... execution details

# Position tracking
@dataclass
class Position:
    symbol: str
    side: OrderSide
    quantity: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    # ... risk metrics
```

## Configuration Options

### Backtesting Configuration

```python
config = BacktestConfig(
    # Data settings
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    symbols=['EURUSD', 'GBPUSD', 'USDJPY'],
    
    # Execution settings
    spread_markup=0.5,          # Additional spread in pips
    slippage_model='linear',    # 'linear', 'random', 'fixed'
    max_slippage=2.0,          # Maximum slippage in pips
    execution_delay_min=0.1,    # Minimum delay in seconds
    execution_delay_max=0.5,    # Maximum delay in seconds
    
    # Account settings
    initial_balance=100000.0,
    currency='USD',
    leverage=100.0,
    commission_per_lot=7.0,
    
    # Risk settings
    max_position_size=10.0,
    margin_requirement=0.01,
    margin_call_level=0.5,
    
    # Data quality
    interpolate_missing_ticks=True,
    max_gap_seconds=60.0
)
```

### Strategy Configuration

```python
strategy_config = StrategyConfig(
    name="My_Strategy",
    description="Strategy description",
    parameters={
        'period': 14,
        'threshold': 0.02,
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
```

## Examples

The `examples/` directory contains:

1. **Moving Average Crossover Strategy**
   - Simple MA crossover implementation
   - Demonstrates basic strategy structure

2. **MA + RSI Strategy**
   - Combines MA crossover with RSI confirmation
   - Shows multi-indicator usage

3. **Complete Usage Example**
   - Single and multi-strategy backtests
   - Parameter optimization example
   - Performance analysis workflow

Run examples:
```bash
cd examples/
python run_backtest_example.py
```

## Performance Metrics

The framework calculates comprehensive performance metrics:

### Return Metrics
- Total Return
- Annualized Return
- Monthly Returns
- Compound Annual Growth Rate (CAGR)

### Risk Metrics
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Value at Risk (VaR 95%, 99%)
- Expected Shortfall
- Volatility

### Trade Statistics
- Total Trades
- Win Rate
- Profit Factor
- Average Winner/Loser
- Maximum Consecutive Losses
- Average Trade Duration

### Execution Analysis
- Total Commission Paid
- Average Slippage
- Execution Delay Statistics
- Fill Ratio

## Data Requirements

### Dukascopy Data Format
The framework expects Dukascopy tick data in .bi5 format with the following structure:
```
/data_path/
├── EURUSD/
│   ├── 2023/
│   │   ├── 01/
│   │   │   ├── 01/
│   │   │   │   ├── 00h_ticks.bi5
│   │   │   │   ├── 01h_ticks.bi5
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

### Data Quality Features
- Automatic tick interpolation for gaps
- Spread validation and filtering
- Volume data support
- Missing data handling

## Advanced Features

### Multi-Strategy Backtesting
Run multiple strategies simultaneously:
```python
engine = BacktestEngine(config, data_path)
engine.add_strategy(strategy1)
engine.add_strategy(strategy2)
engine.add_strategy(strategy3)
result = engine.run_backtest()
```

### Parameter Optimization
Systematic parameter testing:
```python
for fast_period in range(5, 15):
    for slow_period in range(15, 30):
        # Create strategy with parameters
        # Run backtest
        # Store results
```

### Real-time Monitoring
Progress callbacks for long backtests:
```python
def progress_callback(progress, status):
    print(f"Progress: {progress:.1%}")

engine.on_progress_update = progress_callback
```

### Custom Indicators
Extend the technical indicator library:
```python
class MACD(TechnicalIndicator):
    def __init__(self, fast=12, slow=26, signal=9):
        # Implementation
        pass
    
    def calculate(self, price):
        # MACD calculation
        pass
```

## Risk Management

### Position Sizing
Multiple position sizing methods:
- Fixed lot size
- Fixed risk amount
- Percentage of account
- Kelly Criterion
- Custom algorithms

### Risk Controls
- Maximum position size limits
- Daily loss limits
- Drawdown limits
- Margin call simulation
- Stop loss enforcement

## Performance Optimization

### Memory Management
- Efficient tick data streaming
- Configurable history buffer sizes
- Memory-conscious indicator calculations

### Speed Optimization
- Vectorized calculations where possible
- Efficient data structures
- Progress monitoring for long backtests

## Export and Reporting

### Export Formats
- JSON: Complete results with metadata
- Excel: Multi-sheet reports with charts
- CSV: Raw trade data for analysis

### Report Contents
- Executive summary
- Detailed trade analysis
- Risk assessment
- Time-based performance
- Symbol-specific analysis
- Monthly/yearly breakdowns

## Troubleshooting

### Common Issues

1. **Data Path Not Found**
   ```
   FileNotFoundError: Data path does not exist
   ```
   Solution: Verify Dukascopy data path and structure

2. **Insufficient Data**
   ```
   ValueError: Insufficient data for EURUSD
   ```
   Solution: Check date range and data availability

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   Solution: Reduce date range or enable data streaming

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

This backtesting suite is designed to be extensible:

1. **Adding New Indicators**
   - Extend `TechnicalIndicator` base class
   - Implement `calculate()` method
   - Add to strategy framework

2. **Creating Strategy Templates**
   - Inherit from `TradingStrategy`
   - Implement required methods
   - Add configuration options

3. **Extending Analysis**
   - Add new metrics to `PerformanceAnalyzer`
   - Create custom report formats
   - Implement additional visualizations

## License

This project is provided as-is for educational and research purposes.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review example implementations
3. Examine log files for detailed error information

---

**Note**: This backtesting framework is designed for research and educational purposes. Past performance does not guarantee future results. Always validate strategies with out-of-sample testing before live trading.