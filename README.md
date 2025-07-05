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
# Python dependencies
pip install pandas numpy openpyxl

# Node.js dependencies (for data download)
npm install
```

### Setup
1. Clone or download the backtesting suite
2. Install Python and Node.js dependencies
3. Download data using the built-in data manager (see Data Management section)

### Data Download (New!)
The framework now includes automated data download via dukascopy-node:

```bash
# Quick test download (recommended for first-time users)
python scripts/data_manager.py quick --symbol eurusd --days 2

# Full download with custom parameters
python scripts/data_manager.py download --symbols eurusd,eurjpy --from 2024-01-01 --to 2024-01-31

# Validate downloaded data
python scripts/data_manager.py validate --symbols eurusd --from 2024-01-01 --to 2024-01-31

# List available data
python scripts/data_manager.py list
```

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

## Strategy Development Guide

### Creating Custom Strategies

The framework provides a comprehensive strategy development environment. Here's how to create your own trading strategies:

#### 1. Basic Strategy Structure

```python
from strategy.strategy_interface import TradingStrategy, StrategySignal
from core.data_structures import Tick, OrderSide
from typing import List

class MyCustomStrategy(TradingStrategy):
    def initialize(self) -> None:
        """Initialize strategy state and indicators."""
        # Initialize your indicators
        self.sma_fast = []
        self.sma_slow = []
        self.position_count = 0
        
        # Access strategy parameters
        self.fast_period = self.config.parameters.get('fast_period', 10)
        self.slow_period = self.config.parameters.get('slow_period', 20)
        self.position_size = self.config.parameters.get('position_size', 1.0)
        
        # Access risk management settings
        self.max_position_size = self.config.risk_management.get('max_position_size', 2.0)
        self.stop_loss_pips = self.config.risk_management.get('stop_loss_pips', 30)
        self.take_profit_pips = self.config.risk_management.get('take_profit_pips', 60)
    
    def on_tick(self, tick: Tick) -> List[StrategySignal]:
        """Process each tick and generate trading signals."""
        signals = []
        
        # Update indicators
        self._update_indicators(tick)
        
        # Check for trading opportunities
        if self._should_buy(tick):
            signal = self.create_signal(
                tick=tick,
                signal_type='BUY',
                strength=0.8,  # Signal confidence (0.0 to 1.0)
                quantity=self.position_size,
                stop_loss_pips=self.stop_loss_pips,
                take_profit_pips=self.take_profit_pips
            )
            signals.append(signal)
        
        elif self._should_sell(tick):
            signal = self.create_signal(
                tick=tick,
                signal_type='SELL',
                strength=0.8,
                quantity=self.position_size,
                stop_loss_pips=self.stop_loss_pips,
                take_profit_pips=self.take_profit_pips
            )
            signals.append(signal)
        
        # Check for exit conditions
        if self._should_close_position(tick):
            signal = self.create_signal(
                tick=tick,
                signal_type='CLOSE',
                strength=1.0,  # Full confidence for exits
                quantity=self.position_size
            )
            signals.append(signal)
        
        return signals
    
    def on_order_filled(self, order) -> None:
        """Handle order execution events."""
        self.position_count += 1 if order.side == OrderSide.BUY else -1
        self.logger.info(f"Order filled: {order.side.value} {order.quantity} lots at {order.avg_fill_price}")
    
    def on_position_update(self, position) -> None:
        """Handle position updates."""
        self.logger.info(f"Position updated: {position.symbol} {position.quantity} lots, P&L: {position.unrealized_pnl:.2f}")
    
    def _update_indicators(self, tick: Tick) -> None:
        """Update technical indicators with new tick data."""
        price = tick.mid  # Use mid-price for indicators
        
        # Simple Moving Average calculation
        if len(self.price_history) >= self.fast_period:
            self.sma_fast.append(sum(self.price_history[-self.fast_period:]) / self.fast_period)
        
        if len(self.price_history) >= self.slow_period:
            self.sma_slow.append(sum(self.price_history[-self.slow_period:]) / self.slow_period)
    
    def _should_buy(self, tick: Tick) -> bool:
        """Define buy conditions."""
        if len(self.sma_fast) < 2 or len(self.sma_slow) < 2:
            return False
        
        # Golden cross: fast MA crosses above slow MA
        return (self.sma_fast[-1] > self.sma_slow[-1] and 
                self.sma_fast[-2] <= self.sma_slow[-2] and
                self.position_count == 0)  # No existing position
    
    def _should_sell(self, tick: Tick) -> bool:
        """Define sell conditions."""
        if len(self.sma_fast) < 2 or len(self.sma_slow) < 2:
            return False
        
        # Death cross: fast MA crosses below slow MA
        return (self.sma_fast[-1] < self.sma_slow[-1] and 
                self.sma_fast[-2] >= self.sma_slow[-2] and
                self.position_count == 0)  # No existing position
    
    def _should_close_position(self, tick: Tick) -> bool:
        """Define position exit conditions."""
        # Close on opposite signal
        if self.position_count > 0:  # Long position
            return self._should_sell(tick)
        elif self.position_count < 0:  # Short position
            return self._should_buy(tick)
        return False
```

#### 2. Advanced Strategy Features

##### Using Built-in Technical Indicators

```python
from strategy.technical_indicators import MovingAverage, RSI

class AdvancedStrategy(TradingStrategy):
    def initialize(self) -> None:
        # Use built-in indicators
        self.ma_fast = MovingAverage(period=10)
        self.ma_slow = MovingAverage(period=20)
        self.rsi = RSI(period=14)
        
        # Multiple timeframe support
        self.higher_timeframe_trend = None
    
    def on_tick(self, tick: Tick) -> List[StrategySignal]:
        signals = []
        
        # Update indicators
        mid_price = tick.mid
        self.ma_fast.update(mid_price)
        self.ma_slow.update(mid_price)
        self.rsi.update(mid_price)
        
        # Multi-condition signal generation
        if (self.ma_fast.value > self.ma_slow.value and  # Trend confirmation
            self.rsi.value < 70 and                      # Not overbought
            self.rsi.value > 50):                        # Bullish momentum
            
            signal = self.create_signal(
                tick=tick,
                signal_type='BUY',
                strength=self._calculate_signal_strength(tick),
                quantity=self._calculate_position_size(tick),
                stop_loss_pips=self._calculate_stop_loss(tick),
                take_profit_pips=self._calculate_take_profit(tick)
            )
            signals.append(signal)
        
        return signals
    
    def _calculate_signal_strength(self, tick: Tick) -> float:
        """Calculate signal confidence based on multiple factors."""
        strength = 0.5  # Base strength
        
        # Add strength based on RSI position
        if 30 < self.rsi.value < 70:
            strength += 0.2
        
        # Add strength based on MA separation
        ma_separation = abs(self.ma_fast.value - self.ma_slow.value) / tick.mid
        strength += min(ma_separation * 1000, 0.3)  # Cap at 0.3
        
        return min(strength, 1.0)  # Cap at 1.0
    
    def _calculate_position_size(self, tick: Tick) -> float:
        """Dynamic position sizing based on volatility."""
        base_size = self.config.parameters.get('position_size', 1.0)
        
        # Reduce size in high volatility
        if self.rsi.value > 75 or self.rsi.value < 25:
            return base_size * 0.5
        
        return base_size
```

#### 3. Multi-Symbol Strategies

```python
class MultiSymbolStrategy(TradingStrategy):
    def initialize(self) -> None:
        # Track indicators per symbol
        self.indicators = {}
        self.correlations = {}
        
        for symbol in self.backtest_config.symbols:
            self.indicators[symbol] = {
                'ma_fast': MovingAverage(10),
                'ma_slow': MovingAverage(20),
                'price_history': []
            }
    
    def on_tick(self, tick: Tick) -> List[StrategySignal]:
        signals = []
        symbol = tick.symbol
        
        # Update symbol-specific indicators
        self.indicators[symbol]['ma_fast'].update(tick.mid)
        self.indicators[symbol]['ma_slow'].update(tick.mid)
        self.indicators[symbol]['price_history'].append(tick.mid)
        
        # Symbol-specific logic
        if symbol == 'EURUSD':
            signals.extend(self._process_major_pair(tick))
        elif 'JPY' in symbol:
            signals.extend(self._process_jpy_pair(tick))
        else:
            signals.extend(self._process_minor_pair(tick))
        
        return signals
    
    def _process_major_pair(self, tick: Tick) -> List[StrategySignal]:
        """Special logic for major pairs."""
        # Implementation specific to major pairs
        return []
    
    def _process_jpy_pair(self, tick: Tick) -> List[StrategySignal]:
        """Special logic for JPY pairs (different pip values)."""
        # Implementation specific to JPY pairs
        return []
```

#### 4. Strategy Configuration

```python
# Create strategy configuration
strategy_config = StrategyConfig(
    name="MyAdvancedStrategy",
    description="Multi-indicator strategy with dynamic sizing",
    parameters={
        'fast_period': 10,
        'slow_period': 20,
        'rsi_period': 14,
        'position_size': 1.0,
        'volatility_lookback': 20,
        'correlation_threshold': 0.7
    },
    risk_management={
        'max_position_size': 2.0,
        'stop_loss_pips': 30,
        'take_profit_pips': 60,
        'max_daily_loss': 2000,
        'max_drawdown': 0.15,
        'risk_per_trade': 0.02  # 2% of account per trade
    }
)

# Instantiate and add to backtest
strategy = MyAdvancedStrategy(strategy_config, backtest_config)
engine.add_strategy(strategy)
```

#### 5. Signal Validation and Risk Management

The framework automatically validates all signals:

```python
# Signal strength must be between 0.1 and 1.0
signal = self.create_signal(strength=0.05)  # Will be rejected

# Position size limits are enforced
signal = self.create_signal(quantity=10.0)  # May be reduced based on risk limits

# Stop loss and take profit are automatically set
signal = self.create_signal(
    stop_loss_pips=50,     # Automatic SL calculation
    take_profit_pips=100   # Automatic TP calculation
)
```

#### 6. Testing Your Strategy

```python
# Test your strategy before full backtesting
python scripts/testing/simple_backtest_test.py  # Verify basic functionality

# Run with your strategy
config = BacktestConfig(
    start_date=datetime(2025, 7, 3),
    end_date=datetime(2025, 7, 4),
    symbols=['EURUSD'],
    # ... other config
)

engine = BacktestEngine(config, './data')
engine.add_strategy(MyCustomStrategy(strategy_config, config))
result = engine.run_backtest()

# Analyze results
from analysis.performance_analyzer import PerformanceAnalyzer
analyzer = PerformanceAnalyzer(result)
analyzer.print_summary()
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

## Data Management

### Automated Data Download
The framework includes a powerful data management system with built-in dukascopy-node integration:

#### Supported Currency Pairs
- Major pairs: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- EUR crosses: EURJPY, EURGBP, EURCHF, EURAUD, EURCAD, EURNZD  
- GBP crosses: GBPJPY, GBPCHF, GBPAUD, GBPCAD, GBPNZD
- Other crosses: AUDJPY, AUDCHF, AUDCAD, AUDNZD, CADJPY, CADCHF, NZDJPY, NZDCHF, NZDCAD, CHFJPY

#### Quick Start Data Download
```bash
# Download recent data for testing
python scripts/data_manager.py quick --symbol eurusd --days 5

# Download multiple symbols
python scripts/data_manager.py download \
  --symbols eurusd,eurjpy,gbpnzd \
  --from 2024-01-01 \
  --to 2024-01-31 \
  --timeframe tick
```

#### Data Format Support
The framework auto-detects and supports two data formats:

**CSV Format** (Recommended for Development):
```
data/
├── EURUSD/
│   └── EURUSD_2024-01-01_2024-01-31_tick.csv
├── EURJPY/
│   └── EURJPY_2024-01-01_2024-01-31_tick.csv
└── download_summary.json
```

**Dukascopy .bi5 Format** (Recommended for Production):
```
data/
├── EURUSD/
│   ├── 2024/
│   │   ├── 01/
│   │   │   ├── 01/
│   │   │   │   ├── 00h_ticks.bi5
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

### Data Quality Features
- ✅ Automatic bid/ask price validation and correction
- ✅ Realistic spread validation (0.5-6.0 pips based on pair)
- ✅ Volume data integrity checking
- ✅ Gap interpolation for missing ticks
- ✅ Data completeness reporting

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

## Testing and Validation

### Comprehensive Testing Suite
The framework includes a streamlined testing suite that **automatically downloads required test data**:

```bash
# Run individual tests (auto-downloads data if missing)
python scripts/testing/simple_backtest_test.py          # Basic functionality
python scripts/testing/execution_validation_test.py    # Order execution & position management
python scripts/testing/multi_symbol_test.py            # Multi-currency pair processing
python scripts/testing/performance_analysis_test.py    # Performance reporting & analytics

# Run complete test suite (recommended - auto-downloads all data)
python scripts/testing/master_test_suite.py

# Manually manage test data (optional)
python scripts/testing/test_data_utils.py --download    # Download all test data
python scripts/testing/test_data_utils.py --check       # Check data availability
```

### Test Coverage
✅ **All Tests Pass (100% Success Rate)**
- **Simple Backtest**: Basic framework functionality validation
- **Order Execution Validation**: Order lifecycle, position management, P&L calculation
- **Multi-Symbol Backtesting**: Multiple currency pairs, chronological processing
- **Performance Analysis**: Metrics calculation, report generation, export functionality

### Test Results Location
All test results are saved to the `test_results/` directory:
- `test_suite_report.json` - Comprehensive test execution report
- Individual test reports and exports are also saved here
- No test artifacts are created in the main project directory

### Self-Contained Testing
The testing suite is fully self-contained and requires **no manual setup**:
- ✅ **Automatic Data Download**: Tests automatically download required tick data if missing
- ✅ **Data Validation**: Ensures data quality and completeness before running tests
- ✅ **Clean Results**: All outputs organized in dedicated `test_results/` folder
- ✅ **Resilient**: Works even if existing data is deleted or corrupted

### Validation Results
Recent testing with real market data (July 3-4, 2025):
- **EURUSD**: 72,819 ticks processed, 1 successful trade executed ✅
- **EURJPY**: 21,539 ticks processed, multi-symbol support verified ✅  
- **GBPNZD**: 11,760 ticks processed, JPY vs non-JPY pip calculations validated ✅
- **Execution Model**: Realistic slippage (0.3-0.7 pips), commission calculation, delay simulation ✅

## Troubleshooting

### Common Issues

1. **Data Path Not Found**
   ```
   FileNotFoundError: Data path does not exist
   ```
   Solution: Download data first using `python scripts/data_manager.py quick`

2. **No Data for Symbol**
   ```
   WARNING: No data in specified date range for EURUSD
   ```
   Solution: Check available data with `python scripts/data_manager.py list`

3. **Timezone Issues**
   ```
   Error: can't subtract offset-naive and offset-aware datetimes
   ```
   Solution: This has been fixed in the current version. Update your code.

4. **Memory Issues**
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

### Test Your Installation
```bash
# Quick validation test
python scripts/data_manager.py quick --symbol eurusd --days 1
python scripts/testing/simple_backtest_test.py
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