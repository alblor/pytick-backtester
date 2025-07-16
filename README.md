# Professional Algorithmic Trading Backtesting Framework

A comprehensive, enterprise-grade backtesting framework for algorithmic trading with tick-by-tick precision simulation. Built for professional forex strategy development and validation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Framework: Professional](https://img.shields.io/badge/framework-enterprise--grade-green.svg)](#)

## 🚀 Key Features

- **Tick-by-Tick Precision**: Real market tick simulation with accurate timing
- **Professional Execution**: Realistic slippage, spreads, and commission modeling
- **Advanced Strategies**: Template-based strategy development framework
- **Comprehensive Analysis**: 50+ performance metrics and risk analysis
- **Dual Data Support**: CSV and Dukascopy .bi5 binary formats
- **Testing Framework**: Built-in validation and testing strategies

## 📦 Quick Installation

```bash
# Install Python dependencies
pip install pandas numpy openpyxl

# Install Node.js dependencies (for data fetching)
npm install

# Download sample data
python scripts/data_manager.py quick --symbol eurusd --days 3
```

## ⚡ Quick Start

```python
from datetime import datetime
from core.data_structures import BacktestConfig
from engine.backtest_engine import BacktestEngine
from strategy.strategy_interface import StrategyConfig
from strategies.examples.moving_average_strategy import MovingAverageCrossoverStrategy

# Configure backtest
config = BacktestConfig(
    start_date=datetime(2025, 7, 3),
    end_date=datetime(2025, 7, 5),
    symbols=['EURUSD'],
    initial_balance=100000.0
)

# Create strategy
strategy_config = StrategyConfig(
    name="MA_Cross_10_20",
    description="Moving Average Crossover Strategy",
    parameters={'fast_period': 10, 'slow_period': 20},
    risk_management={'max_position_size': 1.0, 'stop_loss_pips': 30}
)

# Run backtest
engine = BacktestEngine(config, "./data")
engine.add_strategy(MovingAverageCrossoverStrategy(strategy_config, config))
result = engine.run_backtest()

# Analyze results
from analysis.performance_analyzer import PerformanceAnalyzer
analyzer = PerformanceAnalyzer(result)
analyzer.print_summary()
```

## 🗂️ Project Structure

```
backtester/
├── 📁 strategies/           # Trading strategies
│   ├── examples/           # Example strategy implementations
│   └── testing/            # Testing and validation strategies
├── 📁 examples/            # Usage examples and demo scripts
├── 📁 core/               # Core data structures and base classes
├── 📁 engine/             # Main backtesting engine
├── 📁 execution/          # Order and position management
├── 📁 strategy/           # Strategy framework and interfaces
├── 📁 data/               # Data loading and processing
├── 📁 analysis/           # Performance analysis and reporting
├── 📁 scripts/            # Utility scripts and data management
└── 📁 docs/               # Comprehensive documentation
```

## 📚 Documentation

### 📖 **Getting Started**
- [Introduction](docs/1-getting-started/01-introduction.md) - Framework overview and key features
- [Installation](docs/1-getting-started/02-installation.md) - Setup and prerequisites  
- [Quick Start](docs/1-getting-started/03-quick-start.md) - Your first backtest
- [Architecture](docs/1-getting-started/04-architecture.md) - System design and components

### 🧠 **Strategy Development**
- [Strategy Framework](docs/3-strategy-development/01-framework.md) - Base classes and architecture
- [Creating Strategies](docs/3-strategy-development/02-creating-strategies.md) - Step-by-step development
- [Technical Indicators](docs/3-strategy-development/03-technical-indicators.md) - Built-in and custom indicators
- [Risk Management](docs/3-strategy-development/05-risk-management.md) - Position sizing and risk controls

### 📊 **Testing & Validation**
- [Testing Overview](docs/6-testing-validation/01-overview.md) - Testing philosophy and approach
- [Testing Templates](docs/6-testing-validation/03-testing-templates.md) - Ready-to-use test strategies
- [Validation Tools](docs/6-testing-validation/04-validation-tools.md) - Results verification

### 📋 **Complete Documentation**
For comprehensive documentation, see [docs/README.md](docs/README.md) with complete navigation.

## 🎯 Available Strategies

### Example Strategies (`strategies/examples/`)
- **Moving Average Crossover** - Classic trend-following strategy
- **MA + RSI Combo** - Multi-indicator confirmation system

### Testing Strategies (`strategies/testing/`)
- **MA Crossover Test** - Comprehensive validation template with 1,200+ lines
- **RSI Mean Reversion Test** - Advanced oscillator testing with educational features

### Usage Examples (`examples/`)
- **Complete Backtest Runner** - Full workflow demonstration
- **CSV Workflow Test** - Data processing examples
- **Parameter Optimization** - Systematic parameter testing

## 🔧 Testing Framework

Run the comprehensive testing suite:

```bash
# Run all tests
python scripts/testing/master_test_suite.py

# Run individual tests
python scripts/testing/simple_backtest_test.py
python scripts/testing/execution_validation_test.py
python scripts/testing/multi_symbol_test.py

# Test strategy templates
python strategies/testing/ma_crossover_test_strategy.py
python strategies/testing/rsi_mean_reversion_test_strategy.py
```

## 📈 Performance Metrics

The framework calculates 50+ professional metrics including:

- **Returns**: Total return, CAGR, monthly/annual returns
- **Risk**: Sharpe ratio, Sortino ratio, max drawdown, VaR
- **Trades**: Win rate, profit factor, avg winner/loser
- **Execution**: Slippage costs, commission impact, timing analysis

## 🛠️ Data Management

```bash
# Download real market data
python scripts/data_manager.py download --symbols eurusd,eurjpy --from 2024-01-01 --to 2024-01-31

# Quick test data
python scripts/data_manager.py quick --symbol eurusd --days 5

# Validate data quality
python scripts/data_manager.py validate --symbols eurusd --from 2024-01-01 --to 2024-01-31

# List available data
python scripts/data_manager.py list
```

**Supported Data Formats:**
- **CSV**: Human-readable, debuggable (dukascopy-node output)
- **Dukascopy .bi5**: Binary compressed, ultra-fast processing
- **Auto-detection**: Framework automatically chooses best format

## 🔬 Professional Features

### Enterprise-Grade Architecture
- Event-driven design with clean separation of concerns
- Comprehensive error handling and logging
- Memory-efficient streaming for large datasets
- Professional code quality with extensive documentation

### Realistic Execution Modeling
- Configurable slippage models (linear, random, fixed)
- Realistic spread markup and commission calculation
- Execution delays and market impact simulation
- Margin requirements and leverage management

### Educational Resources
- Comprehensive documentation with 4,000+ lines
- Educational data collection in testing strategies
- Progressive learning structure from basic to advanced
- Real-world examples and best practices

## 🚦 Quick Validation

Verify your installation:

```bash
# Test basic functionality
python -c "from core.data_structures import BacktestConfig; print('✅ Core imports working')"

# Test strategy imports
python -c "from strategies.examples.moving_average_strategy import MovingAverageCrossoverStrategy; print('✅ Strategy imports working')"

# Run quick test
python scripts/testing/simple_backtest_test.py
```

## 📊 Example Results

```
BACKTESTING RESULTS SUMMARY
============================================================

PERFORMANCE OVERVIEW:
  Period: 2025-07-03 to 2025-07-05
  Initial Balance: $100,000.00
  Final Balance: $100,847.23
  Total Return: 0.85%
  Max Drawdown: -0.12%

TRADE STATISTICS:
  Total Trades: 4
  Winning Trades: 3
  Losing Trades: 1
  Win Rate: 75.00%
  Profit Factor: 2.34

EXECUTION ANALYSIS:
  Total Commission: $28.00
  Total Slippage: 1.2 pips
  Avg Execution Delay: 0.142 seconds
============================================================
```

## 🤝 Contributing

This framework is designed for extension:

1. **Add Strategies**: Inherit from `TradingStrategy` base class
2. **Add Indicators**: Extend `TechnicalIndicator` framework  
3. **Add Data Sources**: Implement data loader interfaces
4. **Add Analysis**: Extend `PerformanceAnalyzer` class

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## ⚠️ Disclaimer

This backtesting framework is designed for research and educational purposes. Past performance does not guarantee future results. Always validate strategies with out-of-sample testing before live trading.

---

**Professional Note**: This framework represents institutional-grade standards for algorithmic trading backtesting. Built for accuracy, performance, and educational value, it provides the foundation for serious quantitative trading research and development.

For detailed documentation and advanced features, see the [complete documentation](docs/README.md).