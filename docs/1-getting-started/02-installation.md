# Installation Guide

## System Requirements

### Python Environment
- **Python 3.8 or higher** (Python 3.9+ recommended)
- **pip** package manager
- **At least 4GB RAM** for reasonable dataset sizes
- **SSD storage recommended** for large tick datasets

### Node.js Environment (Optional but Recommended)
- **Node.js 14 or higher** (Node.js 16+ recommended)
- **npm** package manager
- Required only for real-time data fetching from Dukascopy

### Operating System Support
- **Linux** (Ubuntu 18.04+, CentOS 7+) - Recommended
- **macOS** (10.14+) - Fully supported  
- **Windows** (10/11) - Supported with minor limitations

## Installation Steps

### 1. Clone or Download the Framework

```bash
# If using git
git clone <repository-url> backtester
cd backtester

# Or extract from zip file
unzip backtester.zip
cd backtester
```

### 2. Python Dependencies

#### Install Core Dependencies
```bash
# Install required packages
pip install pandas numpy openpyxl

# For enhanced functionality (optional)
pip install matplotlib seaborn scipy scikit-learn

# Development dependencies (optional)
pip install pytest black flake8
```

#### Alternative: Use requirements.txt
```bash
pip install -r requirements.txt
```

### 3. Node.js Dependencies (Data Fetching)

#### Install Node.js Dependencies
```bash
# Install npm packages
npm install

# This installs:
# - dukascopy-node (data fetching)
# - commander (CLI interface)
# - fs-extra (file operations)
```

#### Verify Installation
```bash
# Test dukascopy-node
node -e "console.log('dukascopy-node:', require('dukascopy-node').getHistoricRates ? 'OK' : 'FAILED')"

# Test data fetching script
node scripts/fetch_single_pair.js --help
```

### 4. Verify Installation

#### Test Python Environment
```bash
# Test core imports
python -c "
import pandas as pd
import numpy as np
from datetime import datetime
print('✅ Python environment ready')
"

# Test framework imports
python -c "
import sys
sys.path.append('.')
from backtester import BacktestEngine, BacktestConfig
print('✅ Backtesting framework ready')
"
```

#### Test Complete Workflow
```bash
# Generate sample data
python scripts/create_sample_data.py

# Run workflow test
python examples/csv_workflow_test.py
```

Expected output should show **100% test success rate**.

## Configuration

### Environment Variables (Optional)

```bash
# Set data directory (default: ./data)
export BACKTEST_DATA_PATH="/path/to/your/data"

# Set log level (default: INFO)
export BACKTEST_LOG_LEVEL="DEBUG"

# Set memory limit for large datasets (default: unlimited)
export BACKTEST_MAX_MEMORY_GB="8"
```

### Data Directory Setup

```bash
# Create data directory structure
mkdir -p data/{EURUSD,GBPUSD,USDJPY}

# Set permissions (Linux/macOS)
chmod -R 755 data/
```

## Troubleshooting

### Common Installation Issues

#### 1. Python Package Conflicts
```bash
# Create virtual environment
python -m venv backtester_env
source backtester_env/bin/activate  # Linux/macOS
# OR
backtester_env\Scripts\activate     # Windows

# Install in clean environment
pip install -r requirements.txt
```

#### 2. Node.js Permission Issues
```bash
# Fix npm permissions (Linux/macOS)
sudo chown -R $(whoami) ~/.npm
npm config set prefix ~/.local

# Or use node version manager
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install node
nvm use node
```

#### 3. pandas/numpy Installation Issues
```bash
# On Ubuntu/Debian
sudo apt-get install python3-dev python3-pip build-essential

# On macOS
xcode-select --install
brew install python

# On Windows
# Use Anaconda distribution (recommended)
```

#### 4. Memory Issues with Large Datasets
```bash
# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Or use batch processing
python scripts/data_manager.py download --batch-size 3
```

### Platform-Specific Notes

#### Linux
- **Recommended platform** for production use
- Best performance with large datasets
- All features fully supported

#### macOS
- Excellent development platform
- All features supported
- May need Xcode command line tools

#### Windows
- Supported but with some limitations
- Use PowerShell or Windows Terminal
- Some path handling differences
- Consider WSL for best experience

## Performance Optimization

### Memory Optimization
```python
# For large datasets, configure chunk size
config = BacktestConfig(
    # ... other settings
    max_chunk_size=10000,  # Process in smaller chunks
    enable_caching=False   # Disable caching for memory-constrained systems
)
```

### Storage Optimization
```bash
# Use compression for data storage
gzip data/**/*.csv

# Or configure automatic compression
export BACKTEST_COMPRESS_DATA=true
```

### CPU Optimization
```python
# Use parallel processing where available
import os
os.environ['NUMBA_NUM_THREADS'] = '4'  # Set thread count
```

## Development Setup

### IDE Configuration

#### Visual Studio Code
Recommended extensions:
- Python (Microsoft)
- Pylance (Microsoft)
- Python Docstring Generator
- GitLens

#### PyCharm
- Professional or Community Edition
- Configure Python interpreter
- Set project root directory

### Pre-commit Hooks
```bash
# Install pre-commit (optional)
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Testing Installation

### Quick Validation
```bash
# 1. Generate test data
python scripts/create_sample_data.py

# 2. Run a simple backtest
python -c "
from datetime import datetime
from backtester import BacktestEngine, BacktestConfig
from examples.moving_average_strategy import MovingAverageCrossoverStrategy
from strategy.strategy_interface import StrategyConfig

config = BacktestConfig(
    start_date=datetime(2024, 1, 2),
    end_date=datetime(2024, 1, 2, 23, 59, 59),
    symbols=['EURUSD'],
    initial_balance=10000
)

strategy_config = StrategyConfig(
    name='Test',
    description='Test strategy',
    parameters={'fast_period': 5, 'slow_period': 10},
    risk_management={'max_position_size': 0.1}
)

strategy = MovingAverageCrossoverStrategy(strategy_config, config)
engine = BacktestEngine(config, './data')
engine.add_strategy(strategy)

result = engine.run_backtest()
print(f'✅ Backtest completed! Processed {len(result.trades)} trades')
"
```

### Full Test Suite
```bash
# Run comprehensive tests
python examples/csv_workflow_test.py

# Expected output: 100% success rate
```

## Next Steps

Once installation is complete:

1. **[Quick Start Guide](03-quick-start.md)** - Run your first backtest
2. **[Data Management](../2-data-management/01-fetching-data.md)** - Download market data
3. **[Creating Strategies](../3-guides/01-creating-a-strategy.md)** - Build custom strategies

## Support

### Getting Help
- Check the **[Troubleshooting Section](#troubleshooting)** above
- Review log files in the project directory
- Ensure all dependencies are correctly installed

### Performance Issues
- Monitor memory usage during large backtests
- Use SSD storage for better I/O performance
- Consider batch processing for very large datasets

---

*Your installation is complete when all tests pass and you can successfully run the sample backtest above.*