# Data Fetching Guide

## Overview

The backtesting framework supports multiple data sources and formats, with professional-grade tools for downloading, validating, and managing market data.

## Data Sources

### 1. Dukascopy (Primary Source)
- **Tick-level precision** with sub-second timestamps
- **Variable spreads** from real market conditions
- **High data quality** from institutional source
- **Comprehensive coverage** of major and minor pairs

### 2. Sample Data Generator
- **Realistic synthetic data** for testing and development
- **Configurable parameters** for different market conditions
- **Immediate availability** without external dependencies
- **Perfect for strategy development** and framework testing

## Data Fetching Methods

### Method 1: dukascopy-node (Recommended)

The framework includes professional Node.js scripts for fetching real market data from Dukascopy.

#### Quick Single Pair Download
```bash
# Downloads 1 day of EURUSD tick data
node scripts/fetch_single_pair.js
```

**Configuration (edit the script):**
```javascript
const CONFIG = {
  symbol: 'eurusd',           // Currency pair
  startDate: '2024-01-02',    // Start date
  endDate: '2024-01-02',      // End date  
  timeframe: 'tick',          // Data granularity
  outputDir: './data'         // Output directory
};
```

#### Advanced Multi-Symbol Download
```bash
# Download multiple symbols with custom parameters
node scripts/fetch_data.js \
  --symbols eurusd,gbpusd,usdjpy \
  --from 2024-01-01 \
  --to 2024-01-31 \
  --timeframe tick \
  --batch-size 7 \
  --verbose
```

**Parameters:**
- `--symbols`: Comma-separated list of currency pairs
- `--from/--to`: Date range (YYYY-MM-DD format)
- `--timeframe`: tick, m1, m5, m15, m30, h1, h4, d1
- `--batch-size`: Days per batch (for large downloads)
- `--verbose`: Detailed logging
- `--output`: Custom output directory

#### Available Symbols
Major pairs: eurusd, gbpusd, usdjpy, usdchf, audusd, usdcad, nzdusd
Cross pairs: eurjpy, eurgbp, eurchf, gbpjpy, gbpchf, audjpy, and many more

### Method 2: Python Data Manager

The Python data manager provides a unified interface for all data operations.

#### Quick Commands
```bash
# Quick download for testing (2 days of EURUSD)
python scripts/data_manager.py quick

# Custom download
python scripts/data_manager.py download \
  --symbols eurusd,gbpusd \
  --from 2024-01-01 \
  --to 2024-01-31 \
  --timeframe tick

# Validate data quality
python scripts/data_manager.py validate \
  --symbols eurusd \
  --from 2024-01-01 \
  --to 2024-01-31

# List available data
python scripts/data_manager.py list

# Clean up old data
python scripts/data_manager.py cleanup --days 30
```

#### Data Manager Features
- **Progress monitoring** for large downloads
- **Automatic retry** on network failures
- **Data validation** with quality checks
- **Batch processing** for memory efficiency
- **Comprehensive logging** for troubleshooting

### Method 3: Sample Data Generation

For development and testing without external dependencies:

```bash
# Generate realistic sample data
python scripts/create_sample_data.py
```

**Generated Data:**
- **15,000+ ticks per symbol** for realistic testing
- **Variable spreads** (1-3 pips, realistic for forex)
- **Random walk price movements** with configurable volatility
- **Multiple currency pairs** (EURUSD, GBPUSD, USDJPY)
- **Full day of data** (24 hours of ticks)

## Data Formats

### CSV Format (Recommended)
Human-readable format with excellent debugging capabilities:

```csv
timestamp,symbol,bid,ask,bid_volume,ask_volume
2024-01-02T00:00:00.000Z,EURUSD,1.10001,1.10019,1500000,1200000
2024-01-02T00:00:03.123Z,EURUSD,1.10002,1.10020,1600000,1300000
2024-01-02T00:00:07.456Z,EURUSD,1.09999,1.10018,1400000,1300000
```

**Advantages:**
- ✅ Human-readable and debuggable
- ✅ Universal compatibility
- ✅ Easy data inspection
- ✅ Simple preprocessing
- ✅ Version control friendly

### .bi5 Format (Legacy)
Dukascopy's native binary format for maximum compression:

**Advantages:**
- ✅ Ultra-compressed (smallest file size)
- ✅ Fastest loading times
- ✅ Native Dukascopy format

**Disadvantages:**
- ❌ Binary format (not human-readable)
- ❌ Harder to debug
- ❌ Requires special tools

### Automatic Format Detection
The framework automatically detects and uses the best available format:

```python
# Framework automatically chooses CSV or .bi5
engine = BacktestEngine(config, './data')

# Force specific format if needed
engine = BacktestEngine(config, './data', data_format='csv')
```

## Data Directory Structure

### Standard Layout
```
data/
├── EURUSD/
│   ├── EURUSD_2024-01-01_2024-01-31_tick.csv
│   ├── EURUSD_2024-02-01_2024-02-28_tick.csv
│   └── ...
├── GBPUSD/
│   ├── GBPUSD_2024-01-01_2024-01-31_tick.csv
│   └── ...
├── USDJPY/
│   └── ...
└── download_summary.json
```

### File Naming Convention
- **Format**: `{SYMBOL}_{START_DATE}_{END_DATE}_{TIMEFRAME}.csv`
- **Example**: `EURUSD_2024-01-01_2024-01-31_tick.csv`
- **Date Format**: YYYY-MM-DD
- **Symbol**: Uppercase currency pair

### Download Summary
Each download creates a summary file with metadata:

```json
{
  "downloadDate": "2024-01-02T10:30:00.000Z",
  "symbols": ["EURUSD", "GBPUSD"],
  "period": {
    "start": "2024-01-01T00:00:00.000Z",
    "end": "2024-01-31T23:59:59.000Z"
  },
  "timeframe": "tick",
  "totalRecords": 1250000,
  "totalSizeBytes": 125000000,
  "symbolSummary": {
    "EURUSD": {
      "records": 625000,
      "size": 62500000
    }
  }
}
```

## Data Quality Management

### Automatic Validation
The framework performs comprehensive data quality checks:

#### Price Validation
- **Invalid prices**: Removes ticks with bid ≤ 0 or ask ≤ 0
- **Inverted spreads**: Filters ticks where bid ≥ ask
- **Extreme spreads**: Removes spreads > 100 pips (configurable)

#### Temporal Validation
- **Chronological ordering**: Ensures ticks are time-ordered
- **Duplicate timestamps**: Removes duplicate entries
- **Gap detection**: Identifies and optionally fills data gaps

#### Volume Validation
- **Zero volumes**: Handles missing volume data gracefully
- **Extreme volumes**: Flags unusually high/low volumes

### Data Interpolation
For missing data points, the framework can interpolate:

```python
config = BacktestConfig(
    # ... other settings
    interpolate_missing_ticks=True,
    max_gap_seconds=60.0  # Maximum gap to interpolate
)
```

**Interpolation Methods:**
- **Linear price interpolation**: Smooth price transitions
- **Volume averaging**: Reasonable volume estimates
- **Smart gap filling**: Only fills reasonable gaps

### Quality Reports
Get detailed data quality information:

```bash
# Validate specific dataset
python scripts/data_manager.py validate \
  --symbols eurusd \
  --from 2024-01-01 \
  --to 2024-01-31
```

**Sample Quality Report:**
```
DATA VALIDATION RESULTS
══════════════════════════════════════════════════════════
Overall Status: PASS
Data Format: CSVDataLoader
Total Issues: 0

Per-Symbol Results:

EURUSD: PASS
  Ticks: 1,250,000
  Interpolation: 0.1%
  Avg Spread: 1.8 pips
  Completeness: 99.9%
```

## Advanced Data Operations

### Batch Processing
For large datasets or limited memory:

```bash
# Download in smaller batches
node scripts/fetch_data.js \
  --symbols eurusd \
  --from 2024-01-01 \
  --to 2024-12-31 \
  --batch-size 7  # 7 days per batch
```

### Data Compression
Compress data files to save storage:

```bash
# Compress all CSV files
find data/ -name "*.csv" -exec gzip {} \;

# Framework automatically handles .gz files
```

### Data Merging
Combine multiple time periods:

```python
# The framework automatically merges files chronologically
symbols = ['EURUSD']
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 3, 31)  # Spans multiple files

# Automatically loads and merges all relevant files
data_loader = DataLoaderFactory.create_loader('./data', config)
ticks = list(data_loader.load_symbol_data('EURUSD', start_date, end_date))
```

### Custom Data Sources
Integrate your own data sources:

```python
class CustomDataLoader:
    def load_symbol_data(self, symbol, start_date, end_date):
        # Your custom data loading logic
        for tick_data in your_data_source:
            yield Tick(
                timestamp=tick_data.time,
                symbol=symbol,
                bid=tick_data.bid,
                ask=tick_data.ask
            )
```

## Performance Considerations

### Storage Requirements
- **Tick data**: ~50MB per symbol per day
- **1-minute data**: ~2MB per symbol per day
- **Hourly data**: ~100KB per symbol per day

### Memory Usage
- **CSV loading**: ~2x file size in RAM
- **Large datasets**: Use streaming or batch processing
- **Optimization**: Enable compression for storage

### Network Considerations
- **Download speed**: Depends on Dukascopy server load
- **Retry logic**: Built-in retry for failed downloads
- **Rate limiting**: Automatic delays between requests

## Troubleshooting

### Common Issues

#### 1. No Data Downloaded
```bash
# Check network connectivity
ping historical.dukascopy.com

# Verify Node.js installation
node --version
npm list dukascopy-node
```

#### 2. Invalid Date Ranges
```bash
# Ensure dates are valid and in order
# Avoid weekends for forex data
# Use YYYY-MM-DD format
```

#### 3. File Permission Errors
```bash
# Fix directory permissions
chmod -R 755 data/
chown -R $USER data/
```

#### 4. Memory Issues
```bash
# Use smaller batch sizes
python scripts/data_manager.py download --batch-size 3

# Or generate sample data
python scripts/create_sample_data.py
```

### Data Validation Failures
If validation fails, check:
- **Date ranges**: Ensure data exists for specified dates
- **Symbol names**: Use correct format (lowercase for download, uppercase for processing)
- **File integrity**: Re-download corrupted files
- **Disk space**: Ensure sufficient storage available

## Best Practices

### Development Workflow
1. **Start with sample data** for initial development
2. **Download small real datasets** for validation
3. **Scale up to full datasets** for production testing
4. **Validate data quality** before running long backtests

### Production Workflow
1. **Download data in advance** for backtesting sessions
2. **Validate data quality** after each download
3. **Monitor disk usage** for large datasets
4. **Backup important datasets** before major changes

### Performance Optimization
1. **Use SSD storage** for better I/O performance
2. **Enable compression** for storage efficiency
3. **Batch process** large datasets
4. **Cache frequently used data** in memory

---

*Proper data management is crucial for reliable backtesting results. Take time to understand and validate your data sources.*