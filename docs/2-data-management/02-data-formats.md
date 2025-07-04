# Data Formats and Structure

## Overview

The backtesting framework supports multiple data formats designed for different use cases, from development and testing to production backtesting with institutional-grade data.

## Supported Data Formats

### 1. CSV Format (Recommended)

The CSV format is the primary format for the framework, offering the best balance of performance, readability, and compatibility.

#### Structure
```csv
timestamp,symbol,bid,ask,bid_volume,ask_volume
2024-01-02T00:00:00.000Z,EURUSD,1.10001,1.10019,1500000,1200000
2024-01-02T00:00:03.123Z,EURUSD,1.10002,1.10020,1600000,1300000
2024-01-02T00:00:07.456Z,EURUSD,1.09999,1.10018,1400000,1300000
```

#### Field Descriptions
- **timestamp**: ISO 8601 format with millisecond precision
- **symbol**: Currency pair in uppercase (e.g., "EURUSD")
- **bid**: Best bid price at the timestamp
- **ask**: Best ask price at the timestamp
- **bid_volume**: Volume available at bid price (optional)
- **ask_volume**: Volume available at ask price (optional)

#### Advantages
- ✅ **Human-readable**: Easy to inspect and debug
- ✅ **Universal compatibility**: Works with any analysis tool
- ✅ **Version control friendly**: Text-based format
- ✅ **Easy preprocessing**: Simple to filter and transform
- ✅ **Debugging support**: Can inspect data visually

#### Performance Characteristics
- **File size**: ~50MB per symbol per day (tick data)
- **Loading speed**: ~2-3 seconds per million ticks
- **Memory usage**: ~2x file size when loaded
- **Compression**: Excellent compression ratios with gzip

### 2. .bi5 Format (Legacy)

Dukascopy's native binary format for ultra-compressed data storage.

#### Structure
Binary format with fixed-size records:
```
[Header: 12 bytes]
[Record 1: 20 bytes] [Record 2: 20 bytes] ... [Record N: 20 bytes]
```

#### Advantages
- ✅ **Ultra-compressed**: Smallest possible file size
- ✅ **Fastest loading**: Optimized binary format
- ✅ **Native format**: Direct from Dukascopy
- ✅ **Precise timing**: Microsecond resolution

#### Disadvantages
- ❌ **Binary format**: Not human-readable
- ❌ **Debugging difficulty**: Requires special tools
- ❌ **Limited compatibility**: Dukascopy-specific
- ❌ **Complex processing**: Requires binary parsing

### 3. JSON Format (Experimental)

For specialized use cases requiring metadata or complex structures.

#### Structure
```json
{
  "metadata": {
    "symbol": "EURUSD",
    "timeframe": "tick",
    "start_date": "2024-01-02T00:00:00.000Z",
    "end_date": "2024-01-02T23:59:59.999Z",
    "record_count": 15755
  },
  "data": [
    {
      "timestamp": "2024-01-02T00:00:00.000Z",
      "bid": 1.10001,
      "ask": 1.10019,
      "bid_volume": 1500000,
      "ask_volume": 1200000
    }
  ]
}
```

#### Use Cases
- Configuration files with embedded data
- API responses with metadata
- Complex data structures with additional fields
- Development and testing scenarios

## Data Quality Standards

### Required Fields
All data formats must include these mandatory fields:
- **timestamp**: Valid ISO 8601 datetime
- **symbol**: Valid currency pair identifier
- **bid**: Valid positive price
- **ask**: Valid positive price greater than bid

### Optional Fields
- **bid_volume**: Volume at bid price (defaults to 0)
- **ask_volume**: Volume at ask price (defaults to 0)
- **spread**: Calculated automatically if not provided
- **mid_price**: Calculated automatically if not provided

### Data Validation Rules

#### Price Validation
```python
# Invalid prices are rejected
assert bid > 0, "Bid price must be positive"
assert ask > 0, "Ask price must be positive"
assert ask > bid, "Ask price must be greater than bid"
assert (ask - bid) / bid < 0.01, "Spread cannot exceed 1% of price"
```

#### Temporal Validation
```python
# Timestamps must be chronologically ordered
assert current_timestamp >= previous_timestamp, "Timestamps must be ordered"
assert current_timestamp - previous_timestamp < timedelta(hours=1), "Gap too large"
```

#### Volume Validation
```python
# Volume constraints (when provided)
assert bid_volume >= 0, "Bid volume cannot be negative"
assert ask_volume >= 0, "Ask volume cannot be negative"
assert bid_volume < 1e9, "Bid volume suspiciously high"
```

## File Naming Conventions

### Standard Naming Pattern
```
{SYMBOL}_{START_DATE}_{END_DATE}_{TIMEFRAME}.{FORMAT}
```

### Examples
```
EURUSD_2024-01-01_2024-01-31_tick.csv
GBPUSD_2024-02-01_2024-02-28_tick.csv
USDJPY_2024-01-01_2024-01-01_h1.csv
```

### Directory Structure
```
data/
├── EURUSD/
│   ├── EURUSD_2024-01-01_2024-01-31_tick.csv
│   ├── EURUSD_2024-02-01_2024-02-28_tick.csv
│   └── metadata.json
├── GBPUSD/
│   └── GBPUSD_2024-01-01_2024-01-31_tick.csv
└── download_summary.json
```

## Data Preprocessing

### Automatic Preprocessing
The framework automatically performs these operations:

#### 1. Data Cleaning
```python
# Remove invalid records
def clean_data(df):
    # Remove negative prices
    df = df[(df['bid'] > 0) & (df['ask'] > 0)]
    
    # Remove inverted spreads
    df = df[df['ask'] > df['bid']]
    
    # Remove extreme spreads (> 100 pips)
    df = df[(df['ask'] - df['bid']) / df['bid'] < 0.01]
    
    return df
```

#### 2. Data Enrichment
```python
# Add computed fields
def enrich_data(df):
    # Calculate spread in pips
    df['spread_pips'] = (df['ask'] - df['bid']) * 10000
    
    # Calculate mid price
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    
    # Add time-based fields
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df
```

#### 3. Data Interpolation
```python
# Fill missing data points
def interpolate_data(df, max_gap_seconds=60):
    # Identify gaps
    gaps = df['timestamp'].diff() > timedelta(seconds=max_gap_seconds)
    
    # Linear interpolation for prices
    df['bid'] = df['bid'].interpolate(method='linear')
    df['ask'] = df['ask'].interpolate(method='linear')
    
    return df
```

## Format Conversion

### CSV to .bi5 Conversion
```python
from data.format_converter import FormatConverter

# Convert CSV to binary format
converter = FormatConverter()
converter.csv_to_bi5(
    input_file='data/EURUSD/EURUSD_2024-01-01_2024-01-31_tick.csv',
    output_file='data/EURUSD/EURUSD_2024-01-01_2024-01-31_tick.bi5'
)
```

### .bi5 to CSV Conversion
```python
# Convert binary to CSV format
converter.bi5_to_csv(
    input_file='data/EURUSD/EURUSD_2024-01-01_2024-01-31_tick.bi5',
    output_file='data/EURUSD/EURUSD_2024-01-01_2024-01-31_tick.csv'
)
```

### Batch Conversion
```bash
# Convert all files in directory
python scripts/convert_formats.py \
  --input-dir data/EURUSD \
  --output-dir data/EURUSD_converted \
  --from-format bi5 \
  --to-format csv
```

## Data Compression

### Compression Options
```python
# Enable compression in config
config = BacktestConfig(
    # ... other settings
    compress_data=True,
    compression_level=6  # 1-9, higher = better compression
)
```

### Manual Compression
```bash
# Compress CSV files
find data/ -name "*.csv" -exec gzip {} \;

# Decompress when needed
find data/ -name "*.csv.gz" -exec gunzip {} \;
```

### Compression Comparison
| Format | Original Size | Compressed Size | Compression Ratio |
|--------|---------------|-----------------|-------------------|
| CSV    | 50MB          | 8MB            | 84% reduction     |
| .bi5   | 12MB          | 10MB           | 17% reduction     |
| JSON   | 85MB          | 12MB           | 86% reduction     |

## Performance Optimization

### Memory Management
```python
# Streaming processing for large files
def process_large_file(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        processed_chunk = clean_data(chunk)
        yield processed_chunk
```

### Caching Strategy
```python
# Enable intelligent caching
config = BacktestConfig(
    # ... other settings
    enable_caching=True,
    cache_size_mb=1000,  # 1GB cache
    cache_ttl_hours=24   # 24 hour cache expiry
)
```

### Parallel Processing
```python
# Parallel data loading
from concurrent.futures import ThreadPoolExecutor

def load_multiple_symbols(symbols, start_date, end_date):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for symbol in symbols:
            future = executor.submit(load_symbol_data, symbol, start_date, end_date)
            futures.append(future)
        
        results = [future.result() for future in futures]
    return results
```

## Quality Assurance

### Data Integrity Checks
```python
# Comprehensive data validation
def validate_data_integrity(df):
    checks = {
        'chronological_order': df['timestamp'].is_monotonic_increasing,
        'no_missing_prices': df[['bid', 'ask']].notna().all().all(),
        'valid_spreads': (df['ask'] > df['bid']).all(),
        'reasonable_prices': ((df['bid'] > 0.01) & (df['ask'] < 100)).all()
    }
    
    return all(checks.values()), checks
```

### Quality Metrics
```python
# Calculate data quality score
def calculate_quality_score(df):
    total_records = len(df)
    valid_records = len(df.dropna())
    
    quality_metrics = {
        'completeness': valid_records / total_records,
        'avg_spread_pips': (df['ask'] - df['bid']).mean() * 10000,
        'max_gap_seconds': df['timestamp'].diff().max().total_seconds(),
        'price_volatility': df['mid_price'].std() / df['mid_price'].mean()
    }
    
    return quality_metrics
```

## Best Practices

### Development Workflow
1. **Start with sample data** for initial development
2. **Use CSV format** for debugging and development
3. **Validate data quality** before running backtests
4. **Compress data** for storage efficiency
5. **Use caching** for frequently accessed data

### Production Workflow
1. **Download data in advance** using batch processing
2. **Validate data integrity** after each download
3. **Use .bi5 format** for maximum performance
4. **Enable compression** for storage optimization
5. **Monitor data quality** continuously

### Storage Optimization
```python
# Optimal storage configuration
storage_config = {
    'format': 'csv',           # or 'bi5' for production
    'compression': 'gzip',     # Enable compression
    'chunk_size': 10000,       # Reasonable chunk size
    'cache_enabled': True,     # Enable caching
    'validation_level': 'full' # Full validation
}
```

## Troubleshooting

### Common Data Issues

#### 1. Invalid Timestamps
```python
# Fix timestamp format issues
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
```

#### 2. Missing Data
```python
# Handle missing data
df = df.dropna(subset=['bid', 'ask'])  # Remove rows with missing prices
df = df.interpolate(method='linear')   # Or interpolate missing values
```

#### 3. Extreme Values
```python
# Remove outliers
Q1 = df['bid'].quantile(0.25)
Q3 = df['bid'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['bid'] < (Q1 - 1.5 * IQR)) | (df['bid'] > (Q3 + 1.5 * IQR)))]
```

### Performance Issues
```python
# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
```

---

*Understanding data formats and quality standards is crucial for reliable backtesting. Choose the appropriate format based on your specific use case and performance requirements.*