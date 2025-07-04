# Data Validation Tools

## Overview

Data quality is paramount for reliable backtesting results. The framework provides comprehensive validation tools to ensure data integrity, identify issues, and maintain high-quality datasets.

## Quick Validation

### Command Line Validation
```bash
# Validate specific dataset
python scripts/data_manager.py validate \
  --symbols eurusd,gbpusd \
  --from 2024-01-01 \
  --to 2024-01-31

# Validate all available data
python scripts/data_manager.py validate --all

# Quick validation with summary only
python scripts/data_manager.py validate --symbols eurusd --quick
```

### Python API Validation
```python
from data.data_validator import DataValidator
from datetime import datetime

# Create validator
validator = DataValidator()

# Validate specific dataset
result = validator.validate_symbol_data(
    symbol='EURUSD',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    data_path='./data'
)

print(f"Validation Status: {result.status}")
print(f"Issues Found: {len(result.issues)}")
```

## Validation Categories

### 1. Data Completeness

#### Missing Data Detection
```python
def check_data_completeness(df):
    """Check for missing data in dataset"""
    results = {
        'total_records': len(df),
        'missing_timestamps': df['timestamp'].isna().sum(),
        'missing_prices': df[['bid', 'ask']].isna().sum().sum(),
        'missing_volumes': df[['bid_volume', 'ask_volume']].isna().sum().sum()
    }
    
    # Calculate completeness percentage
    results['completeness_pct'] = (
        (results['total_records'] - results['missing_prices']) / 
        results['total_records'] * 100
    )
    
    return results
```

#### Gap Analysis
```python
def analyze_data_gaps(df, expected_interval_seconds=1):
    """Identify gaps in time series data"""
    # Calculate time differences
    time_diffs = df['timestamp'].diff().dt.total_seconds()
    
    # Find gaps larger than expected
    gaps = time_diffs[time_diffs > expected_interval_seconds * 2]
    
    gap_analysis = {
        'total_gaps': len(gaps),
        'largest_gap_seconds': gaps.max() if len(gaps) > 0 else 0,
        'total_missing_time': gaps.sum() if len(gaps) > 0 else 0,
        'gap_locations': gaps.index.tolist() if len(gaps) > 0 else []
    }
    
    return gap_analysis
```

### 2. Price Validation

#### Price Sanity Checks
```python
def validate_prices(df):
    """Comprehensive price validation"""
    issues = []
    
    # Check for negative prices
    negative_bids = df[df['bid'] <= 0]
    if len(negative_bids) > 0:
        issues.append({
            'type': 'negative_bid',
            'count': len(negative_bids),
            'severity': 'high',
            'message': f'Found {len(negative_bids)} records with negative bid prices'
        })
    
    # Check for inverted spreads
    inverted_spreads = df[df['ask'] <= df['bid']]
    if len(inverted_spreads) > 0:
        issues.append({
            'type': 'inverted_spread',
            'count': len(inverted_spreads),
            'severity': 'high',
            'message': f'Found {len(inverted_spreads)} records with inverted spreads'
        })
    
    # Check for extreme spreads
    df['spread_pips'] = (df['ask'] - df['bid']) * 10000
    extreme_spreads = df[df['spread_pips'] > 100]  # > 100 pips
    if len(extreme_spreads) > 0:
        issues.append({
            'type': 'extreme_spread',
            'count': len(extreme_spreads),
            'severity': 'medium',
            'message': f'Found {len(extreme_spreads)} records with extreme spreads (>100 pips)'
        })
    
    # Check for price discontinuities
    price_changes = df['bid'].pct_change().abs()
    large_jumps = price_changes[price_changes > 0.01]  # > 1% change
    if len(large_jumps) > 0:
        issues.append({
            'type': 'price_discontinuity',
            'count': len(large_jumps),
            'severity': 'medium',
            'message': f'Found {len(large_jumps)} records with large price jumps (>1%)'
        })
    
    return issues
```

#### Statistical Price Analysis
```python
def analyze_price_statistics(df):
    """Statistical analysis of price data"""
    stats = {
        'bid_price': {
            'mean': df['bid'].mean(),
            'std': df['bid'].std(),
            'min': df['bid'].min(),
            'max': df['bid'].max(),
            'q25': df['bid'].quantile(0.25),
            'q75': df['bid'].quantile(0.75)
        },
        'ask_price': {
            'mean': df['ask'].mean(),
            'std': df['ask'].std(),
            'min': df['ask'].min(),
            'max': df['ask'].max(),
            'q25': df['ask'].quantile(0.25),
            'q75': df['ask'].quantile(0.75)
        },
        'spread': {
            'mean_pips': (df['ask'] - df['bid']).mean() * 10000,
            'std_pips': (df['ask'] - df['bid']).std() * 10000,
            'min_pips': (df['ask'] - df['bid']).min() * 10000,
            'max_pips': (df['ask'] - df['bid']).max() * 10000
        }
    }
    
    return stats
```

### 3. Temporal Validation

#### Timestamp Validation
```python
def validate_timestamps(df):
    """Validate timestamp data quality"""
    issues = []
    
    # Check for duplicate timestamps
    duplicates = df[df['timestamp'].duplicated()]
    if len(duplicates) > 0:
        issues.append({
            'type': 'duplicate_timestamps',
            'count': len(duplicates),
            'severity': 'high',
            'message': f'Found {len(duplicates)} duplicate timestamps'
        })
    
    # Check chronological order
    if not df['timestamp'].is_monotonic_increasing:
        issues.append({
            'type': 'chronological_order',
            'count': 1,
            'severity': 'high',
            'message': 'Timestamps are not in chronological order'
        })
    
    # Check for future timestamps
    now = datetime.now()
    future_timestamps = df[df['timestamp'] > now]
    if len(future_timestamps) > 0:
        issues.append({
            'type': 'future_timestamps',
            'count': len(future_timestamps),
            'severity': 'medium',
            'message': f'Found {len(future_timestamps)} future timestamps'
        })
    
    return issues
```

#### Time Distribution Analysis
```python
def analyze_time_distribution(df):
    """Analyze temporal distribution of data"""
    # Convert to datetime if needed
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['minute'] = df['timestamp'].dt.minute
    
    distribution = {
        'hourly_distribution': df['hour'].value_counts().sort_index(),
        'daily_distribution': df['day_of_week'].value_counts().sort_index(),
        'minute_distribution': df['minute'].value_counts().sort_index(),
        'total_days': (df['timestamp'].max() - df['timestamp'].min()).days,
        'average_ticks_per_hour': len(df) / 24 if len(df) > 0 else 0
    }
    
    return distribution
```

### 4. Volume Validation

#### Volume Analysis
```python
def validate_volumes(df):
    """Validate volume data"""
    issues = []
    
    # Check for negative volumes
    if 'bid_volume' in df.columns:
        negative_bid_volumes = df[df['bid_volume'] < 0]
        if len(negative_bid_volumes) > 0:
            issues.append({
                'type': 'negative_bid_volume',
                'count': len(negative_bid_volumes),
                'severity': 'medium',
                'message': f'Found {len(negative_bid_volumes)} negative bid volumes'
            })
    
    # Check for extremely high volumes
    if 'bid_volume' in df.columns:
        high_volumes = df[df['bid_volume'] > 1e9]  # > 1 billion
        if len(high_volumes) > 0:
            issues.append({
                'type': 'extreme_volume',
                'count': len(high_volumes),
                'severity': 'low',
                'message': f'Found {len(high_volumes)} extremely high volumes'
            })
    
    return issues
```

## Validation Reports

### Summary Report
```python
def generate_validation_summary(symbol, start_date, end_date, validation_results):
    """Generate comprehensive validation summary"""
    
    summary = f"""
DATA VALIDATION RESULTS
══════════════════════════════════════════════════════════
Symbol: {symbol}
Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Overall Status: {'PASS' if validation_results.is_valid else 'FAIL'}
Data Format: {validation_results.data_format}
Total Issues: {len(validation_results.issues)}

COMPLETENESS ANALYSIS:
  Total Records: {validation_results.total_records:,}
  Missing Data: {validation_results.missing_records:,}
  Completeness: {validation_results.completeness_pct:.1f}%
  Data Gaps: {validation_results.gap_count}
  Largest Gap: {validation_results.largest_gap_minutes:.1f} minutes

QUALITY METRICS:
  Average Spread: {validation_results.avg_spread_pips:.2f} pips
  Price Volatility: {validation_results.price_volatility:.4f}
  Timestamp Accuracy: {validation_results.timestamp_accuracy:.1f}%
  Volume Completeness: {validation_results.volume_completeness:.1f}%

ISSUES SUMMARY:
"""
    
    if validation_results.issues:
        for issue in validation_results.issues:
            summary += f"  {issue['severity'].upper()}: {issue['message']}\n"
    else:
        summary += "  No issues found ✅\n"
    
    summary += "══════════════════════════════════════════════════════════\n"
    
    return summary
```

### Detailed Report
```python
def generate_detailed_report(validation_results):
    """Generate detailed validation report"""
    
    report = {
        'metadata': {
            'validation_date': datetime.now().isoformat(),
            'framework_version': '1.0.0',
            'validator_version': '1.0.0'
        },
        'summary': {
            'symbol': validation_results.symbol,
            'period': {
                'start': validation_results.start_date.isoformat(),
                'end': validation_results.end_date.isoformat()
            },
            'status': validation_results.status,
            'total_issues': len(validation_results.issues)
        },
        'completeness': {
            'total_records': validation_results.total_records,
            'missing_records': validation_results.missing_records,
            'completeness_pct': validation_results.completeness_pct,
            'gap_analysis': validation_results.gap_analysis
        },
        'quality_metrics': {
            'price_statistics': validation_results.price_stats,
            'spread_analysis': validation_results.spread_analysis,
            'temporal_distribution': validation_results.temporal_distribution
        },
        'issues': validation_results.issues,
        'recommendations': validation_results.recommendations
    }
    
    return report
```

## Automated Validation

### Continuous Validation
```python
class ContinuousValidator:
    def __init__(self, data_path, validation_rules):
        self.data_path = data_path
        self.validation_rules = validation_rules
        self.validation_history = []
    
    def validate_new_data(self, symbol, date):
        """Validate newly downloaded data"""
        # Load new data
        file_path = self.get_file_path(symbol, date)
        df = pd.read_csv(file_path)
        
        # Run validation
        validator = DataValidator()
        result = validator.validate_dataframe(df, symbol, date)
        
        # Store results
        self.validation_history.append(result)
        
        # Alert on issues
        if result.has_issues():
            self.send_alert(result)
        
        return result
    
    def send_alert(self, validation_result):
        """Send alert for validation issues"""
        alert_message = f"""
        Data Validation Alert
        Symbol: {validation_result.symbol}
        Issues: {len(validation_result.issues)}
        Severity: {validation_result.highest_severity}
        """
        # Send email, webhook, or log alert
        print(alert_message)
```

### Batch Validation
```bash
# Validate all data in directory
python scripts/batch_validator.py \
  --data-dir ./data \
  --output-report validation_report.json \
  --fix-issues \
  --backup-original

# Validate specific time period
python scripts/batch_validator.py \
  --symbols eurusd,gbpusd \
  --from 2024-01-01 \
  --to 2024-01-31 \
  --parallel-jobs 4
```

## Issue Resolution

### Automatic Fixes
```python
def auto_fix_common_issues(df):
    """Automatically fix common data issues"""
    original_count = len(df)
    fixes_applied = []
    
    # Fix 1: Remove duplicate timestamps
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['timestamp'])
    after_dedup = len(df)
    if before_dedup != after_dedup:
        fixes_applied.append(f"Removed {before_dedup - after_dedup} duplicate timestamps")
    
    # Fix 2: Remove invalid prices
    before_price_fix = len(df)
    df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['ask'] > df['bid'])]
    after_price_fix = len(df)
    if before_price_fix != after_price_fix:
        fixes_applied.append(f"Removed {before_price_fix - after_price_fix} invalid price records")
    
    # Fix 3: Remove extreme spreads
    before_spread_fix = len(df)
    df['spread_pips'] = (df['ask'] - df['bid']) * 10000
    df = df[df['spread_pips'] <= 100]  # Remove spreads > 100 pips
    after_spread_fix = len(df)
    if before_spread_fix != after_spread_fix:
        fixes_applied.append(f"Removed {before_spread_fix - after_spread_fix} extreme spread records")
    
    # Fix 4: Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df, fixes_applied
```

### Manual Fix Guidelines
```python
def suggest_manual_fixes(validation_results):
    """Suggest manual fixes for validation issues"""
    suggestions = []
    
    for issue in validation_results.issues:
        if issue['type'] == 'price_discontinuity':
            suggestions.append({
                'issue': issue,
                'fix': 'Review price jumps manually. Consider removing outliers or checking data source.',
                'command': 'python scripts/price_analysis.py --identify-outliers --symbol {symbol}'
            })
        elif issue['type'] == 'large_gaps':
            suggestions.append({
                'issue': issue,
                'fix': 'Download missing data or enable interpolation.',
                'command': 'python scripts/data_manager.py download --fill-gaps --symbol {symbol}'
            })
        elif issue['type'] == 'extreme_volume':
            suggestions.append({
                'issue': issue,
                'fix': 'Verify volume data accuracy. Consider setting volume limits.',
                'command': 'python scripts/volume_analysis.py --symbol {symbol}'
            })
    
    return suggestions
```

## Performance Monitoring

### Validation Performance
```python
def monitor_validation_performance():
    """Monitor validation performance"""
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Run validation
    validator = DataValidator()
    result = validator.validate_large_dataset(symbol='EURUSD', 
                                            start_date=datetime(2024, 1, 1),
                                            end_date=datetime(2024, 12, 31))
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    performance_metrics = {
        'validation_time_seconds': end_time - start_time,
        'memory_usage_mb': end_memory - start_memory,
        'records_per_second': result.total_records / (end_time - start_time),
        'memory_per_record_bytes': (end_memory - start_memory) * 1024 * 1024 / result.total_records
    }
    
    return performance_metrics
```

## Integration with Backtesting

### Pre-Backtest Validation
```python
def validate_before_backtest(config):
    """Validate data before running backtest"""
    validator = DataValidator()
    
    for symbol in config.symbols:
        result = validator.validate_symbol_data(
            symbol=symbol,
            start_date=config.start_date,
            end_date=config.end_date,
            data_path=config.data_path
        )
        
        if not result.is_valid:
            high_severity_issues = [i for i in result.issues if i['severity'] == 'high']
            if high_severity_issues:
                raise ValidationError(f"High severity issues found in {symbol}: {high_severity_issues}")
            else:
                print(f"Warning: Issues found in {symbol} but continuing with backtest")
    
    print("✅ All data validation checks passed")
```

## Best Practices

### Development Workflow
1. **Validate immediately** after data download
2. **Run quick validation** before each backtest
3. **Store validation results** for audit trail
4. **Monitor validation trends** over time
5. **Automate common fixes** where safe

### Production Workflow
1. **Continuous validation** of new data
2. **Automated alerts** for validation failures
3. **Backup original data** before fixes
4. **Comprehensive logging** of all validation activities
5. **Regular validation reports** for quality monitoring

### Quality Thresholds
```python
VALIDATION_THRESHOLDS = {
    'completeness_min': 99.0,      # Minimum 99% completeness
    'max_spread_pips': 50,         # Maximum 50 pips spread
    'max_gap_minutes': 60,         # Maximum 60 minute gaps
    'max_price_jump_pct': 0.05,    # Maximum 5% price jump
    'min_records_per_hour': 100    # Minimum 100 records per hour
}
```

## Troubleshooting

### Common Validation Issues

#### 1. High Memory Usage
```bash
# Use streaming validation for large datasets
python scripts/data_manager.py validate \
  --symbols eurusd \
  --streaming \
  --chunk-size 50000
```

#### 2. Slow Validation
```bash
# Use parallel validation
python scripts/data_manager.py validate \
  --symbols eurusd,gbpusd,usdjpy \
  --parallel-jobs 4
```

#### 3. False Positives
```python
# Customize validation rules
validation_config = {
    'price_jump_threshold': 0.10,  # More lenient price jump threshold
    'spread_threshold_pips': 100,  # Higher spread threshold
    'enable_interpolation': True    # Enable gap filling
}
```

---

*Data validation is crucial for reliable backtesting. Implement comprehensive validation workflows to ensure data quality and catch issues early in the development process.*