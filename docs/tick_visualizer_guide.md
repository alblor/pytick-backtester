# Enhanced Tick Data Visualizer

## Overview

The Enhanced Tick Data Visualizer is a professional tool for analyzing and visualizing massive tick datasets with comprehensive data quality analysis. It provides real-time navigation through millions of ticks while clearly showing data gaps, quality metrics, and market structure.

**Author**: Lorenzo Albanese (alblor)

## Features

### 🔍 **Data Quality Analysis**
- Real-time data quality scoring (0-100%)
- Comprehensive gap detection and analysis
- Real vs interpolated tick identification
- CSV file coverage reporting

### 📊 **Professional Visualization**
- High-performance rendering of massive datasets
- Clear gap visualization with red shaded areas
- Real-time navigation without lag
- Professional trading chart appearance

### 🎯 **Gap Detection Algorithm**
- Intelligent 5-minute threshold for significant gaps
- Weekend and holiday gap identification
- Data provider outage detection
- Comprehensive gap timing and duration analysis

### 🚀 **Performance Optimized**
- Windowed data loading for millions of ticks
- Efficient memory management
- Fast chart updates without clearing axes
- Smooth navigation across multi-year datasets

## Installation and Usage

### Quick Start

```bash
# Navigate to the backtester directory
cd /path/to/backtester

# Run the enhanced visualizer
python scripts/tick_visualizer_enhanced.py EURUSD
```

### Requirements

- Python 3.7+
- matplotlib (for GUI visualization)
- numpy (for data processing)
- Core backtester framework dependencies

### Command Line Usage

```bash
# Basic usage - visualize EURUSD data
python scripts/tick_visualizer_enhanced.py EURUSD

# Visualize other currency pairs
python scripts/tick_visualizer_enhanced.py GBPUSD
python scripts/tick_visualizer_enhanced.py USDJPY
```

## User Interface

### Chart Window

#### **Main Chart Area**
- **Blue lines**: Bid prices (solid = real data, dashed = interpolated)
- **Red lines**: Ask prices (solid = real data, dashed = interpolated)
- **Red shaded areas**: Data gaps with duration annotations
- **Green vertical lines**: Year boundaries
- **Yellow vertical line**: Current position marker

#### **Gap Indicators**
- **Red shaded areas**: Mark periods with missing data
- **Gap annotations**: Show duration (e.g., "GAP 72.3h" for weekend)
- **Gap center markers**: Indicate gap midpoint for reference

### Information Panel

#### **Symbol Information**
```
SYMBOL: EURUSD
POSITION: 132,701 / 265,404
PROGRESS: 50.0%
```

#### **Current Tick Details**
```
CURRENT TICK:
Time: 2025-07-04 14:23:45
Bid: 1.17889
Ask: 1.17893
Spread: 0.40 pips
Status: REAL DATA
```

#### **Data Quality Metrics**
```
DATA QUALITY:
Real ticks: 94.2%
Quality score: 94.2%
Data gaps: 23 (>5min)
```

#### **Data Range Information**
```
DATA RANGE:
From: 2025-07-03 00:00:00
To: 2025-07-04 23:59:58

CSV FILES: 2 files
```

#### **Window Statistics**
```
WINDOW INFO:
Loaded: 5,000 ticks
Interpolated: 142 (2.8%)
Nearby gaps: 1
```

#### **Navigation Controls**
```
NAVIGATION:
Mode: 10 ticks
Left/Right: Navigate
Up/Down: Change mode
Space: Jump to middle
Home/End: Start/End
Esc: Exit
```

## Navigation System

### **Movement Modes**
- **1 tick**: Precise tick-by-tick navigation
- **10 ticks**: Fine-grained movement
- **100 ticks**: Medium jumps
- **1000 ticks**: Large jumps
- **10% jump**: Quick dataset traversal

### **Keyboard Controls**

| Key | Action |
|-----|--------|
| `←` `→` | Navigate backward/forward by current mode |
| `↑` `↓` | Change movement mode |
| `Home` | Jump to dataset beginning |
| `End` | Jump to dataset end |
| `Space` | Jump to dataset middle |
| `Esc` | Exit visualizer |

## Gap Detection Algorithm

### **Detection Criteria**

The visualizer identifies data gaps using a sophisticated algorithm:

1. **Threshold**: 5-minute minimum gap duration
2. **Sequential Analysis**: Examines consecutive tick pairs
3. **Time Calculation**: Measures exact time differences
4. **Classification**: Categorizes significant vs normal intervals

### **Gap Types Detected**

#### **Market Closures**
- **Weekend gaps**: Friday 5pm EST → Sunday 5pm EST (~72 hours)
- **Holiday gaps**: Christmas, New Year, national holidays
- **Session breaks**: Minor gaps between trading sessions

#### **Technical Issues**
- **Data provider outages**: Network or server problems
- **Connectivity issues**: Internet interruptions during collection
- **Processing delays**: Data pipeline bottlenecks

#### **Market Events**
- **Liquidity dry-ups**: During major news or volatility
- **Circuit breakers**: Market halt mechanisms
- **Emergency closures**: Regulatory interventions

### **Gap Information Captured**

For each detected gap:
```python
{
    'start_time': datetime(2025, 7, 5, 21, 59, 58),
    'end_time': datetime(2025, 7, 8, 22, 0, 5),
    'duration_seconds': 259207,
    'duration_minutes': 4320.1,
    'duration_hours': 72.0
}
```

## Data Quality Scoring

### **Quality Metrics**

#### **Real Data Percentage**
- Percentage of ticks that are original market data
- Higher percentages indicate better data quality
- 100% = No interpolated or synthetic data

#### **Quality Score Calculation**
```python
quality_score = (real_ticks / total_ticks) * 100
```

#### **Quality Categories**
- **Excellent (95-100%)**: Professional-grade data
- **Good (85-94%)**: Suitable for most strategies
- **Fair (70-84%)**: Acceptable with caution
- **Poor (<70%)**: Significant data quality issues

### **Gap Impact Analysis**

#### **Gap Statistics**
- **Total gaps**: Count of significant gaps (>5min)
- **Gap duration**: Cumulative missing time
- **Gap distribution**: Temporal pattern analysis

#### **Data Completeness**
```
Total dataset: 7 days
Actual data: 5.2 days
Missing data: 1.8 days (gaps)
Completeness: 74.3%
```

## Performance Characteristics

### **Scalability**

#### **Dataset Sizes Supported**
- **Small datasets**: 1K - 100K ticks (instant loading)
- **Medium datasets**: 100K - 1M ticks (fast loading)
- **Large datasets**: 1M - 10M ticks (windowed loading)
- **Massive datasets**: 10M+ ticks (optimized streaming)

#### **Memory Management**
- **Window size**: 5,000 ticks in memory
- **Display size**: 1,000 ticks on chart
- **Cache optimization**: LRU caching for file access
- **Garbage collection**: Automatic cleanup of old data

### **Response Times**

| Operation | Performance |
|-----------|-------------|
| Initial load | < 5 seconds for 1M ticks |
| Navigation | < 100ms response time |
| Chart update | < 50ms rendering |
| Gap detection | < 2 seconds for 1M ticks |

## Technical Implementation

### **Architecture**

#### **Core Components**
- **EnhancedTickViewer**: Main visualization class
- **Gap Detection Engine**: Intelligent gap analysis
- **Quality Analyzer**: Data quality assessment
- **Navigation System**: Efficient dataset traversal

#### **Data Processing Pipeline**
```
CSV Files → Data Loader → Tick Objects → Quality Analysis → Gap Detection → Visualization
```

### **Configuration Options**

#### **Gap Detection Tuning**
```python
gap_threshold_seconds = 300  # 5-minute threshold
window_size = 5000          # Memory window size
display_size = 1000         # Chart display size
```

#### **Visualization Settings**
```python
matplotlib.rcParams.update({
    'interactive': False,
    'axes.unicode_minus': False,
    'font.size': 9,
    'figure.max_open_warning': 0
})
```

## Best Practices

### **Data Preparation**

1. **CSV File Organization**
   - Use standard naming: `SYMBOL_YYYY-MM-DD_YYYY-MM-DD_tick.csv`
   - Ensure chronological ordering within files
   - Validate data integrity before visualization

2. **Data Quality Checks**
   - Run basic validation on CSV files
   - Check for duplicate timestamps
   - Verify bid/ask spread reasonableness

### **Effective Usage**

1. **Initial Assessment**
   - Review data quality score upon loading
   - Examine gap distribution and patterns
   - Identify problem areas for further investigation

2. **Navigation Strategy**
   - Start with 10% jumps for overview
   - Use smaller increments for detailed analysis
   - Focus on areas around major gaps

3. **Gap Analysis**
   - Investigate unexpected gaps during trading hours
   - Verify weekend/holiday gaps are reasonable
   - Document any data quality issues found

## Troubleshooting

### **Common Issues**

#### **Performance Problems**
```
Issue: Slow loading or navigation
Solution: Check dataset size, reduce window_size if needed
```

#### **Display Issues**
```
Issue: GUI doesn't appear
Solution: Verify matplotlib backend, check display settings
```

#### **Data Loading Errors**
```
Issue: No data found for symbol
Solution: Verify CSV files exist in correct directory structure
```

### **Error Messages**

#### **Data Not Found**
```
Error: "No data found for EURUSD"
Solution: Check data/EURUSD/ directory exists with CSV files
```

#### **Insufficient Data**
```
Error: "Dataset too small for visualization"
Solution: Ensure minimum 100 ticks available
```

## Comparison with Previous Versions

### **Enhanced vs Original Visualizer**

| Feature | Original | Enhanced |
|---------|----------|----------|
| Gap visualization | Interpolated lines | Red shaded areas |
| Data quality | Basic info | Comprehensive scoring |
| Performance | Limited | Optimized for millions |
| Gap detection | Simple | Intelligent algorithm |
| Documentation | Minimal | Professional guide |

### **Key Improvements**

1. **Gap Transparency**: Shows real gaps instead of hiding with interpolation
2. **Professional Quality**: Enterprise-grade visualization and analysis
3. **Performance**: Handles massive datasets efficiently
4. **User Experience**: Intuitive navigation and comprehensive information

## API Reference

### **EnhancedTickViewer Class**

```python
class EnhancedTickViewer:
    def __init__(self, ticks: List[Tick], symbol: str)
    def run(self) -> None
    def _analyze_data_quality(self) -> Dict
    def _detect_gaps(self) -> List[Dict]
    def _get_csv_file_ranges(self) -> List[Dict]
```

### **Main Function**

```python
def main():
    """CLI entry point for enhanced tick visualizer."""
```

## Future Enhancements

### **Planned Features**

1. **Export Capabilities**
   - Gap report generation (PDF/Excel)
   - Data quality assessment reports
   - Screenshot capture functionality

2. **Advanced Analysis**
   - Statistical gap analysis
   - Pattern recognition in gaps
   - Comparative quality analysis

3. **Integration Features**
   - Direct backtesting integration
   - Strategy performance overlay
   - Risk assessment integration

## Support and Maintenance

### **Author Information**
- **Developer**: Lorenzo Albanese (alblor)
- **Project**: Trading Backtester Framework
- **License**: GPL-3.0

### **Contributing**
This tool is part of the larger backtesting framework. Contributions should follow the project's coding standards and include appropriate documentation.

---

*This documentation reflects the current implementation as of the enhanced visualizer release.*