#!/usr/bin/env python3

"""
Create sample CSV data for testing the backtesting framework.
Generates realistic tick data for testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_tick_data(symbol: str, start_date: datetime, end_date: datetime, 
                           base_price: float = 1.1000, volatility: float = 0.0001) -> pd.DataFrame:
    """
    Create realistic sample tick data.
    
    Args:
        symbol: Currency pair symbol
        start_date: Start date
        end_date: End date
        base_price: Base price to start from
        volatility: Price volatility
        
    Returns:
        DataFrame with tick data
    """
    # Generate timestamps (every 1-10 seconds during market hours)
    timestamps = []
    current_time = start_date
    
    while current_time <= end_date:
        # Skip weekends
        if current_time.weekday() < 5:  # Monday = 0, Friday = 4
            # Market hours: 00:00 to 23:59 UTC (24/5 market)
            timestamps.append(current_time)
        
        # Random interval between 1-10 seconds
        interval = np.random.uniform(1, 10)
        current_time += timedelta(seconds=interval)
    
    if not timestamps:
        logger.warning("No timestamps generated")
        return pd.DataFrame()
    
    num_ticks = len(timestamps)
    logger.info(f"Generating {num_ticks} ticks for {symbol}")
    
    # Generate price data using random walk
    np.random.seed(42)  # For reproducible results
    
    # Price changes (small random walk)
    price_changes = np.random.normal(0, volatility, num_ticks)
    price_changes = np.cumsum(price_changes)
    
    # Mid prices
    mid_prices = base_price + price_changes
    
    # Spread (realistic forex spreads: 0.5-3 pips)
    pip_value = 0.01 if 'JPY' in symbol else 0.0001
    spread_pips = np.random.uniform(0.5, 3.0, num_ticks)
    spread_price = spread_pips * pip_value
    
    # Bid/Ask prices
    bid_prices = mid_prices - spread_price / 2
    ask_prices = mid_prices + spread_price / 2
    
    # Volumes (random but realistic)
    bid_volumes = np.random.uniform(1000000, 5000000, num_ticks)  # 1-5 million
    ask_volumes = np.random.uniform(1000000, 5000000, num_ticks)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': symbol,
        'bid': bid_prices,
        'ask': ask_prices,
        'bid_volume': bid_volumes,
        'ask_volume': ask_volumes
    })
    
    # Ensure bid < ask
    df['bid'] = np.minimum(df['bid'], df['ask'] - pip_value * 0.1)
    
    return df


def create_sample_data_files():
    """Create sample data files for testing."""
    logger.info("Creating sample data for testing...")
    
    # Create data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    # Test symbols with realistic base prices
    symbols_config = {
        'EURUSD': {'base_price': 1.1000, 'volatility': 0.0001},
        'GBPUSD': {'base_price': 1.2700, 'volatility': 0.0002},
        'USDJPY': {'base_price': 110.00, 'volatility': 0.01}
    }
    
    # Date range for testing (recent dates, avoid weekends)
    start_date = datetime(2024, 1, 2, 0, 0, 0)  # Tuesday
    end_date = datetime(2024, 1, 2, 23, 59, 59)   # Same day
    
    total_files = 0
    total_records = 0
    
    for symbol, config in symbols_config.items():
        logger.info(f"Creating data for {symbol}...")
        
        # Create symbol directory
        symbol_dir = data_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        # Generate data
        df = create_sample_tick_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            base_price=config['base_price'],
            volatility=config['volatility']
        )
        
        if df.empty:
            logger.warning(f"No data generated for {symbol}")
            continue
        
        # Save to CSV
        filename = f"{symbol}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_tick.csv"
        file_path = symbol_dir / filename
        
        df.to_csv(file_path, index=False)
        
        file_size = file_path.stat().st_size
        logger.info(f"✅ Saved {len(df)} records to {file_path} ({file_size/1024:.1f} KB)")
        
        total_files += 1
        total_records += len(df)
        
        # Show sample data
        logger.info(f"📋 Sample data for {symbol}:")
        logger.info(f"   First tick: {df.iloc[0]['timestamp']} - Bid: {df.iloc[0]['bid']:.5f}, Ask: {df.iloc[0]['ask']:.5f}")
        logger.info(f"   Last tick:  {df.iloc[-1]['timestamp']} - Bid: {df.iloc[-1]['bid']:.5f}, Ask: {df.iloc[-1]['ask']:.5f}")
        logger.info(f"   Avg spread: {((df['ask'] - df['bid']) / (0.01 if 'JPY' in symbol else 0.0001)).mean():.1f} pips")
    
    # Create download summary
    summary = {
        'downloadDate': datetime.now().isoformat(),
        'symbols': list(symbols_config.keys()),
        'period': {
            'start': start_date.isoformat(),
            'end': end_date.isoformat()
        },
        'timeframe': 'tick',
        'totalRecords': total_records,
        'totalFiles': total_files,
        'dataType': 'sample_generated',
        'note': 'This is sample data generated for testing purposes'
    }
    
    summary_path = data_dir / 'download_summary.json'
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("📋 Summary:")
    logger.info(f"  Total files created: {total_files}")
    logger.info(f"  Total records: {total_records:,}")
    logger.info(f"  Data directory: {data_dir.absolute()}")
    logger.info(f"  Summary saved to: {summary_path}")
    
    logger.info("\n🎉 Sample data creation completed!")
    logger.info("You can now test the backtesting framework with this data.")


if __name__ == '__main__':
    create_sample_data_files()