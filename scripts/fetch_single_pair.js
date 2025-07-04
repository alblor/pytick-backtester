#!/usr/bin/env node

/**
 * Simple script to fetch data for a single currency pair.
 * Useful for quick testing and small downloads.
 */

const { getHistoricRates } = require('dukascopy-node');
const fs = require('fs-extra');
const path = require('path');

// Configuration - modify these values as needed
const CONFIG = {
  symbol: 'eurusd',           // Currency pair
  startDate: '2024-01-02',    // Start date (more recent)
  endDate: '2024-01-02',      // End date (same day for testing)
  timeframe: 'm1',            // Start with m1 instead of tick
  outputDir: './data'         // Output directory
};

console.log('🚀 Fetching Dukascopy data for quick testing...');
console.log(`📊 Symbol: ${CONFIG.symbol.toUpperCase()}`);
console.log(`📅 Period: ${CONFIG.startDate} to ${CONFIG.endDate}`);
console.log(`⏱️  Timeframe: ${CONFIG.timeframe}`);

async function fetchSinglePair() {
  try {
    // Ensure output directory exists
    const symbolDir = path.join(CONFIG.outputDir, CONFIG.symbol.toUpperCase());
    await fs.ensureDir(symbolDir);
    
    console.log('📡 Downloading data...');
    
    const startDate = new Date(CONFIG.startDate);
    const endDate = new Date(CONFIG.endDate);
    
    const data = await getHistoricRates({
      instrument: CONFIG.symbol,
      dates: {
        from: startDate,
        to: endDate
      },
      timeframe: CONFIG.timeframe,
      priceType: 'bid',
      utcOffset: 0,
      volumes: true,
      ignoreFlats: false,
      format: 'array'
    });
    
    if (!data || data.length === 0) {
      console.warn('⚠️  No data received');
      return;
    }
    
    console.log(`✅ Downloaded ${data.length} records`);
    
    // Format to CSV
    let csvContent;
    
    if (CONFIG.timeframe === 'tick') {
      // Tick data format
      const headers = 'timestamp,symbol,bid,ask,bid_volume,ask_volume\n';
      const rows = data.map(tick => {
        let timestamp;
        try {
          timestamp = new Date(tick.timestamp).toISOString();
        } catch (e) {
          // If timestamp is invalid, use the tick timestamp directly
          timestamp = tick.timestamp;
        }
        
        const bid = tick.bid || tick.close;
        const ask = tick.ask || tick.close;
        const bidVolume = tick.bidVolume || 0;
        const askVolume = tick.askVolume || 0;
        
        return `${timestamp},${CONFIG.symbol.toUpperCase()},${bid},${ask},${bidVolume},${askVolume}`;
      }).join('\n');
      
      csvContent = headers + rows;
    } else {
      // OHLC data format
      const headers = 'timestamp,symbol,open,high,low,close,volume\n';
      const rows = data.map(candle => {
        let timestamp;
        try {
          timestamp = new Date(candle.timestamp).toISOString();
        } catch (e) {
          timestamp = candle.timestamp;
        }
        return `${timestamp},${CONFIG.symbol.toUpperCase()},${candle.open},${candle.high},${candle.low},${candle.close},${candle.volume || 0}`;
      }).join('\n');
      
      csvContent = headers + rows;
    }
    
    // Save file
    const fileName = `${CONFIG.symbol.toUpperCase()}_${CONFIG.startDate}_${CONFIG.endDate}_${CONFIG.timeframe}.csv`;
    const filePath = path.join(symbolDir, fileName);
    
    await fs.writeFile(filePath, csvContent);
    
    const fileSize = (await fs.stat(filePath)).size;
    
    console.log('💾 File saved successfully!');
    console.log(`📁 Path: ${filePath}`);
    console.log(`📏 Size: ${(fileSize / 1024).toFixed(2)} KB`);
    console.log(`📊 Records: ${data.length}`);
    
    // Show sample data
    console.log('\n📋 Sample data (first 3 rows):');
    const lines = csvContent.split('\n');
    for (let i = 0; i < Math.min(4, lines.length); i++) {
      console.log(`   ${lines[i]}`);
    }
    
    console.log('\n🎉 Quick fetch completed! You can now test the backtesting framework.');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    console.error(error);
  }
}

fetchSinglePair();