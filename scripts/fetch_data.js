#!/usr/bin/env node

/**
 * Data fetching script using dukascopy-node for the backtesting framework.
 * Downloads tick data and organizes it in CSV format for the backtesting engine.
 */

const { getHistoricRates } = require('dukascopy-node');
const { Command } = require('commander');
const fs = require('fs-extra');
const path = require('path');

const program = new Command();

// Configure command line interface
program
  .name('fetch-data')
  .description('Download Dukascopy tick data for backtesting')
  .version('1.0.0')
  .option('-s, --symbols <symbols>', 'Comma-separated list of symbols (e.g., eurusd,gbpusd)', 'eurusd')
  .option('-f, --from <date>', 'Start date (YYYY-MM-DD)', '2023-01-01')
  .option('-t, --to <date>', 'End date (YYYY-MM-DD)', '2023-01-31')
  .option('-o, --output <path>', 'Output directory for CSV files', './data')
  .option('-r, --timeframe <timeframe>', 'Timeframe (tick, m1, m5, m15, m30, h1, h4, d1)', 'tick')
  .option('-v, --verbose', 'Verbose logging', false)
  .option('--format <format>', 'Output format (csv, json)', 'csv')
  .option('--batch-size <size>', 'Batch size in days for large downloads', '7')
  .parse();

const options = program.opts();

// Parse and validate options
const symbols = options.symbols.toLowerCase().split(',').map(s => s.trim());
const startDate = new Date(options.from);
const endDate = new Date(options.to);
const outputDir = path.resolve(options.output);
const timeframe = options.timeframe.toLowerCase();
const batchSize = parseInt(options.batchSize);

// Validate dates
if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
  console.error('❌ Invalid date format. Use YYYY-MM-DD format.');
  process.exit(1);
}

if (startDate >= endDate) {
  console.error('❌ Start date must be before end date.');
  process.exit(1);
}

// Validate symbols
const validSymbols = [
  'eurusd', 'gbpusd', 'usdjpy', 'usdchf', 'audusd', 'usdcad', 'nzdusd',
  'eurjpy', 'eurgbp', 'eurchf', 'euraud', 'eurcad', 'eurnzd',
  'gbpjpy', 'gbpchf', 'gbpaud', 'gbpcad', 'gbpnzd',
  'audjpy', 'audchf', 'audcad', 'audnzd',
  'cadjpy', 'cadchf', 'nzdjpy', 'nzdchf', 'nzdcad', 'chfjpy'
];

const invalidSymbols = symbols.filter(s => !validSymbols.includes(s));
if (invalidSymbols.length > 0) {
  console.error(`❌ Invalid symbols: ${invalidSymbols.join(', ')}`);
  console.log(`Valid symbols: ${validSymbols.join(', ')}`);
  process.exit(1);
}

console.log('🚀 Starting Dukascopy data download...');
console.log(`📊 Symbols: ${symbols.join(', ')}`);
console.log(`📅 Period: ${startDate.toISOString().split('T')[0]} to ${endDate.toISOString().split('T')[0]}`);
console.log(`⏱️  Timeframe: ${timeframe}`);
console.log(`📁 Output: ${outputDir}`);

/**
 * Ensure output directory structure exists
 */
async function createDirectoryStructure() {
  await fs.ensureDir(outputDir);
  
  for (const symbol of symbols) {
    const symbolDir = path.join(outputDir, symbol.toUpperCase());
    await fs.ensureDir(symbolDir);
    
    if (options.verbose) {
      console.log(`📁 Created directory: ${symbolDir}`);
    }
  }
}

/**
 * Format tick data to CSV format compatible with backtesting framework
 */
function formatTickDataToCSV(data, symbol) {
  const headers = 'timestamp,symbol,bid,ask,bid_volume,ask_volume\n';
  
  const rows = data.map(tick => {
    const timestamp = new Date(tick.timestamp).toISOString();
    const bid = tick.bid || tick.low || tick.close;
    const ask = tick.ask || tick.high || tick.close;
    const bidVolume = tick.bidVolume || 0;
    const askVolume = tick.askVolume || 0;
    
    return `${timestamp},${symbol.toUpperCase()},${bid},${ask},${bidVolume},${askVolume}`;
  }).join('\n');
  
  return headers + rows;
}

/**
 * Format OHLC data to CSV format
 */
function formatOHLCDataToCSV(data, symbol) {
  const headers = 'timestamp,symbol,open,high,low,close,volume\n';
  
  const rows = data.map(candle => {
    const timestamp = new Date(candle.timestamp).toISOString();
    return `${timestamp},${symbol.toUpperCase()},${candle.open},${candle.high},${candle.low},${candle.close},${candle.volume || 0}`;
  }).join('\n');
  
  return headers + rows;
}

/**
 * Download data for a single symbol and date range
 */
async function downloadSymbolData(symbol, startDate, endDate) {
  try {
    console.log(`📈 Downloading ${symbol.toUpperCase()} from ${startDate.toISOString().split('T')[0]} to ${endDate.toISOString().split('T')[0]}...`);
    
    const config = {
      instrument: symbol,
      dates: {
        from: startDate,
        to: endDate
      },
      timeframe: timeframe,
      priceType: 'bid', // For tick data, we'll get both bid and ask
      utcOffset: 0,
      volumes: true,
      ignoreFlats: false,
      format: 'array'
    };
    
    if (options.verbose) {
      console.log(`🔧 Config: ${JSON.stringify(config, null, 2)}`);
    }
    
    const data = await getHistoricRates(config);
    
    if (!data || data.length === 0) {
      console.warn(`⚠️  No data received for ${symbol.toUpperCase()}`);
      return null;
    }
    
    console.log(`✅ Downloaded ${data.length} ${timeframe} records for ${symbol.toUpperCase()}`);
    
    // Format data based on timeframe
    let csvData;
    if (timeframe === 'tick') {
      csvData = formatTickDataToCSV(data, symbol);
    } else {
      csvData = formatOHLCDataToCSV(data, symbol);
    }
    
    // Save to file
    const fileName = `${symbol.toUpperCase()}_${startDate.toISOString().split('T')[0]}_${endDate.toISOString().split('T')[0]}_${timeframe}.csv`;
    const filePath = path.join(outputDir, symbol.toUpperCase(), fileName);
    
    await fs.writeFile(filePath, csvData);
    console.log(`💾 Saved: ${filePath}`);
    
    return {
      symbol: symbol.toUpperCase(),
      records: data.length,
      file: filePath,
      size: (await fs.stat(filePath)).size
    };
    
  } catch (error) {
    console.error(`❌ Error downloading ${symbol.toUpperCase()}: ${error.message}`);
    if (options.verbose) {
      console.error(error);
    }
    return null;
  }
}

/**
 * Split date range into batches for large downloads
 */
function createDateBatches(startDate, endDate, batchSizeDays) {
  const batches = [];
  let currentStart = new Date(startDate);
  
  while (currentStart < endDate) {
    let currentEnd = new Date(currentStart);
    currentEnd.setDate(currentEnd.getDate() + batchSizeDays - 1);
    
    if (currentEnd > endDate) {
      currentEnd = new Date(endDate);
    }
    
    batches.push({
      start: new Date(currentStart),
      end: new Date(currentEnd)
    });
    
    currentStart.setDate(currentStart.getDate() + batchSizeDays);
  }
  
  return batches;
}

/**
 * Main download function
 */
async function downloadData() {
  try {
    await createDirectoryStructure();
    
    const totalDays = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24));
    const batches = totalDays > batchSize ? createDateBatches(startDate, endDate, batchSize) : [{ start: startDate, end: endDate }];
    
    console.log(`📦 Processing ${batches.length} batch(es) of data...`);
    
    const results = [];
    
    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      console.log(`\n📦 Batch ${batchIndex + 1}/${batches.length}: ${batch.start.toISOString().split('T')[0]} to ${batch.end.toISOString().split('T')[0]}`);
      
      for (const symbol of symbols) {
        const result = await downloadSymbolData(symbol, batch.start, batch.end);
        if (result) {
          results.push(result);
        }
        
        // Add delay between requests to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    // Summary
    console.log('\n🎉 Download Summary:');
    console.log('═'.repeat(60));
    
    let totalRecords = 0;
    let totalSize = 0;
    
    const symbolSummary = {};
    
    results.forEach(result => {
      if (!symbolSummary[result.symbol]) {
        symbolSummary[result.symbol] = { records: 0, size: 0, files: 0 };
      }
      symbolSummary[result.symbol].records += result.records;
      symbolSummary[result.symbol].size += result.size;
      symbolSummary[result.symbol].files += 1;
      
      totalRecords += result.records;
      totalSize += result.size;
    });
    
    Object.entries(symbolSummary).forEach(([symbol, summary]) => {
      console.log(`📊 ${symbol}: ${summary.records.toLocaleString()} records, ${(summary.size / 1024 / 1024).toFixed(2)} MB, ${summary.files} files`);
    });
    
    console.log('─'.repeat(60));
    console.log(`📈 Total: ${totalRecords.toLocaleString()} records, ${(totalSize / 1024 / 1024).toFixed(2)} MB`);
    console.log(`📁 Data saved to: ${outputDir}`);
    
    // Create summary file
    const summaryData = {
      downloadDate: new Date().toISOString(),
      symbols: symbols,
      period: {
        start: startDate.toISOString(),
        end: endDate.toISOString()
      },
      timeframe: timeframe,
      totalRecords: totalRecords,
      totalSizeBytes: totalSize,
      symbolSummary: symbolSummary,
      files: results.map(r => ({ symbol: r.symbol, file: path.basename(r.file), records: r.records, size: r.size }))
    };
    
    const summaryPath = path.join(outputDir, 'download_summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summaryData, null, 2));
    console.log(`📋 Summary saved to: ${summaryPath}`);
    
  } catch (error) {
    console.error('❌ Download failed:', error.message);
    if (options.verbose) {
      console.error(error);
    }
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n⚠️  Download interrupted by user');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n⚠️  Download terminated');
  process.exit(0);
});

// Start download
downloadData().then(() => {
  console.log('\n✅ Download completed successfully!');
  process.exit(0);
}).catch(error => {
  console.error('\n❌ Download failed:', error.message);
  process.exit(1);
});