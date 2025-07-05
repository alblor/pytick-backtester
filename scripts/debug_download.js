#!/usr/bin/env node

/**
 * Debug script to understand dukascopy-node data structure
 */

const { getHistoricRates } = require('dukascopy-node');

async function testDownload() {
  try {
    console.log('🔍 Testing dukascopy-node download...');
    
    // Test with both bid and ask price types
    const configs = [
      {
        name: 'BID prices',
        config: {
          instrument: 'eurusd',
          dates: {
            from: new Date('2024-01-15'),
            to: new Date('2024-01-16')
          },
          timeframe: 'tick',
          priceType: 'bid',
          utcOffset: 0,
          volumes: true,
          ignoreFlats: false,
          format: 'array'
        }
      },
      {
        name: 'ASK prices',
        config: {
          instrument: 'eurusd',
          dates: {
            from: new Date('2024-01-15'),
            to: new Date('2024-01-16')
          },
          timeframe: 'tick',
          priceType: 'ask',
          utcOffset: 0,
          volumes: true,
          ignoreFlats: false,
          format: 'array'
        }
      }
    ];
    
    for (const testCase of configs) {
      console.log(`\n📊 Testing ${testCase.name}:`);
      console.log('Config:', JSON.stringify(testCase.config, null, 2));
      
      const data = await getHistoricRates(testCase.config);
      
      if (!data || data.length === 0) {
        console.log('⚠️  No data received');
        continue;
      }
      
      console.log(`✅ Downloaded ${data.length} records`);
      console.log('📋 First 3 records:');
      
      for (let i = 0; i < Math.min(3, data.length); i++) {
        const tick = data[i];
        const timestamp = new Date(tick[0]).toISOString();
        console.log(`  ${i + 1}: [${tick[0]}, ${tick[1]}, ${tick[2]}, ${tick[3]}, ${tick[4]}]`);
        console.log(`     Time: ${timestamp}`);
        console.log(`     Price1: ${tick[1]}, Price2: ${tick[2]}`);
        console.log(`     Volume1: ${tick[3]}, Volume2: ${tick[4]}`);
      }
    }
    
    console.log('\n🔍 Analysis:');
    console.log('- When priceType is "bid", we get bid prices and volumes');
    console.log('- When priceType is "ask", we get ask prices and volumes');
    console.log('- To get both bid AND ask, we need to make TWO requests');
    console.log('- The current implementation in fetch_data.js is INCORRECT');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    console.error('Stack:', error.stack);
  }
}

testDownload();