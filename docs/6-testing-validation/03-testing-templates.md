# 🧪 Testing Strategy Templates

## Overview

This section provides comprehensive, ready-to-use testing strategies designed to validate the backtesting framework and serve as learning resources for strategy development. Each template includes complete documentation, expected behaviors, and validation criteria.

## 📊 Template Philosophy

### Dual Purpose Design
Each testing template serves two purposes:
1. **Framework Validation**: Verify backtesting engine functionality
2. **Educational Resource**: Demonstrate best practices and techniques

### Real Market Testing
All templates use actual market data to ensure:
- **Realistic Conditions**: Real spreads, volatility, and market behavior
- **Edge Case Handling**: Market gaps, extreme movements, and unusual conditions
- **Professional Standards**: Institutional-grade testing methodology

## 🎯 Template Categories

### 1. **Basic Strategy Templates**
- Simple, easy-to-understand strategies
- Focus on fundamental concepts
- Minimal complexity for learning

### 2. **Intermediate Strategy Templates**
- Multi-indicator combinations
- More sophisticated logic
- Real-world considerations

### 3. **Advanced Strategy Templates**
- Complex trading systems
- Portfolio-level strategies
- Professional implementations

### 4. **Validation Templates**
- Benchmark strategies for comparison
- Edge case testing strategies
- Performance validation strategies

## 📋 Template Specifications

### Template Structure
Each template includes:
- **Strategy Implementation**: Complete, runnable code
- **Configuration**: Predefined parameters and settings
- **Expected Results**: Benchmark performance metrics
- **Validation Criteria**: Success/failure conditions
- **Documentation**: Detailed explanation and usage

### Testing Data
Templates use standardized test data:
- **Date Range**: July 3-5, 2025 (3 days of recent data)
- **Symbols**: EURUSD, EURJPY, GBPNZD (different pip values)
- **Data Quality**: Validated, clean tick data
- **Automatic Download**: Self-contained data management

---

## 🔄 Template 1: Moving Average Crossover

### Strategy Description
Classic trend-following strategy using fast and slow moving averages.

**Logic:**
- **Buy Signal**: Fast MA crosses above Slow MA
- **Sell Signal**: Fast MA crosses below Slow MA
- **Exit**: Opposite crossover or risk management

### Implementation

```python
"""
Moving Average Crossover Testing Template
========================================

Purpose: Validate basic trend-following strategy logic
Complexity: Basic
Expected Signals: 2-5 per symbol over 3-day period
Risk Level: Medium
"""

from typing import List, Optional
import logging
from datetime import datetime

from core.data_structures import Tick, Order, Position, OrderSide, BacktestConfig
from strategy.strategy_interface import (
    TradingStrategy, StrategyConfig, StrategySignal, MovingAverage
)


class MovingAverageCrossoverTemplate(TradingStrategy):
    """
    Template Strategy: Moving Average Crossover
    
    Educational Focus:
    - Basic trend identification
    - Signal generation and validation
    - Risk management implementation
    - Position tracking
    
    Validation Criteria:
    - Generate 2-5 signals per symbol over test period
    - Maintain win rate between 40-60%
    - Respect risk management limits
    - Handle all market conditions gracefully
    """
    
    def __init__(self, config: StrategyConfig, backtest_config: BacktestConfig):
        super().__init__(config, backtest_config)
        
        # Strategy parameters with validation
        self.fast_period = max(2, config.parameters.get('fast_period', 10))
        self.slow_period = max(self.fast_period + 1, config.parameters.get('slow_period', 20))
        self.min_spread_pips = config.parameters.get('min_spread_pips', 1.0)
        self.max_spread_pips = config.parameters.get('max_spread_pips', 5.0)
        
        # Technical indicators (per symbol)
        self.fast_ma: dict = {}
        self.slow_ma: dict = {}
        
        # Strategy state tracking
        self.last_signals: dict = {}
        self.position_tracking: dict = {}
        self.signal_history: List[StrategySignal] = []
        
        # Validation metrics
        self.signals_generated = 0
        self.signals_executed = 0
        self.spread_rejections = 0
        self.validation_errors = []
        
        logging.info(f"MA Crossover Template initialized: {self.fast_period}/{self.slow_period}")
    
    def initialize(self) -> None:
        """Initialize strategy components."""
        # Initialize indicators for each symbol
        for symbol in self.backtest_config.symbols:
            self.fast_ma[symbol] = MovingAverage(self.fast_period)
            self.slow_ma[symbol] = MovingAverage(self.slow_period)
            self.last_signals[symbol] = None
            self.position_tracking[symbol] = {'position': None, 'entry_time': None}
        
        logging.info("Moving Average Crossover Template initialized")
    
    def on_tick(self, tick: Tick) -> List[StrategySignal]:
        """
        Process tick and generate signals.
        
        Template Testing Points:
        1. Indicator calculation accuracy
        2. Signal generation logic
        3. Risk management compliance
        4. Spread filtering effectiveness
        """
        signals = []
        
        # Add to price history
        self.add_tick_to_history(tick)
        
        # Validate spread conditions
        if not self._is_spread_acceptable(tick):
            self.spread_rejections += 1
            return signals
        
        # Update technical indicators
        mid_price = tick.mid
        fast_ma_value = self.fast_ma[tick.symbol].update(mid_price)
        slow_ma_value = self.slow_ma[tick.symbol].update(mid_price)
        
        # Ensure indicators are ready
        if not (self.fast_ma[tick.symbol].is_ready and self.slow_ma[tick.symbol].is_ready):
            return signals
        
        # Get indicator history for crossover detection
        fast_history = self.fast_ma[tick.symbol].get_history(2)
        slow_history = self.slow_ma[tick.symbol].get_history(2)
        
        if len(fast_history) < 2 or len(slow_history) < 2:
            return signals
        
        # Detect crossovers
        prev_fast, curr_fast = fast_history[-2], fast_history[-1]
        prev_slow, curr_slow = slow_history[-2], slow_history[-1]
        
        # Generate signals based on crossovers
        try:
            # Bullish crossover (Golden Cross)
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                if not self._has_position(tick.symbol, OrderSide.BUY):
                    signal = self._create_buy_signal(tick, curr_fast, curr_slow)
                    if signal:
                        signals.append(signal)
                        self.signals_generated += 1
            
            # Bearish crossover (Death Cross)
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                if not self._has_position(tick.symbol, OrderSide.SELL):
                    signal = self._create_sell_signal(tick, curr_fast, curr_slow)
                    if signal:
                        signals.append(signal)
                        self.signals_generated += 1
            
            # Check for exit conditions
            exit_signal = self._check_exit_conditions(tick, curr_fast, curr_slow)
            if exit_signal:
                signals.append(exit_signal)
                self.signals_generated += 1
        
        except Exception as e:
            error_msg = f"Error generating signal for {tick.symbol}: {str(e)}"
            logging.error(error_msg)
            self.validation_errors.append(error_msg)
        
        return signals
    
    def _is_spread_acceptable(self, tick: Tick) -> bool:
        """
        Validate spread conditions.
        
        Template Testing:
        - Spread calculation accuracy
        - Filtering effectiveness
        - Different currency pair handling
        """
        pip_value = self._get_pip_value(tick.symbol)
        spread_pips = tick.spread / pip_value
        
        is_acceptable = self.min_spread_pips <= spread_pips <= self.max_spread_pips
        
        if not is_acceptable:
            logging.debug(f"Spread rejected for {tick.symbol}: {spread_pips:.2f} pips")
        
        return is_acceptable
    
    def _has_position(self, symbol: str, side: OrderSide) -> bool:
        """
        Check for existing positions.
        
        Template Testing:
        - Position tracking accuracy
        - Direction-specific logic
        - State management
        """
        position = self.position_tracking[symbol]['position']
        
        if not position or position.is_closed:
            return False
        
        if side == OrderSide.BUY:
            return position.is_long
        else:
            return position.is_short
    
    def _create_buy_signal(self, tick: Tick, fast_ma: float, slow_ma: float) -> Optional[StrategySignal]:
        """
        Create validated buy signal.
        
        Template Testing:
        - Signal creation accuracy
        - Risk management integration
        - Position sizing logic
        """
        try:
            # Calculate signal strength based on MA separation
            separation = abs(fast_ma - slow_ma) / tick.mid
            strength = min(0.5 + (separation * 1000), 1.0)  # 0.5 to 1.0 range
            
            # Calculate position size
            risk_amount = self.config.risk_management.get('risk_per_trade', 500)
            position_size = self.calculate_position_size(tick, risk_amount)
            
            # Create signal with comprehensive metadata
            signal = self.create_signal(
                tick=tick,
                signal_type='BUY',
                strength=strength,
                quantity=position_size,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'ma_separation': separation,
                    'entry_reason': 'GOLDEN_CROSS',
                    'validation_template': 'MovingAverageCrossover'
                }
            )
            
            self.last_signals[tick.symbol] = signal
            self.signal_history.append(signal)
            
            logging.info(f"BUY signal created for {tick.symbol}: strength={strength:.2f}, size={position_size:.2f}")
            return signal
            
        except Exception as e:
            error_msg = f"Error creating buy signal: {str(e)}"
            logging.error(error_msg)
            self.validation_errors.append(error_msg)
            return None
    
    def _create_sell_signal(self, tick: Tick, fast_ma: float, slow_ma: float) -> Optional[StrategySignal]:
        """
        Create validated sell signal.
        
        Template Testing:
        - Signal creation accuracy
        - Risk management integration
        - Position sizing logic
        """
        try:
            # Calculate signal strength based on MA separation
            separation = abs(fast_ma - slow_ma) / tick.mid
            strength = min(0.5 + (separation * 1000), 1.0)  # 0.5 to 1.0 range
            
            # Calculate position size
            risk_amount = self.config.risk_management.get('risk_per_trade', 500)
            position_size = self.calculate_position_size(tick, risk_amount)
            
            # Create signal with comprehensive metadata
            signal = self.create_signal(
                tick=tick,
                signal_type='SELL',
                strength=strength,
                quantity=position_size,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'ma_separation': separation,
                    'entry_reason': 'DEATH_CROSS',
                    'validation_template': 'MovingAverageCrossover'
                }
            )
            
            self.last_signals[tick.symbol] = signal
            self.signal_history.append(signal)
            
            logging.info(f"SELL signal created for {tick.symbol}: strength={strength:.2f}, size={position_size:.2f}")
            return signal
            
        except Exception as e:
            error_msg = f"Error creating sell signal: {str(e)}"
            logging.error(error_msg)
            self.validation_errors.append(error_msg)
            return None
    
    def _check_exit_conditions(self, tick: Tick, fast_ma: float, slow_ma: float) -> Optional[StrategySignal]:
        """
        Check for position exit conditions.
        
        Template Testing:
        - Exit logic accuracy
        - Position management
        - Risk control effectiveness
        """
        position = self.position_tracking[tick.symbol]['position']
        
        if not position or position.is_closed:
            return None
        
        # Get MA history for exit detection
        fast_history = self.fast_ma[tick.symbol].get_history(2)
        slow_history = self.slow_ma[tick.symbol].get_history(2)
        
        if len(fast_history) < 2 or len(slow_history) < 2:
            return None
        
        prev_fast, curr_fast = fast_history[-2], fast_history[-1]
        prev_slow, curr_slow = slow_history[-2], slow_history[-1]
        
        should_exit = False
        exit_reason = ""
        
        # Exit long position on bearish crossover
        if position.is_long and prev_fast >= prev_slow and curr_fast < curr_slow:
            should_exit = True
            exit_reason = "DEATH_CROSS_EXIT"
        
        # Exit short position on bullish crossover
        elif position.is_short and prev_fast <= prev_slow and curr_fast > curr_slow:
            should_exit = True
            exit_reason = "GOLDEN_CROSS_EXIT"
        
        if should_exit:
            try:
                signal = StrategySignal(
                    timestamp=tick.timestamp,
                    symbol=tick.symbol,
                    signal_type='CLOSE',
                    strength=1.0,
                    price=tick.mid,
                    quantity=abs(position.quantity),
                    metadata={
                        'exit_reason': exit_reason,
                        'position_duration': (tick.timestamp - self.position_tracking[tick.symbol]['entry_time']).total_seconds(),
                        'validation_template': 'MovingAverageCrossover'
                    }
                )
                
                logging.info(f"EXIT signal created for {tick.symbol}: {exit_reason}")
                return signal
                
            except Exception as e:
                error_msg = f"Error creating exit signal: {str(e)}"
                logging.error(error_msg)
                self.validation_errors.append(error_msg)
        
        return None
    
    def on_order_filled(self, order: Order) -> None:
        """
        Handle order execution events.
        
        Template Testing:
        - Order execution tracking
        - Position state updates
        - Performance metrics
        """
        try:
            self.signals_executed += 1
            logging.info(f"Order filled: {order.side.value} {order.filled_quantity} lots "
                        f"of {order.symbol} at {order.avg_fill_price:.5f}")
            
            # Track execution for validation
            if hasattr(self, 'execution_tracking'):
                self.execution_tracking.append({
                    'timestamp': order.filled_at,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.filled_quantity,
                    'price': order.avg_fill_price,
                    'commission': order.commission,
                    'slippage': order.slippage
                })
            
        except Exception as e:
            error_msg = f"Error handling order fill: {str(e)}"
            logging.error(error_msg)
            self.validation_errors.append(error_msg)
    
    def on_position_update(self, position: Position) -> None:
        """
        Handle position updates.
        
        Template Testing:
        - Position tracking accuracy
        - P&L calculation
        - Risk management compliance
        """
        try:
            # Update position tracking
            self.position_tracking[position.symbol]['position'] = position
            
            if not position.is_closed:
                # Position opened or modified
                if self.position_tracking[position.symbol]['entry_time'] is None:
                    self.position_tracking[position.symbol]['entry_time'] = position.opened_at
                
                # Update performance tracking
                if position.total_pnl > 0:
                    self.winning_signals += 1
                elif position.total_pnl < 0:
                    self.losing_signals += 1
                
                logging.info(f"Position updated: {position.symbol} "
                           f"{position.quantity} lots, P&L: {position.total_pnl:.2f}")
            else:
                # Position closed
                self.position_tracking[position.symbol]['entry_time'] = None
                
                logging.info(f"Position closed: {position.symbol} "
                           f"Final P&L: {position.total_pnl:.2f}")
        
        except Exception as e:
            error_msg = f"Error handling position update: {str(e)}"
            logging.error(error_msg)
            self.validation_errors.append(error_msg)
    
    def get_validation_metrics(self) -> dict:
        """
        Get comprehensive validation metrics.
        
        Template Testing:
        - Performance measurement
        - Error tracking
        - Success criteria validation
        """
        base_metrics = self.get_performance_metrics()
        
        validation_metrics = {
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'execution_rate': self.signals_executed / max(self.signals_generated, 1),
            'spread_rejections': self.spread_rejections,
            'validation_errors': len(self.validation_errors),
            'error_messages': self.validation_errors[-5:],  # Last 5 errors
            'signal_history_count': len(self.signal_history),
            'template_version': '1.0',
            'test_criteria': self._get_test_criteria()
        }
        
        return {**base_metrics, **validation_metrics}
    
    def _get_test_criteria(self) -> dict:
        """Define success criteria for template validation."""
        return {
            'min_signals_per_symbol': 1,
            'max_signals_per_symbol': 10,
            'min_execution_rate': 0.8,
            'max_validation_errors': 2,
            'expected_win_rate_range': (0.3, 0.7),
            'max_spread_rejection_rate': 0.3
        }
    
    def validate_template_results(self, results) -> dict:
        """
        Validate template execution results.
        
        Returns comprehensive validation report.
        """
        metrics = self.get_validation_metrics()
        criteria = metrics['test_criteria']
        
        validation_report = {
            'template_name': 'MovingAverageCrossover',
            'validation_timestamp': datetime.now().isoformat(),
            'overall_success': True,
            'detailed_results': {}
        }
        
        # Check individual criteria
        tests = [
            ('signals_generated', metrics['signals_generated'] >= criteria['min_signals_per_symbol']),
            ('execution_rate', metrics['execution_rate'] >= criteria['min_execution_rate']),
            ('validation_errors', metrics['validation_errors'] <= criteria['max_validation_errors']),
            ('win_rate_range', criteria['expected_win_rate_range'][0] <= metrics['win_rate'] <= criteria['expected_win_rate_range'][1]),
        ]
        
        for test_name, passed in tests:
            validation_report['detailed_results'][test_name] = {
                'passed': passed,
                'actual_value': metrics.get(test_name.split('_')[0], 'N/A'),
                'expected_range': criteria.get(f"expected_{test_name}", 'N/A')
            }
            
            if not passed:
                validation_report['overall_success'] = False
        
        return validation_report


def create_ma_crossover_template_config() -> StrategyConfig:
    """
    Create standardized configuration for MA Crossover template.
    
    Template Testing:
    - Configuration validation
    - Parameter range testing
    - Risk management compliance
    """
    return StrategyConfig(
        name="MA_Crossover_Template",
        description="Moving Average Crossover Testing Template - Educational and Validation",
        parameters={
            'fast_period': 10,
            'slow_period': 20,
            'min_spread_pips': 1.0,
            'max_spread_pips': 4.0,
            'risk_per_trade': 500,  # Risk $500 per trade
            'position_size_method': 'fixed_risk'
        },
        risk_management={
            'max_position_size': 1.0,
            'stop_loss_pips': 30,
            'take_profit_pips': 60,
            'max_daily_loss': 2000,
            'max_drawdown': 0.2,
            'risk_per_trade': 500
        }
    )


def run_ma_crossover_template_test(data_path: str = "./data") -> dict:
    """
    Run complete MA Crossover template test.
    
    Returns:
        Comprehensive test results and validation report
    """
    from datetime import datetime
    from engine.backtest_engine import BacktestEngine
    from analysis.performance_analyzer import PerformanceAnalyzer
    
    # Test configuration
    backtest_config = BacktestConfig(
        start_date=datetime(2025, 7, 3),
        end_date=datetime(2025, 7, 5),
        symbols=['EURUSD', 'EURJPY', 'GBPNZD'],
        
        # Realistic execution settings
        spread_markup=0.5,
        slippage_model='linear',
        max_slippage=1.5,
        execution_delay_min=0.1,
        execution_delay_max=0.3,
        
        # Account settings
        initial_balance=100000.0,
        leverage=100.0,
        commission_per_lot=7.0,
        
        # Risk settings
        max_position_size=2.0,
        margin_requirement=0.01,
        
        # Data quality
        interpolate_missing_ticks=True,
        max_gap_seconds=60.0
    )
    
    # Create strategy
    strategy_config = create_ma_crossover_template_config()
    strategy = MovingAverageCrossoverTemplate(strategy_config, backtest_config)
    
    # Run backtest
    engine = BacktestEngine(backtest_config, data_path)
    engine.add_strategy(strategy)
    
    try:
        result = engine.run_backtest()
        
        # Analyze results
        analyzer = PerformanceAnalyzer(result)
        
        # Get validation metrics
        validation_metrics = strategy.get_validation_metrics()
        validation_report = strategy.validate_template_results(result)
        
        return {
            'backtest_result': result,
            'performance_analysis': analyzer.generate_summary_report(),
            'validation_metrics': validation_metrics,
            'validation_report': validation_report,
            'strategy_config': strategy_config,
            'test_successful': validation_report['overall_success']
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'validation_metrics': strategy.get_validation_metrics(),
            'test_successful': False
        }


# Example usage
if __name__ == "__main__":
    # Run the template test
    test_results = run_ma_crossover_template_test()
    
    if test_results['test_successful']:
        print("✅ MA Crossover Template Test: PASSED")
        print(f"Signals Generated: {test_results['validation_metrics']['signals_generated']}")
        print(f"Execution Rate: {test_results['validation_metrics']['execution_rate']:.2%}")
        print(f"Win Rate: {test_results['validation_metrics']['win_rate']:.2%}")
    else:
        print("❌ MA Crossover Template Test: FAILED")
        print(f"Errors: {test_results['validation_metrics']['validation_errors']}")
```

### Expected Results

**Performance Metrics:**
- **Signals Generated**: 3-6 per symbol over 3-day period
- **Win Rate**: 40-60%
- **Execution Rate**: >80%
- **Max Drawdown**: <15%
- **Profit Factor**: 0.8-1.5

**Validation Criteria:**
- All signals properly validated
- Risk management rules enforced
- No critical errors in execution
- Position tracking accurate

### Usage Instructions

1. **Run Template Test**:
   ```bash
   python docs/6-testing-validation/templates/ma_crossover_template.py
   ```

2. **Analyze Results**:
   - Check validation report for pass/fail status
   - Review performance metrics
   - Examine signal generation patterns

3. **Customize for Learning**:
   - Modify parameters (fast_period, slow_period)
   - Adjust risk management settings
   - Test different market conditions

---

## 🎯 Template 2: RSI Mean Reversion

### Strategy Description
Oscillator-based strategy using RSI for mean reversion signals.

**Logic:**
- **Buy Signal**: RSI < 30 (oversold) with confirmation
- **Sell Signal**: RSI > 70 (overbought) with confirmation
- **Exit**: RSI returns to neutral zone (30-70)

### Key Features
- **Oscillator Usage**: RSI calculation and interpretation
- **Mean Reversion Logic**: Counter-trend trading
- **Confirmation Signals**: Multiple condition validation
- **Dynamic Exit**: RSI-based exit conditions

### Template Structure
```python
class RSIMeanReversionTemplate(TradingStrategy):
    """
    Template Strategy: RSI Mean Reversion
    
    Educational Focus:
    - Oscillator indicators
    - Mean reversion concepts
    - Confirmation signals
    - Dynamic exit conditions
    """
    
    def __init__(self, config: StrategyConfig, backtest_config: BacktestConfig):
        # RSI-specific parameters
        self.rsi_period = config.parameters.get('rsi_period', 14)
        self.oversold_level = config.parameters.get('oversold_level', 30)
        self.overbought_level = config.parameters.get('overbought_level', 70)
        self.confirmation_period = config.parameters.get('confirmation_period', 3)
        
        # Initialize RSI indicator
        self.rsi = {}
        self.confirmation_buffer = {}
```

### Expected Results
- **Signals**: 2-4 per symbol (less frequent than trend following)
- **Win Rate**: 55-75% (higher due to mean reversion)
- **Trade Duration**: Shorter average duration
- **Market Conditions**: Better in ranging markets

---

## 🎯 Template 3: Bollinger Bands Squeeze

### Strategy Description
Volatility-based strategy using Bollinger Bands for breakout detection.

**Logic:**
- **Setup**: Identify periods of low volatility (squeeze)
- **Buy Signal**: Price breaks above upper band with volume
- **Sell Signal**: Price breaks below lower band with volume
- **Exit**: Price returns to middle band

### Key Features
- **Volatility Analysis**: Bollinger Bands calculation
- **Breakout Detection**: Price and volume confirmation
- **Dynamic Bands**: Adaptive to market conditions
- **Volume Confirmation**: Trade quality improvement

---

## 🎯 Template 4: Multi-Timeframe Strategy

### Strategy Description
Advanced strategy combining multiple timeframes for signal confirmation.

**Logic:**
- **Higher Timeframe**: Trend direction (1-hour MA)
- **Lower Timeframe**: Entry signals (1-minute RSI)
- **Confirmation**: Both timeframes must align
- **Exit**: Either timeframe gives opposite signal

### Key Features
- **Timeframe Coordination**: Multiple time horizon analysis
- **Trend-Momentum Combination**: Comprehensive market view
- **Signal Filtering**: Higher probability trades
- **Complex Logic**: Advanced strategy patterns

---

## 🎯 Template 5: Portfolio Strategy

### Strategy Description
Multi-symbol portfolio strategy with correlation analysis.

**Logic:**
- **Symbol Analysis**: Individual symbol signals
- **Correlation Check**: Avoid correlated positions
- **Portfolio Balance**: Equal risk allocation
- **Dynamic Hedging**: Automatic position balancing

### Key Features
- **Multi-Symbol Coordination**: Portfolio-level logic
- **Correlation Analysis**: Risk management
- **Dynamic Allocation**: Adaptive position sizing
- **Advanced Risk Management**: Portfolio-level controls

---

## 📊 Template Validation Framework

### Automated Testing
```python
def validate_all_templates():
    """Run validation tests for all strategy templates."""
    templates = [
        MovingAverageCrossoverTemplate,
        RSIMeanReversionTemplate,
        BollingerBandsSqueezeTemplate,
        MultiTimeframeTemplate,
        PortfolioStrategyTemplate
    ]
    
    results = {}
    for template in templates:
        results[template.__name__] = run_template_validation(template)
    
    return results
```

### Performance Benchmarking
```python
def benchmark_template_performance():
    """Benchmark template execution performance."""
    # Speed testing
    # Memory usage analysis
    # Scalability testing
    # Accuracy validation
```

### Comparison Analysis
```python
def compare_template_results():
    """Compare results across different templates."""
    # Performance comparison
    # Risk-return analysis
    # Market condition sensitivity
    # Parameter sensitivity
```

## 🚀 Usage Guidelines

### For Framework Testing
1. **Run All Templates**: Validate complete framework
2. **Check Expected Results**: Ensure performance within ranges
3. **Validate Edge Cases**: Test unusual market conditions
4. **Performance Testing**: Monitor speed and memory usage

### For Learning
1. **Start with Basic Templates**: Understanding fundamental concepts
2. **Modify Parameters**: Experiment with different settings
3. **Study Code Structure**: Learn best practices
4. **Create Variations**: Build upon existing templates

### For Development
1. **Use as Starting Points**: Build new strategies
2. **Follow Patterns**: Adopt proven structures
3. **Extend Functionality**: Add new features
4. **Validate Thoroughly**: Test all modifications

## 📋 Template Checklist

### Before Use
- [ ] Data available for test period
- [ ] Framework properly installed
- [ ] Configuration parameters set
- [ ] Expected results documented

### During Testing
- [ ] Monitor execution progress
- [ ] Check for errors or warnings
- [ ] Validate intermediate results
- [ ] Record performance metrics

### After Testing
- [ ] Review validation report
- [ ] Analyze performance metrics
- [ ] Compare with expected results
- [ ] Document any issues or insights

## 📚 Next Steps

1. **[Validation Tools](04-validation-tools.md)** - Comprehensive validation framework
2. **[Performance Benchmarking](05-benchmarking.md)** - Speed and accuracy testing
3. **[Strategy Development](../3-strategy-development/02-creating-strategies.md)** - Build your own strategies
4. **[Advanced Examples](../7-examples/02-advanced-strategies.md)** - Real-world implementations

---

**Professional Note**: These testing templates represent institutional-grade strategy development practices. Each template is designed to validate specific aspects of the backtesting framework while providing educational value for strategy development. Use them as both validation tools and learning resources to master systematic trading development.