# Creating a Custom Trading Strategy

## Overview

This guide walks you through creating custom trading strategies using the framework's strategy interface. You'll learn to build strategies from simple moving averages to complex multi-indicator systems.

## Strategy Architecture

### Base Strategy Class
All strategies inherit from the `TradingStrategy` base class:

```python
from strategy.strategy_interface import TradingStrategy, StrategyConfig
from core.data_structures import Signal, Tick
from typing import List

class MyCustomStrategy(TradingStrategy):
    def __init__(self, config: StrategyConfig, backtest_config):
        super().__init__(config, backtest_config)
        # Initialize your strategy-specific attributes
        self.initialize_strategy()
    
    def initialize_strategy(self):
        """Initialize strategy-specific components"""
        # Setup indicators, parameters, etc.
        pass
    
    def on_tick(self, tick: Tick) -> List[Signal]:
        """Process each tick and generate trading signals"""
        # Your trading logic here
        signals = []
        return signals
    
    def on_order_filled(self, order, fill_price, fill_time):
        """Handle order fill events"""
        # Update strategy state after order fills
        pass
```

### Strategy Configuration
```python
from strategy.strategy_interface import StrategyConfig

strategy_config = StrategyConfig(
    name="MyCustomStrategy",
    description="Custom trading strategy description",
    parameters={
        # Strategy-specific parameters
        'lookback_period': 20,
        'threshold': 0.02,
        'min_spread_pips': 1,
        'max_spread_pips': 5
    },
    risk_management={
        'max_position_size': 1.0,      # Maximum position size (lots)
        'stop_loss_pips': 30,          # Stop loss in pips
        'take_profit_pips': 60,        # Take profit in pips
        'risk_per_trade': 500,         # Risk per trade in base currency
        'max_open_positions': 3        # Maximum concurrent positions
    }
)
```

## Simple Strategy Example

### Basic Moving Average Strategy
```python
from indicators.moving_average import MovingAverage
from strategy.strategy_interface import TradingStrategy, StrategyConfig

class SimpleMovingAverageStrategy(TradingStrategy):
    def __init__(self, config: StrategyConfig, backtest_config):
        super().__init__(config, backtest_config)
        self.initialize_strategy()
    
    def initialize_strategy(self):
        # Get parameters
        self.ma_period = self.config.parameters.get('ma_period', 20)
        self.threshold = self.config.parameters.get('threshold', 0.001)
        
        # Initialize moving average indicator
        self.ma = MovingAverage(self.ma_period)
        
        # State tracking
        self.last_price = None
        self.last_signal_time = None
    
    def on_tick(self, tick: Tick) -> List[Signal]:
        signals = []
        
        # Update moving average
        mid_price = (tick.bid + tick.ask) / 2
        self.ma.update(mid_price)
        
        # Need enough data points
        if not self.ma.is_ready():
            self.last_price = mid_price
            return signals
        
        # Check spread conditions
        spread_pips = (tick.ask - tick.bid) * 10000
        if not self.is_spread_acceptable(spread_pips):
            return signals
        
        # Generate signals
        current_ma = self.ma.get_value()
        
        # Buy signal: price crosses above MA
        if (self.last_price is not None and 
            self.last_price <= current_ma and 
            mid_price > current_ma * (1 + self.threshold)):
            
            signal = self.create_signal(tick, 'BUY', strength=0.8)
            signals.append(signal)
        
        # Sell signal: price crosses below MA
        elif (self.last_price is not None and 
              self.last_price >= current_ma and 
              mid_price < current_ma * (1 - self.threshold)):
            
            signal = self.create_signal(tick, 'SELL', strength=0.8)
            signals.append(signal)
        
        self.last_price = mid_price
        return signals
    
    def is_spread_acceptable(self, spread_pips):
        """Check if spread is within acceptable range"""
        min_spread = self.config.parameters.get('min_spread_pips', 0)
        max_spread = self.config.parameters.get('max_spread_pips', 10)
        return min_spread <= spread_pips <= max_spread
```

## Advanced Strategy Example

### RSI Mean Reversion Strategy
```python
from indicators.rsi import RSI
from indicators.moving_average import MovingAverage
from strategy.strategy_interface import TradingStrategy, StrategyConfig

class RSIMeanReversionStrategy(TradingStrategy):
    def __init__(self, config: StrategyConfig, backtest_config):
        super().__init__(config, backtest_config)
        self.initialize_strategy()
    
    def initialize_strategy(self):
        # Parameters
        self.rsi_period = self.config.parameters.get('rsi_period', 14)
        self.ma_period = self.config.parameters.get('ma_period', 50)
        self.overbought_level = self.config.parameters.get('overbought_level', 70)
        self.oversold_level = self.config.parameters.get('oversold_level', 30)
        self.confirmation_threshold = self.config.parameters.get('confirmation_threshold', 0.001)
        
        # Initialize indicators
        self.rsi = RSI(self.rsi_period)
        self.ma = MovingAverage(self.ma_period)
        
        # State tracking
        self.price_history = []
        self.signal_cooldown = 0
        self.last_signal_type = None
    
    def on_tick(self, tick: Tick) -> List[Signal]:
        signals = []
        
        # Calculate mid price
        mid_price = (tick.bid + tick.ask) / 2
        
        # Update indicators
        self.rsi.update(mid_price)
        self.ma.update(mid_price)
        
        # Store price history
        self.price_history.append(mid_price)
        if len(self.price_history) > 100:  # Keep last 100 prices
            self.price_history.pop(0)
        
        # Reduce signal cooldown
        if self.signal_cooldown > 0:
            self.signal_cooldown -= 1
        
        # Need both indicators ready
        if not (self.rsi.is_ready() and self.ma.is_ready()):
            return signals
        
        # Check spread conditions
        spread_pips = (tick.ask - tick.bid) * 10000
        if not self.is_spread_acceptable(spread_pips):
            return signals
        
        # Get indicator values
        current_rsi = self.rsi.get_value()
        current_ma = self.ma.get_value()
        
        # Generate signals based on RSI and MA confirmation
        signal = self.evaluate_conditions(tick, current_rsi, current_ma, mid_price)
        if signal:
            signals.append(signal)
        
        return signals
    
    def evaluate_conditions(self, tick, rsi_value, ma_value, price):
        """Evaluate trading conditions and generate signals"""
        
        # Skip if in cooldown period
        if self.signal_cooldown > 0:
            return None
        
        # Buy conditions: RSI oversold + price near MA
        if (rsi_value < self.oversold_level and 
            price > ma_value * (1 - self.confirmation_threshold) and
            self.last_signal_type != 'BUY'):
            
            # Additional confirmation: price momentum
            if self.check_momentum_confirmation('BUY'):
                self.signal_cooldown = 10  # Wait 10 ticks before next signal
                self.last_signal_type = 'BUY'
                return self.create_signal(tick, 'BUY', strength=0.9)
        
        # Sell conditions: RSI overbought + price near MA
        elif (rsi_value > self.overbought_level and 
              price < ma_value * (1 + self.confirmation_threshold) and
              self.last_signal_type != 'SELL'):
            
            # Additional confirmation: price momentum
            if self.check_momentum_confirmation('SELL'):
                self.signal_cooldown = 10  # Wait 10 ticks before next signal
                self.last_signal_type = 'SELL'
                return self.create_signal(tick, 'SELL', strength=0.9)
        
        return None
    
    def check_momentum_confirmation(self, signal_type):
        """Check price momentum for signal confirmation"""
        if len(self.price_history) < 5:
            return False
        
        # Calculate recent momentum
        recent_momentum = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
        
        if signal_type == 'BUY':
            # For buy signals, we want recent downward momentum (mean reversion)
            return recent_momentum < -0.0005  # Small downward movement
        else:
            # For sell signals, we want recent upward momentum (mean reversion)
            return recent_momentum > 0.0005   # Small upward movement
    
    def on_order_filled(self, order, fill_price, fill_time):
        """Handle order fill events"""
        # Reset signal state after order fill
        self.last_signal_type = None
        self.signal_cooldown = 20  # Longer cooldown after fills
```

## Multi-Indicator Strategy

### Comprehensive Trend Following Strategy
```python
from indicators.moving_average import MovingAverage
from indicators.rsi import RSI
from indicators.bollinger_bands import BollingerBands
from strategy.strategy_interface import TradingStrategy, StrategyConfig

class TrendFollowingStrategy(TradingStrategy):
    def __init__(self, config: StrategyConfig, backtest_config):
        super().__init__(config, backtest_config)
        self.initialize_strategy()
    
    def initialize_strategy(self):
        # Parameters
        self.fast_ma_period = self.config.parameters.get('fast_ma_period', 10)
        self.slow_ma_period = self.config.parameters.get('slow_ma_period', 30)
        self.rsi_period = self.config.parameters.get('rsi_period', 14)
        self.bb_period = self.config.parameters.get('bb_period', 20)
        self.bb_std_dev = self.config.parameters.get('bb_std_dev', 2.0)
        
        # Initialize indicators
        self.fast_ma = MovingAverage(self.fast_ma_period)
        self.slow_ma = MovingAverage(self.slow_ma_period)
        self.rsi = RSI(self.rsi_period)
        self.bb = BollingerBands(self.bb_period, self.bb_std_dev)
        
        # State tracking
        self.trend_direction = None
        self.last_crossover_time = None
        self.position_count = 0
    
    def on_tick(self, tick: Tick) -> List[Signal]:
        signals = []
        
        # Calculate mid price
        mid_price = (tick.bid + tick.ask) / 2
        
        # Update all indicators
        self.fast_ma.update(mid_price)
        self.slow_ma.update(mid_price)
        self.rsi.update(mid_price)
        self.bb.update(mid_price)
        
        # Need all indicators ready
        if not all([self.fast_ma.is_ready(), self.slow_ma.is_ready(), 
                   self.rsi.is_ready(), self.bb.is_ready()]):
            return signals
        
        # Check spread conditions
        spread_pips = (tick.ask - tick.bid) * 10000
        if not self.is_spread_acceptable(spread_pips):
            return signals
        
        # Analyze market conditions
        market_analysis = self.analyze_market_conditions(tick, mid_price)
        
        # Generate signals based on comprehensive analysis
        signal = self.generate_signal(tick, market_analysis)
        if signal:
            signals.append(signal)
        
        return signals
    
    def analyze_market_conditions(self, tick, price):
        """Comprehensive market analysis"""
        
        # Get indicator values
        fast_ma = self.fast_ma.get_value()
        slow_ma = self.slow_ma.get_value()
        rsi_value = self.rsi.get_value()
        bb_upper, bb_middle, bb_lower = self.bb.get_values()
        
        # Trend analysis
        ma_trend = 'BULLISH' if fast_ma > slow_ma else 'BEARISH'
        price_vs_ma = 'ABOVE' if price > slow_ma else 'BELOW'
        
        # Momentum analysis
        momentum = 'STRONG' if abs(rsi_value - 50) > 20 else 'WEAK'
        rsi_direction = 'BULLISH' if rsi_value > 50 else 'BEARISH'
        
        # Volatility analysis
        bb_position = 'UPPER' if price > bb_upper else 'LOWER' if price < bb_lower else 'MIDDLE'
        bb_squeeze = (bb_upper - bb_lower) / bb_middle < 0.02  # Tight bands
        
        # Crossover detection
        crossover = self.detect_ma_crossover(fast_ma, slow_ma)
        
        return {
            'ma_trend': ma_trend,
            'price_vs_ma': price_vs_ma,
            'momentum': momentum,
            'rsi_direction': rsi_direction,
            'bb_position': bb_position,
            'bb_squeeze': bb_squeeze,
            'crossover': crossover,
            'price': price,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'rsi': rsi_value
        }
    
    def generate_signal(self, tick, analysis):
        """Generate trading signal based on comprehensive analysis"""
        
        # Bullish conditions
        bullish_conditions = [
            analysis['ma_trend'] == 'BULLISH',
            analysis['price_vs_ma'] == 'ABOVE',
            analysis['rsi_direction'] == 'BULLISH',
            analysis['rsi'] < 80,  # Not overbought
            analysis['crossover'] == 'BULLISH'
        ]
        
        # Bearish conditions
        bearish_conditions = [
            analysis['ma_trend'] == 'BEARISH',
            analysis['price_vs_ma'] == 'BELOW',
            analysis['rsi_direction'] == 'BEARISH',
            analysis['rsi'] > 20,  # Not oversold
            analysis['crossover'] == 'BEARISH'
        ]
        
        # Calculate signal strength
        bullish_strength = sum(bullish_conditions) / len(bullish_conditions)
        bearish_strength = sum(bearish_conditions) / len(bearish_conditions)
        
        # Generate signal if conditions are strong enough
        if bullish_strength >= 0.8 and bullish_strength > bearish_strength:
            return self.create_signal(tick, 'BUY', strength=bullish_strength)
        elif bearish_strength >= 0.8 and bearish_strength > bullish_strength:
            return self.create_signal(tick, 'SELL', strength=bearish_strength)
        
        return None
    
    def detect_ma_crossover(self, fast_ma, slow_ma):
        """Detect moving average crossover"""
        # This is simplified - in practice you'd track previous values
        if hasattr(self, 'prev_fast_ma') and hasattr(self, 'prev_slow_ma'):
            if (self.prev_fast_ma <= self.prev_slow_ma and fast_ma > slow_ma):
                crossover = 'BULLISH'
            elif (self.prev_fast_ma >= self.prev_slow_ma and fast_ma < slow_ma):
                crossover = 'BEARISH'
            else:
                crossover = None
        else:
            crossover = None
        
        # Store current values for next iteration
        self.prev_fast_ma = fast_ma
        self.prev_slow_ma = slow_ma
        
        return crossover
```

## Strategy Testing and Optimization

### Parameter Optimization
```python
def optimize_strategy_parameters(base_config, parameter_ranges):
    """Optimize strategy parameters using grid search"""
    
    best_result = None
    best_params = None
    best_return = -999
    
    # Generate parameter combinations
    from itertools import product
    
    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    
    for param_combo in product(*param_values):
        # Create parameter set
        params = dict(zip(param_names, param_combo))
        
        # Update strategy config
        test_config = base_config.copy()
        test_config.parameters.update(params)
        
        # Run backtest
        strategy = TrendFollowingStrategy(test_config, backtest_config)
        engine = BacktestEngine(backtest_config, './data')
        engine.add_strategy(strategy)
        
        result = engine.run_backtest()
        
        # Check if this is the best result
        if result.total_return > best_return:
            best_return = result.total_return
            best_params = params
            best_result = result
    
    return best_params, best_result

# Example parameter optimization
parameter_ranges = {
    'fast_ma_period': [5, 10, 15],
    'slow_ma_period': [20, 30, 40],
    'rsi_period': [10, 14, 20],
    'bb_period': [15, 20, 25]
}

best_params, best_result = optimize_strategy_parameters(
    strategy_config, parameter_ranges
)
```

### Walk-Forward Analysis
```python
def walk_forward_analysis(strategy_class, config, periods):
    """Perform walk-forward analysis"""
    
    results = []
    
    for i, period in enumerate(periods):
        # Training period
        train_start = period['train_start']
        train_end = period['train_end']
        
        # Test period
        test_start = period['test_start']
        test_end = period['test_end']
        
        # Optimize on training data
        train_config = config.copy()
        train_config.start_date = train_start
        train_config.end_date = train_end
        
        best_params, _ = optimize_strategy_parameters(train_config, parameter_ranges)
        
        # Test on out-of-sample data
        test_config = config.copy()
        test_config.start_date = test_start
        test_config.end_date = test_end
        test_config.parameters.update(best_params)
        
        strategy = strategy_class(test_config, backtest_config)
        engine = BacktestEngine(backtest_config, './data')
        engine.add_strategy(strategy)
        
        result = engine.run_backtest()
        results.append({
            'period': i,
            'train_period': (train_start, train_end),
            'test_period': (test_start, test_end),
            'best_params': best_params,
            'result': result
        })
    
    return results
```

## Strategy Utilities

### Signal Creation Helper
```python
def create_signal(self, tick, direction, strength=1.0, reason=None):
    """Create a trading signal with comprehensive information"""
    
    # Calculate position size based on risk management
    risk_amount = self.config.risk_management.get('risk_per_trade', 1000)
    stop_loss_pips = self.config.risk_management.get('stop_loss_pips', 30)
    
    # Calculate position size
    pip_value = self.calculate_pip_value(tick.symbol)
    position_size = risk_amount / (stop_loss_pips * pip_value)
    
    # Limit position size
    max_position_size = self.config.risk_management.get('max_position_size', 1.0)
    position_size = min(position_size, max_position_size)
    
    # Create signal
    signal = Signal(
        timestamp=tick.timestamp,
        symbol=tick.symbol,
        direction=direction,
        strength=strength,
        entry_price=tick.ask if direction == 'BUY' else tick.bid,
        position_size=position_size,
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=self.config.risk_management.get('take_profit_pips', 60),
        reason=reason or f"{self.config.name} signal"
    )
    
    return signal
```

### Risk Management
```python
def check_risk_limits(self, signal):
    """Check if signal passes risk management rules"""
    
    # Check maximum open positions
    max_positions = self.config.risk_management.get('max_open_positions', 5)
    if len(self.open_positions) >= max_positions:
        return False, "Maximum open positions reached"
    
    # Check maximum exposure per symbol
    symbol_exposure = sum(pos.position_size for pos in self.open_positions 
                         if pos.symbol == signal.symbol)
    max_symbol_exposure = self.config.risk_management.get('max_symbol_exposure', 2.0)
    if symbol_exposure + signal.position_size > max_symbol_exposure:
        return False, "Maximum symbol exposure exceeded"
    
    # Check total portfolio risk
    total_risk = self.calculate_total_portfolio_risk()
    max_portfolio_risk = self.config.risk_management.get('max_portfolio_risk', 0.05)
    if total_risk > max_portfolio_risk:
        return False, "Maximum portfolio risk exceeded"
    
    return True, "Risk checks passed"
```

## Best Practices

### Strategy Development
1. **Start simple**: Begin with basic strategies and add complexity gradually
2. **Use proper indicators**: Implement robust technical indicators
3. **Validate signals**: Always validate signal quality before trading
4. **Handle edge cases**: Account for market gaps, extreme volatility, etc.
5. **Test thoroughly**: Use multiple market conditions and time periods

### Performance Optimization
1. **Minimize calculations**: Cache expensive calculations
2. **Use vectorized operations**: Leverage pandas/numpy for speed
3. **Limit indicator lookback**: Don't store excessive historical data
4. **Profile performance**: Identify and optimize bottlenecks

### Risk Management
1. **Define clear rules**: Establish stop-loss and take-profit levels
2. **Position sizing**: Use proper risk-based position sizing
3. **Diversification**: Avoid over-concentration in single symbols
4. **Maximum drawdown**: Monitor and limit portfolio drawdown

---

*Building effective trading strategies requires careful design, thorough testing, and robust risk management. Start with simple concepts and gradually build more sophisticated systems as you gain experience.*