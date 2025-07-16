"""
Moving Average Crossover Testing Strategy
========================================

A comprehensive testing strategy implementing moving average crossover logic
with full validation, error handling, and performance tracking.

Purpose:
- Validate backtesting framework functionality
- Demonstrate best practices in strategy development
- Provide educational reference for strategy creation
- Serve as benchmark for performance comparison

Author: Backtesting Framework
Version: 1.0
Last Updated: July 2025
"""

from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import json

from core.data_structures import (
    Tick, Order, Position, OrderSide, BacktestConfig
)
from strategy.strategy_interface import (
    TradingStrategy, StrategyConfig, StrategySignal, 
    MovingAverage, TechnicalIndicator
)


logger = logging.getLogger(__name__)


class MovingAverageCrossoverTestStrategy(TradingStrategy):
    """
    Professional Moving Average Crossover Strategy for Testing
    
    This strategy implements a classic moving average crossover system with:
    - Comprehensive validation and error handling
    - Performance tracking and metrics
    - Risk management integration
    - Educational documentation
    
    Trading Logic:
    - Buy when fast MA crosses above slow MA
    - Sell when fast MA crosses below slow MA
    - Exit on opposite crossover or risk management
    
    Validation Features:
    - Signal generation accuracy
    - Risk management compliance
    - Position tracking correctness
    - Performance metrics calculation
    """
    
    def __init__(self, config: StrategyConfig, backtest_config: BacktestConfig):
        """
        Initialize the Moving Average Crossover Test Strategy.
        
        Args:
            config: Strategy configuration with parameters and risk settings
            backtest_config: Backtesting configuration for market simulation
        """
        super().__init__(config, backtest_config)
        
        # Strategy parameters with validation
        self.fast_period = self._validate_parameter('fast_period', 10, min_val=2, max_val=50)
        self.slow_period = self._validate_parameter('slow_period', 20, min_val=5, max_val=200)
        
        # Ensure fast period is less than slow period
        if self.fast_period >= self.slow_period:
            self.slow_period = self.fast_period + 10
            logger.warning(f"Adjusted slow_period to {self.slow_period} to ensure fast < slow")
        
        # Market condition filters
        self.min_spread_pips = self._validate_parameter('min_spread_pips', 0.5, min_val=0.1, max_val=5.0)
        self.max_spread_pips = self._validate_parameter('max_spread_pips', 4.0, min_val=1.0, max_val=20.0)
        self.min_volatility = self._validate_parameter('min_volatility', 0.0001, min_val=0.0, max_val=0.01)
        
        # Signal confirmation settings
        self.confirmation_bars = self._validate_parameter('confirmation_bars', 1, min_val=1, max_val=5)
        self.min_signal_strength = self._validate_parameter('min_signal_strength', 0.3, min_val=0.1, max_val=1.0)
        
        # Technical indicators (per symbol)
        self.fast_ma: Dict[str, MovingAverage] = {}
        self.slow_ma: Dict[str, MovingAverage] = {}
        self.volatility_tracker: Dict[str, List[float]] = {}
        
        # Strategy state tracking
        self.position_states: Dict[str, Dict[str, Any]] = {}
        self.signal_history: List[StrategySignal] = []
        self.last_crossover_time: Dict[str, Optional[datetime]] = {}
        
        # Performance and validation metrics
        self.metrics = {
            'signals_generated': 0,
            'signals_executed': 0,
            'valid_signals': 0,
            'invalid_signals': 0,
            'spread_rejections': 0,
            'volatility_rejections': 0,
            'crossover_detections': 0,
            'risk_management_triggers': 0,
            'errors': []
        }
        
        # Educational tracking
        self.educational_data = {
            'crossover_examples': [],
            'risk_management_examples': [],
            'signal_validation_examples': []
        }
        
        logger.info(f"MA Crossover Test Strategy initialized: {self.fast_period}/{self.slow_period}")
    
    def _validate_parameter(self, param_name: str, default: float, min_val: float, max_val: float) -> float:
        """
        Validate and constrain strategy parameters.
        
        Args:
            param_name: Parameter name
            default: Default value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated parameter value
        """
        value = self.config.parameters.get(param_name, default)
        
        if value < min_val:
            logger.warning(f"Parameter {param_name}={value} below minimum {min_val}, using minimum")
            return min_val
        elif value > max_val:
            logger.warning(f"Parameter {param_name}={value} above maximum {max_val}, using maximum")
            return max_val
        
        return value
    
    def initialize(self) -> None:
        """
        Initialize strategy components for all symbols.
        
        Educational Note:
        This method is called once before backtesting begins.
        It sets up all necessary indicators and state tracking.
        """
        try:
            # Initialize indicators and state for each symbol
            for symbol in self.backtest_config.symbols:
                # Technical indicators
                self.fast_ma[symbol] = MovingAverage(self.fast_period)
                self.slow_ma[symbol] = MovingAverage(self.slow_period)
                
                # Volatility tracking
                self.volatility_tracker[symbol] = []
                
                # Position and signal state
                self.position_states[symbol] = {
                    'position': None,
                    'entry_time': None,
                    'entry_price': None,
                    'last_signal_type': None,
                    'last_signal_time': None,
                    'signal_count': 0
                }
                
                # Crossover tracking
                self.last_crossover_time[symbol] = None
                
                logger.info(f"Initialized indicators for {symbol}")
            
            logger.info(f"Strategy initialization completed for {len(self.backtest_config.symbols)} symbols")
            
        except Exception as e:
            error_msg = f"Strategy initialization failed: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            raise
    
    def on_tick(self, tick: Tick) -> List[StrategySignal]:
        """
        Process each tick and generate trading signals.
        
        Educational Note:
        This method is called for every tick in the backtest.
        It implements the core trading logic and signal generation.
        
        Args:
            tick: Market tick data
            
        Returns:
            List of generated trading signals
        """
        signals = []
        
        try:
            # Add tick to price history
            self.add_tick_to_history(tick)
            
            # Pre-filtering: Check market conditions
            if not self._is_market_suitable(tick):
                return signals
            
            # Update technical indicators
            self._update_indicators(tick)
            
            # Check if indicators are ready
            if not self._are_indicators_ready(tick.symbol):
                return signals
            
            # Detect crossovers and generate signals
            crossover_signal = self._detect_crossover_signal(tick)
            if crossover_signal:
                signals.append(crossover_signal)
                self.metrics['crossover_detections'] += 1
            
            # Check for exit conditions
            exit_signal = self._check_exit_conditions(tick)
            if exit_signal:
                signals.append(exit_signal)
            
            # Validate and process all signals
            validated_signals = []
            for signal in signals:
                if self._validate_signal(signal):
                    validated_signals.append(signal)
                    self.metrics['valid_signals'] += 1
                    self.signal_history.append(signal)
                else:
                    self.metrics['invalid_signals'] += 1
            
            # Update metrics
            self.metrics['signals_generated'] += len(validated_signals)
            
            return validated_signals
            
        except Exception as e:
            error_msg = f"Error processing tick for {tick.symbol}: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            return []
    
    def _is_market_suitable(self, tick: Tick) -> bool:
        """
        Check if market conditions are suitable for trading.
        
        Educational Note:
        This method demonstrates pre-filtering based on market conditions
        like spread and volatility to improve signal quality.
        
        Args:
            tick: Current market tick
            
        Returns:
            True if market conditions are suitable for trading
        """
        # Check spread conditions
        if not self._is_spread_acceptable(tick):
            self.metrics['spread_rejections'] += 1
            return False
        
        # Check volatility conditions
        if not self._is_volatility_acceptable(tick):
            self.metrics['volatility_rejections'] += 1
            return False
        
        return True
    
    def _is_spread_acceptable(self, tick: Tick) -> bool:
        """
        Validate spread conditions.
        
        Educational Note:
        Wide spreads can significantly impact trading profitability.
        This method filters out periods with excessive spreads.
        """
        try:
            pip_value = self._get_pip_value(tick.symbol)
            spread_pips = tick.spread / pip_value
            
            is_acceptable = self.min_spread_pips <= spread_pips <= self.max_spread_pips
            
            if not is_acceptable:
                logger.debug(f"Spread rejected for {tick.symbol}: {spread_pips:.2f} pips "
                           f"(range: {self.min_spread_pips}-{self.max_spread_pips})")
            
            return is_acceptable
            
        except Exception as e:
            logger.error(f"Error checking spread for {tick.symbol}: {str(e)}")
            return False
    
    def _is_volatility_acceptable(self, tick: Tick) -> bool:
        """
        Check if volatility is within acceptable range.
        
        Educational Note:
        Very low volatility can result in false signals.
        This method ensures minimum volatility for trading.
        """
        try:
            symbol = tick.symbol
            
            # Update volatility tracker
            if symbol not in self.volatility_tracker:
                self.volatility_tracker[symbol] = []
            
            price_history = self.get_price_history(symbol, 20)
            if len(price_history) < 10:
                return True  # Allow trading with insufficient history
            
            # Calculate recent volatility
            prices = [t.mid for t in price_history]
            returns = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            avg_volatility = sum(returns) / len(returns) if returns else 0
            
            # Update volatility tracker
            self.volatility_tracker[symbol].append(avg_volatility)
            if len(self.volatility_tracker[symbol]) > 100:
                self.volatility_tracker[symbol].pop(0)
            
            return avg_volatility >= self.min_volatility
            
        except Exception as e:
            logger.error(f"Error checking volatility for {tick.symbol}: {str(e)}")
            return True  # Default to allowing trading
    
    def _update_indicators(self, tick: Tick) -> None:
        """
        Update technical indicators with new tick data.
        
        Educational Note:
        This method shows how to properly update indicators
        and handle potential calculation errors.
        """
        try:
            symbol = tick.symbol
            mid_price = tick.mid
            
            # Update moving averages
            self.fast_ma[symbol].update(mid_price)
            self.slow_ma[symbol].update(mid_price)
            
            logger.debug(f"Updated indicators for {symbol}: price={mid_price:.5f}")
            
        except Exception as e:
            error_msg = f"Error updating indicators for {tick.symbol}: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
    
    def _are_indicators_ready(self, symbol: str) -> bool:
        """
        Check if all indicators have sufficient data.
        
        Educational Note:
        Indicators need a minimum number of data points before
        they can produce meaningful values.
        """
        try:
            return (self.fast_ma[symbol].is_ready and 
                   self.slow_ma[symbol].is_ready)
        except Exception as e:
            logger.error(f"Error checking indicator readiness for {symbol}: {str(e)}")
            return False
    
    def _detect_crossover_signal(self, tick: Tick) -> Optional[StrategySignal]:
        """
        Detect moving average crossover signals.
        
        Educational Note:
        This method demonstrates crossover detection logic with
        proper historical data handling and signal strength calculation.
        
        Args:
            tick: Current market tick
            
        Returns:
            Trading signal if crossover detected, None otherwise
        """
        try:
            symbol = tick.symbol
            
            # Get indicator history for crossover detection
            fast_history = self.fast_ma[symbol].get_history(self.confirmation_bars + 1)
            slow_history = self.slow_ma[symbol].get_history(self.confirmation_bars + 1)
            
            if len(fast_history) < 2 or len(slow_history) < 2:
                return None
            
            # Check for crossover conditions
            prev_fast, curr_fast = fast_history[-2], fast_history[-1]
            prev_slow, curr_slow = slow_history[-2], slow_history[-1]
            
            # Prevent duplicate signals
            if self._is_recent_crossover(symbol, tick.timestamp):
                return None
            
            # Check for existing position
            position = self.position_states[symbol]['position']
            
            # Golden Cross: Fast MA crosses above Slow MA
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                if not position or position.is_closed or position.is_short:
                    signal = self._create_buy_signal(tick, curr_fast, curr_slow)
                    if signal:
                        self.last_crossover_time[symbol] = tick.timestamp
                        self._record_crossover_example(symbol, 'GOLDEN_CROSS', 
                                                     prev_fast, curr_fast, prev_slow, curr_slow)
                        return signal
            
            # Death Cross: Fast MA crosses below Slow MA
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                if not position or position.is_closed or position.is_long:
                    signal = self._create_sell_signal(tick, curr_fast, curr_slow)
                    if signal:
                        self.last_crossover_time[symbol] = tick.timestamp
                        self._record_crossover_example(symbol, 'DEATH_CROSS', 
                                                     prev_fast, curr_fast, prev_slow, curr_slow)
                        return signal
            
            return None
            
        except Exception as e:
            error_msg = f"Error detecting crossover for {tick.symbol}: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            return None
    
    def _is_recent_crossover(self, symbol: str, current_time: datetime) -> bool:
        """
        Check if a crossover signal was recently generated.
        
        Educational Note:
        This prevents duplicate signals from being generated
        in rapid succession during volatile periods.
        """
        last_time = self.last_crossover_time.get(symbol)
        if last_time is None:
            return False
        
        time_diff = (current_time - last_time).total_seconds()
        return time_diff < 300  # 5 minutes minimum between signals
    
    def _create_buy_signal(self, tick: Tick, fast_ma: float, slow_ma: float) -> Optional[StrategySignal]:
        """
        Create a validated buy signal.
        
        Educational Note:
        This method demonstrates comprehensive signal creation with
        risk management integration and metadata tracking.
        """
        try:
            symbol = tick.symbol
            
            # Calculate signal strength based on multiple factors
            strength = self._calculate_signal_strength(tick, fast_ma, slow_ma, 'BUY')
            
            if strength < self.min_signal_strength:
                logger.debug(f"Buy signal strength too low: {strength:.2f}")
                return None
            
            # Calculate position size based on risk management
            position_size = self._calculate_position_size(tick, strength)
            
            # Create signal with comprehensive metadata
            signal = self.create_signal(
                tick=tick,
                signal_type='BUY',
                strength=strength,
                quantity=position_size,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'ma_separation': abs(fast_ma - slow_ma),
                    'signal_quality': 'HIGH' if strength > 0.7 else 'MEDIUM' if strength > 0.5 else 'LOW',
                    'entry_reason': 'GOLDEN_CROSS',
                    'confirmation_bars': self.confirmation_bars,
                    'spread_pips': tick.spread / self._get_pip_value(symbol),
                    'strategy_version': '1.0'
                }
            )
            
            # Update position state
            self.position_states[symbol]['last_signal_type'] = 'BUY'
            self.position_states[symbol]['last_signal_time'] = tick.timestamp
            self.position_states[symbol]['signal_count'] += 1
            
            # Record educational example
            self._record_signal_example(signal, 'BUY', strength, fast_ma, slow_ma)
            
            logger.info(f"BUY signal created for {symbol}: strength={strength:.2f}, "
                       f"size={position_size:.2f}, MA({fast_ma:.5f}>{slow_ma:.5f})")
            
            return signal
            
        except Exception as e:
            error_msg = f"Error creating buy signal for {tick.symbol}: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            return None
    
    def _create_sell_signal(self, tick: Tick, fast_ma: float, slow_ma: float) -> Optional[StrategySignal]:
        """
        Create a validated sell signal.
        
        Educational Note:
        This method mirrors the buy signal creation but with
        appropriate adjustments for short positions.
        """
        try:
            symbol = tick.symbol
            
            # Calculate signal strength based on multiple factors
            strength = self._calculate_signal_strength(tick, fast_ma, slow_ma, 'SELL')
            
            if strength < self.min_signal_strength:
                logger.debug(f"Sell signal strength too low: {strength:.2f}")
                return None
            
            # Calculate position size based on risk management
            position_size = self._calculate_position_size(tick, strength)
            
            # Create signal with comprehensive metadata
            signal = self.create_signal(
                tick=tick,
                signal_type='SELL',
                strength=strength,
                quantity=position_size,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'ma_separation': abs(fast_ma - slow_ma),
                    'signal_quality': 'HIGH' if strength > 0.7 else 'MEDIUM' if strength > 0.5 else 'LOW',
                    'entry_reason': 'DEATH_CROSS',
                    'confirmation_bars': self.confirmation_bars,
                    'spread_pips': tick.spread / self._get_pip_value(symbol),
                    'strategy_version': '1.0'
                }
            )
            
            # Update position state
            self.position_states[symbol]['last_signal_type'] = 'SELL'
            self.position_states[symbol]['last_signal_time'] = tick.timestamp
            self.position_states[symbol]['signal_count'] += 1
            
            # Record educational example
            self._record_signal_example(signal, 'SELL', strength, fast_ma, slow_ma)
            
            logger.info(f"SELL signal created for {symbol}: strength={strength:.2f}, "
                       f"size={position_size:.2f}, MA({fast_ma:.5f}<{slow_ma:.5f})")
            
            return signal
            
        except Exception as e:
            error_msg = f"Error creating sell signal for {tick.symbol}: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            return None
    
    def _calculate_signal_strength(self, tick: Tick, fast_ma: float, slow_ma: float, signal_type: str) -> float:
        """
        Calculate signal strength based on multiple market factors.
        
        Educational Note:
        Signal strength helps prioritize trades and adjust position sizing.
        Higher strength signals generally have better success rates.
        """
        try:
            base_strength = 0.5
            
            # Factor 1: MA separation (stronger when MAs are further apart)
            ma_separation = abs(fast_ma - slow_ma) / tick.mid
            separation_strength = min(ma_separation * 2000, 0.3)  # Cap at 0.3
            
            # Factor 2: Recent volatility (stronger in trending markets)
            volatility_strength = 0.0
            if tick.symbol in self.volatility_tracker and self.volatility_tracker[tick.symbol]:
                recent_volatility = self.volatility_tracker[tick.symbol][-1]
                volatility_strength = min(recent_volatility * 500, 0.2)  # Cap at 0.2
            
            # Factor 3: Price momentum (stronger when price supports signal)
            momentum_strength = 0.0
            price_history = self.get_price_history(tick.symbol, 5)
            if len(price_history) >= 3:
                recent_trend = (price_history[-1].mid - price_history[-3].mid) / price_history[-3].mid
                if signal_type == 'BUY' and recent_trend > 0:
                    momentum_strength = min(abs(recent_trend) * 100, 0.2)
                elif signal_type == 'SELL' and recent_trend < 0:
                    momentum_strength = min(abs(recent_trend) * 100, 0.2)
            
            # Combine all factors
            total_strength = base_strength + separation_strength + volatility_strength + momentum_strength
            
            return min(total_strength, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return 0.5  # Default strength
    
    def _calculate_position_size(self, tick: Tick, signal_strength: float) -> float:
        """
        Calculate position size based on risk management and signal strength.
        
        Educational Note:
        Position sizing is crucial for risk management. This method
        demonstrates how to integrate signal quality with position sizing.
        """
        try:
            # Base position size from risk management
            base_size = self.config.risk_management.get('max_position_size', 1.0)
            
            # Adjust based on signal strength
            strength_multiplier = 0.5 + (signal_strength * 0.5)  # 0.5 to 1.0 range
            
            # Apply risk management constraints
            risk_adjusted_size = base_size * strength_multiplier
            
            # Final position size
            final_size = min(risk_adjusted_size, 
                           self.config.risk_management.get('max_position_size', 1.0))
            
            return max(final_size, 0.1)  # Minimum 0.1 lots
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1.0  # Default size
    
    def _check_exit_conditions(self, tick: Tick) -> Optional[StrategySignal]:
        """
        Check for position exit conditions.
        
        Educational Note:
        Exit conditions are as important as entry conditions.
        This method demonstrates systematic exit logic.
        """
        try:
            symbol = tick.symbol
            position = self.position_states[symbol]['position']
            
            if not position or position.is_closed:
                return None
            
            # Get current MA values
            fast_ma = self.fast_ma[symbol].get_value()
            slow_ma = self.slow_ma[symbol].get_value()
            
            if fast_ma is None or slow_ma is None:
                return None
            
            # Get MA history for exit detection
            fast_history = self.fast_ma[symbol].get_history(2)
            slow_history = self.slow_ma[symbol].get_history(2)
            
            if len(fast_history) < 2 or len(slow_history) < 2:
                return None
            
            prev_fast, curr_fast = fast_history[-2], fast_history[-1]
            prev_slow, curr_slow = slow_history[-2], slow_history[-1]
            
            exit_reason = None
            
            # Exit long position on death cross
            if position.is_long and prev_fast >= prev_slow and curr_fast < curr_slow:
                exit_reason = "DEATH_CROSS_EXIT"
            
            # Exit short position on golden cross
            elif position.is_short and prev_fast <= prev_slow and curr_fast > curr_slow:
                exit_reason = "GOLDEN_CROSS_EXIT"
            
            if exit_reason:
                # Create exit signal
                signal = StrategySignal(
                    timestamp=tick.timestamp,
                    symbol=symbol,
                    signal_type='CLOSE',
                    strength=1.0,  # Always exit with full confidence
                    price=tick.mid,
                    quantity=abs(position.quantity),
                    metadata={
                        'exit_reason': exit_reason,
                        'position_duration': (tick.timestamp - 
                                           self.position_states[symbol]['entry_time']).total_seconds(),
                        'entry_price': self.position_states[symbol]['entry_price'],
                        'current_pnl': position.unrealized_pnl,
                        'fast_ma': curr_fast,
                        'slow_ma': curr_slow,
                        'strategy_version': '1.0'
                    }
                )
                
                # Record exit example
                self._record_exit_example(signal, exit_reason, position)
                
                logger.info(f"EXIT signal created for {symbol}: {exit_reason}")
                return signal
            
            return None
            
        except Exception as e:
            error_msg = f"Error checking exit conditions for {tick.symbol}: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            return None
    
    def _validate_signal(self, signal: StrategySignal) -> bool:
        """
        Validate signal before execution.
        
        Educational Note:
        Signal validation is crucial for preventing invalid trades.
        This method demonstrates comprehensive signal validation.
        """
        try:
            # Validate signal strength
            if signal.strength < 0.1 or signal.strength > 1.0:
                logger.warning(f"Invalid signal strength: {signal.strength}")
                return False
            
            # Validate quantity
            if signal.quantity <= 0:
                logger.warning(f"Invalid signal quantity: {signal.quantity}")
                return False
            
            # Validate price
            if signal.price <= 0:
                logger.warning(f"Invalid signal price: {signal.price}")
                return False
            
            # Validate signal type
            if signal.signal_type not in ['BUY', 'SELL', 'CLOSE']:
                logger.warning(f"Invalid signal type: {signal.signal_type}")
                return False
            
            # Add to validation examples
            self._record_validation_example(signal, True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {str(e)}")
            self._record_validation_example(signal, False)
            return False
    
    def on_order_filled(self, order: Order) -> None:
        """
        Handle order execution events.
        
        Educational Note:
        This method is called when orders are filled, allowing
        the strategy to update its state and track performance.
        """
        try:
            self.metrics['signals_executed'] += 1
            symbol = order.symbol
            
            # Update position state
            if symbol in self.position_states:
                self.position_states[symbol]['entry_time'] = order.filled_at
                self.position_states[symbol]['entry_price'] = order.avg_fill_price
            
            logger.info(f"Order filled: {order.side.value} {order.filled_quantity} lots "
                       f"of {symbol} at {order.avg_fill_price:.5f} "
                       f"(commission: {order.commission:.2f}, slippage: {order.slippage:.2f})")
            
        except Exception as e:
            error_msg = f"Error handling order fill: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
    
    def on_position_update(self, position: Position) -> None:
        """
        Handle position updates.
        
        Educational Note:
        This method tracks position changes and updates strategy state.
        It's essential for maintaining accurate position information.
        """
        try:
            symbol = position.symbol
            
            # Update position tracking
            self.position_states[symbol]['position'] = position
            
            # Update performance metrics
            if position.is_closed:
                # Position closed
                self.position_states[symbol]['entry_time'] = None
                self.position_states[symbol]['entry_price'] = None
                
                # Update win/loss counts
                if position.total_pnl > 0:
                    self.winning_signals += 1
                else:
                    self.losing_signals += 1
                
                logger.info(f"Position closed: {symbol} P&L: {position.total_pnl:.2f}")
            else:
                # Position updated
                logger.info(f"Position updated: {symbol} {position.quantity} lots, "
                           f"Unrealized P&L: {position.unrealized_pnl:.2f}")
            
        except Exception as e:
            error_msg = f"Error handling position update: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
    
    def _record_crossover_example(self, symbol: str, crossover_type: str, 
                                 prev_fast: float, curr_fast: float, 
                                 prev_slow: float, curr_slow: float) -> None:
        """Record crossover example for educational purposes."""
        example = {
            'symbol': symbol,
            'type': crossover_type,
            'timestamp': datetime.now().isoformat(),
            'prev_fast_ma': prev_fast,
            'curr_fast_ma': curr_fast,
            'prev_slow_ma': prev_slow,
            'curr_slow_ma': curr_slow,
            'crossover_magnitude': abs(curr_fast - curr_slow)
        }
        
        self.educational_data['crossover_examples'].append(example)
    
    def _record_signal_example(self, signal: StrategySignal, signal_type: str, 
                              strength: float, fast_ma: float, slow_ma: float) -> None:
        """Record signal example for educational purposes."""
        example = {
            'signal_type': signal_type,
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'strength': strength,
            'quantity': signal.quantity,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'metadata': signal.metadata
        }
        
        self.educational_data['signal_validation_examples'].append(example)
    
    def _record_exit_example(self, signal: StrategySignal, exit_reason: str, position: Position) -> None:
        """Record exit example for educational purposes."""
        example = {
            'exit_reason': exit_reason,
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'position_pnl': position.unrealized_pnl,
            'position_duration': signal.metadata.get('position_duration', 0),
            'entry_price': signal.metadata.get('entry_price', 0),
            'exit_price': signal.price
        }
        
        self.educational_data['risk_management_examples'].append(example)
    
    def _record_validation_example(self, signal: StrategySignal, is_valid: bool) -> None:
        """Record validation example for educational purposes."""
        example = {
            'signal_type': signal.signal_type,
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'is_valid': is_valid,
            'strength': signal.strength,
            'quantity': signal.quantity,
            'price': signal.price
        }
        
        self.educational_data['signal_validation_examples'].append(example)
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy metrics.
        
        Educational Note:
        This method provides detailed metrics for strategy evaluation
        and performance analysis.
        """
        base_metrics = self.get_performance_metrics()
        
        # Calculate additional metrics
        execution_rate = (self.metrics['signals_executed'] / 
                         max(self.metrics['signals_generated'], 1))
        
        validation_rate = (self.metrics['valid_signals'] / 
                          max(self.metrics['valid_signals'] + self.metrics['invalid_signals'], 1))
        
        strategy_metrics = {
            **base_metrics,
            **self.metrics,
            'execution_rate': execution_rate,
            'validation_rate': validation_rate,
            'strategy_type': 'MovingAverageCrossover',
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'min_spread_pips': self.min_spread_pips,
                'max_spread_pips': self.max_spread_pips,
                'confirmation_bars': self.confirmation_bars
            },
            'educational_data': self.educational_data
        }
        
        return strategy_metrics
    
    def export_educational_data(self, filename: str) -> None:
        """
        Export educational data for analysis.
        
        Educational Note:
        This method exports detailed examples and metrics for
        educational analysis and strategy improvement.
        """
        try:
            educational_export = {
                'strategy_info': {
                    'name': self.config.name,
                    'version': '1.0',
                    'parameters': {
                        'fast_period': self.fast_period,
                        'slow_period': self.slow_period,
                        'min_spread_pips': self.min_spread_pips,
                        'max_spread_pips': self.max_spread_pips
                    }
                },
                'metrics': self.get_strategy_metrics(),
                'educational_examples': self.educational_data,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(educational_export, f, indent=2)
            
            logger.info(f"Educational data exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting educational data: {str(e)}")
    
    def finalize(self) -> None:
        """
        Finalize strategy execution.
        
        Educational Note:
        This method is called after backtesting completes,
        providing a chance to calculate final metrics and clean up.
        """
        super().finalize()
        
        # Calculate final metrics
        total_signals = self.metrics['signals_generated']
        total_errors = len(self.metrics['errors'])
        
        # Log final statistics
        logger.info(f"Strategy finalized - Total signals: {total_signals}, "
                   f"Executed: {self.metrics['signals_executed']}, "
                   f"Errors: {total_errors}")
        
        # Update final performance metrics
        self.state.performance.update({
            'final_metrics': self.get_strategy_metrics(),
            'crossover_detection_count': self.metrics['crossover_detections'],
            'signal_validation_rate': self.metrics['valid_signals'] / max(total_signals, 1),
            'error_rate': total_errors / max(total_signals, 1)
        })


def create_test_strategy_config() -> StrategyConfig:
    """
    Create a standardized configuration for testing.
    
    Educational Note:
    This function provides a baseline configuration that can be
    modified for different testing scenarios.
    """
    return StrategyConfig(
        name="MA_Crossover_Test_Strategy",
        description="Professional Moving Average Crossover Strategy for Framework Testing",
        parameters={
            'fast_period': 10,
            'slow_period': 20,
            'min_spread_pips': 0.5,
            'max_spread_pips': 4.0,
            'min_volatility': 0.0001,
            'confirmation_bars': 1,
            'min_signal_strength': 0.3
        },
        risk_management={
            'max_position_size': 1.0,
            'stop_loss_pips': 30,
            'take_profit_pips': 60,
            'max_daily_loss': 2000,
            'max_drawdown': 0.15,
            'risk_per_trade': 500
        }
    )


def run_strategy_test(data_path: str = "./data") -> Dict[str, Any]:
    """
    Run a complete strategy test with validation.
    
    Educational Note:
    This function demonstrates how to set up and run a complete
    strategy test with comprehensive validation and reporting.
    """
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
    
    # Create and run strategy
    try:
        strategy_config = create_test_strategy_config()
        strategy = MovingAverageCrossoverTestStrategy(strategy_config, backtest_config)
        
        engine = BacktestEngine(backtest_config, data_path)
        engine.add_strategy(strategy)
        
        result = engine.run_backtest()
        
        # Analyze results
        analyzer = PerformanceAnalyzer(result)
        performance_report = analyzer.generate_summary_report()
        
        # Get strategy metrics
        strategy_metrics = strategy.get_strategy_metrics()
        
        # Export educational data
        strategy.export_educational_data('ma_crossover_educational_data.json')
        
        return {
            'success': True,
            'backtest_result': result,
            'performance_report': performance_report,
            'strategy_metrics': strategy_metrics,
            'test_summary': {
                'total_signals': strategy_metrics['signals_generated'],
                'execution_rate': strategy_metrics['execution_rate'],
                'validation_rate': strategy_metrics['validation_rate'],
                'error_count': len(strategy_metrics['errors']),
                'final_balance': performance_report['basic_metrics']['final_balance'],
                'total_return': performance_report['basic_metrics']['total_return']
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'strategy_metrics': strategy.get_strategy_metrics() if 'strategy' in locals() else {}
        }


# Example usage for testing
if __name__ == "__main__":
    # Run the strategy test
    test_results = run_strategy_test()
    
    if test_results['success']:
        print("✅ Moving Average Crossover Test Strategy: SUCCESS")
        print(f"Total Signals: {test_results['test_summary']['total_signals']}")
        print(f"Execution Rate: {test_results['test_summary']['execution_rate']:.2%}")
        print(f"Final Balance: ${test_results['test_summary']['final_balance']:,.2f}")
        print(f"Total Return: {test_results['test_summary']['total_return']:.2%}")
    else:
        print("❌ Moving Average Crossover Test Strategy: FAILED")
        print(f"Error: {test_results['error']}")