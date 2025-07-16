"""
RSI Mean Reversion Testing Strategy
==================================

A comprehensive testing strategy implementing RSI mean reversion logic
with advanced signal validation, risk management, and educational features.

Purpose:
- Validate oscillator-based trading strategies
- Demonstrate mean reversion trading concepts
- Test RSI indicator implementation
- Provide educational reference for counter-trend strategies

Key Features:
- RSI overbought/oversold detection
- Multi-timeframe confirmation
- Dynamic position sizing
- Comprehensive risk management
- Educational data collection

Author: Backtesting Framework
Version: 1.0
Last Updated: July 2025
"""

from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import json
import numpy as np

from core.data_structures import (
    Tick, Order, Position, OrderSide, BacktestConfig
)
from strategy.strategy_interface import (
    TradingStrategy, StrategyConfig, StrategySignal, 
    RSI, MovingAverage, TechnicalIndicator
)


logger = logging.getLogger(__name__)


class RSIMeanReversionTestStrategy(TradingStrategy):
    """
    Professional RSI Mean Reversion Strategy for Testing
    
    This strategy implements a sophisticated mean reversion system using RSI with:
    - Multi-level RSI analysis (oversold/overbought with gradients)
    - Trend filter using moving averages
    - Dynamic position sizing based on RSI extremes
    - Comprehensive risk management
    - Educational data collection and analysis
    
    Trading Logic:
    - Buy when RSI < oversold_level with trend confirmation
    - Sell when RSI > overbought_level with trend confirmation
    - Exit when RSI returns to neutral zone or risk management triggers
    
    Educational Features:
    - RSI calculation validation
    - Mean reversion concept demonstration
    - Risk management in counter-trend trading
    - Signal quality assessment
    """
    
    def __init__(self, config: StrategyConfig, backtest_config: BacktestConfig):
        """
        Initialize the RSI Mean Reversion Test Strategy.
        
        Args:
            config: Strategy configuration with RSI parameters
            backtest_config: Backtesting configuration
        """
        super().__init__(config, backtest_config)
        
        # RSI parameters with validation
        self.rsi_period = self._validate_parameter('rsi_period', 14, min_val=7, max_val=21)
        self.oversold_level = self._validate_parameter('oversold_level', 30, min_val=10, max_val=40)
        self.overbought_level = self._validate_parameter('overbought_level', 70, min_val=60, max_val=90)
        
        # Extreme levels for stronger signals
        self.extreme_oversold = self._validate_parameter('extreme_oversold', 20, min_val=5, max_val=25)
        self.extreme_overbought = self._validate_parameter('extreme_overbought', 80, min_val=75, max_val=95)
        
        # Neutral zone for exits
        self.neutral_lower = self._validate_parameter('neutral_lower', 40, min_val=30, max_val=45)
        self.neutral_upper = self._validate_parameter('neutral_upper', 60, min_val=55, max_val=70)
        
        # Trend filter parameters
        self.use_trend_filter = config.parameters.get('use_trend_filter', True)
        self.trend_ma_period = self._validate_parameter('trend_ma_period', 50, min_val=20, max_val=100)
        
        # Signal confirmation parameters
        self.confirmation_bars = self._validate_parameter('confirmation_bars', 2, min_val=1, max_val=5)
        self.min_signal_strength = self._validate_parameter('min_signal_strength', 0.4, min_val=0.1, max_val=1.0)
        
        # Market condition filters
        self.min_spread_pips = self._validate_parameter('min_spread_pips', 0.5, min_val=0.1, max_val=5.0)
        self.max_spread_pips = self._validate_parameter('max_spread_pips', 3.0, min_val=1.0, max_val=15.0)
        
        # Technical indicators (per symbol)
        self.rsi: Dict[str, RSI] = {}
        self.trend_ma: Dict[str, MovingAverage] = {}
        self.price_buffers: Dict[str, List[float]] = {}
        
        # Strategy state tracking
        self.position_states: Dict[str, Dict[str, Any]] = {}
        self.signal_history: List[StrategySignal] = []
        self.rsi_history: Dict[str, List[float]] = {}
        
        # Performance and validation metrics
        self.metrics = {
            'signals_generated': 0,
            'signals_executed': 0,
            'oversold_signals': 0,
            'overbought_signals': 0,
            'extreme_signals': 0,
            'trend_filtered_signals': 0,
            'neutral_zone_exits': 0,
            'spread_rejections': 0,
            'rsi_calculation_errors': 0,
            'validation_failures': 0,
            'errors': []
        }
        
        # Educational data collection
        self.educational_data = {
            'rsi_examples': [],
            'mean_reversion_examples': [],
            'risk_management_examples': [],
            'signal_validation_examples': [],
            'trend_filter_examples': []
        }
        
        # Advanced analytics
        self.analytics = {
            'rsi_distribution': {'oversold': 0, 'neutral': 0, 'overbought': 0},
            'signal_quality_by_rsi_level': {},
            'mean_reversion_success_rate': 0.0,
            'average_reversal_time': 0.0
        }
        
        logger.info(f"RSI Mean Reversion Strategy initialized: RSI({self.rsi_period}), "
                   f"Levels: {self.oversold_level}/{self.overbought_level}")
    
    def _validate_parameter(self, param_name: str, default: float, min_val: float, max_val: float) -> float:
        """Validate and constrain strategy parameters."""
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
        RSI strategies require careful initialization of the RSI indicator
        and proper handling of the initial warm-up period.
        """
        try:
            # Initialize indicators and state for each symbol
            for symbol in self.backtest_config.symbols:
                # Core indicators
                self.rsi[symbol] = RSI(self.rsi_period)
                self.trend_ma[symbol] = MovingAverage(self.trend_ma_period)
                
                # Data buffers
                self.price_buffers[symbol] = []
                self.rsi_history[symbol] = []
                
                # Position and signal state
                self.position_states[symbol] = {
                    'position': None,
                    'entry_time': None,
                    'entry_price': None,
                    'entry_rsi': None,
                    'last_signal_type': None,
                    'last_signal_time': None,
                    'signal_count': 0,
                    'reversal_start_time': None,
                    'max_adverse_rsi': None
                }
                
                # Analytics initialization
                self.analytics['signal_quality_by_rsi_level'][symbol] = {
                    'extreme_oversold': {'count': 0, 'wins': 0},
                    'oversold': {'count': 0, 'wins': 0},
                    'overbought': {'count': 0, 'wins': 0},
                    'extreme_overbought': {'count': 0, 'wins': 0}
                }
                
                logger.info(f"Initialized RSI and trend filter for {symbol}")
            
            logger.info(f"RSI Mean Reversion strategy initialized for {len(self.backtest_config.symbols)} symbols")
            
        except Exception as e:
            error_msg = f"Strategy initialization failed: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            raise
    
    def on_tick(self, tick: Tick) -> List[StrategySignal]:
        """
        Process each tick and generate RSI-based mean reversion signals.
        
        Educational Note:
        This method demonstrates how RSI values are interpreted for
        mean reversion trading, including signal confirmation and filtering.
        
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
            
            # Get current RSI value
            current_rsi = self.rsi[tick.symbol].get_value()
            if current_rsi is None:
                return signals
            
            # Update RSI analytics
            self._update_rsi_analytics(tick.symbol, current_rsi)
            
            # Generate signals based on RSI levels
            entry_signal = self._generate_rsi_signal(tick, current_rsi)
            if entry_signal:
                signals.append(entry_signal)
            
            # Check for exit conditions
            exit_signal = self._check_exit_conditions(tick, current_rsi)
            if exit_signal:
                signals.append(exit_signal)
            
            # Validate and process all signals
            validated_signals = []
            for signal in signals:
                if self._validate_signal(signal):
                    validated_signals.append(signal)
                    self.signal_history.append(signal)
                else:
                    self.metrics['validation_failures'] += 1
            
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
        Check if market conditions are suitable for RSI trading.
        
        Educational Note:
        RSI strategies work best in ranging markets with reasonable spreads.
        This method filters out unsuitable market conditions.
        """
        # Check spread conditions
        if not self._is_spread_acceptable(tick):
            self.metrics['spread_rejections'] += 1
            return False
        
        return True
    
    def _is_spread_acceptable(self, tick: Tick) -> bool:
        """Validate spread conditions for RSI trading."""
        try:
            pip_value = self._get_pip_value(tick.symbol)
            spread_pips = tick.spread / pip_value
            
            is_acceptable = self.min_spread_pips <= spread_pips <= self.max_spread_pips
            
            if not is_acceptable:
                logger.debug(f"Spread rejected for {tick.symbol}: {spread_pips:.2f} pips")
            
            return is_acceptable
            
        except Exception as e:
            logger.error(f"Error checking spread for {tick.symbol}: {str(e)}")
            return False
    
    def _update_indicators(self, tick: Tick) -> None:
        """
        Update RSI and trend filter indicators.
        
        Educational Note:
        RSI calculation requires proper price data handling and
        error management for accurate results.
        """
        try:
            symbol = tick.symbol
            mid_price = tick.mid
            
            # Update price buffer
            self.price_buffers[symbol].append(mid_price)
            if len(self.price_buffers[symbol]) > 100:
                self.price_buffers[symbol].pop(0)
            
            # Update RSI indicator
            rsi_value = self.rsi[symbol].update(mid_price)
            
            # Update trend filter
            self.trend_ma[symbol].update(mid_price)
            
            # Store RSI history
            if rsi_value is not None:
                self.rsi_history[symbol].append(rsi_value)
                if len(self.rsi_history[symbol]) > 100:
                    self.rsi_history[symbol].pop(0)
            
            logger.debug(f"Updated indicators for {symbol}: RSI={rsi_value:.2f if rsi_value else 'N/A'}")
            
        except Exception as e:
            error_msg = f"Error updating indicators for {tick.symbol}: {str(e)}"
            logger.error(error_msg)
            self.metrics['rsi_calculation_errors'] += 1
            self.metrics['errors'].append(error_msg)
    
    def _are_indicators_ready(self, symbol: str) -> bool:
        """Check if RSI indicator has sufficient data."""
        try:
            rsi_ready = self.rsi[symbol].is_ready
            trend_ready = self.trend_ma[symbol].is_ready if self.use_trend_filter else True
            
            return rsi_ready and trend_ready
            
        except Exception as e:
            logger.error(f"Error checking indicator readiness for {symbol}: {str(e)}")
            return False
    
    def _update_rsi_analytics(self, symbol: str, rsi_value: float) -> None:
        """Update RSI distribution analytics."""
        try:
            if rsi_value <= self.oversold_level:
                self.analytics['rsi_distribution']['oversold'] += 1
            elif rsi_value >= self.overbought_level:
                self.analytics['rsi_distribution']['overbought'] += 1
            else:
                self.analytics['rsi_distribution']['neutral'] += 1
            
        except Exception as e:
            logger.error(f"Error updating RSI analytics: {str(e)}")
    
    def _generate_rsi_signal(self, tick: Tick, current_rsi: float) -> Optional[StrategySignal]:
        """
        Generate RSI-based mean reversion signals.
        
        Educational Note:
        This method demonstrates how RSI levels are interpreted for
        mean reversion trading, including signal strength calculation.
        """
        try:
            symbol = tick.symbol
            position = self.position_states[symbol]['position']
            
            # Don't generate entry signals if we already have a position
            if position and not position.is_closed:
                return None
            
            # Check for oversold condition (potential buy signal)
            if current_rsi <= self.oversold_level:
                if self._confirm_oversold_signal(tick, current_rsi):
                    return self._create_oversold_buy_signal(tick, current_rsi)
            
            # Check for overbought condition (potential sell signal)
            elif current_rsi >= self.overbought_level:
                if self._confirm_overbought_signal(tick, current_rsi):
                    return self._create_overbought_sell_signal(tick, current_rsi)
            
            return None
            
        except Exception as e:
            error_msg = f"Error generating RSI signal for {tick.symbol}: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            return None
    
    def _confirm_oversold_signal(self, tick: Tick, current_rsi: float) -> bool:
        """
        Confirm oversold signal with additional filters.
        
        Educational Note:
        This method demonstrates how to add confirmation filters
        to improve signal quality in mean reversion strategies.
        """
        try:
            symbol = tick.symbol
            
            # Check trend filter if enabled
            if self.use_trend_filter:
                trend_ma = self.trend_ma[symbol].get_value()
                if trend_ma and tick.mid < trend_ma * 0.995:  # Price significantly below trend
                    self.metrics['trend_filtered_signals'] += 1
                    return False
            
            # Check RSI momentum (RSI should be stabilizing or turning up)
            rsi_history = self.rsi_history[symbol]
            if len(rsi_history) >= 3:
                recent_rsi_slope = rsi_history[-1] - rsi_history[-3]
                if recent_rsi_slope < -5:  # RSI still falling rapidly
                    return False
            
            # Check for recent similar signals
            if self._has_recent_signal(symbol, 'BUY', 300):  # 5 minutes
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error confirming oversold signal: {str(e)}")
            return False
    
    def _confirm_overbought_signal(self, tick: Tick, current_rsi: float) -> bool:
        """
        Confirm overbought signal with additional filters.
        
        Educational Note:
        Similar to oversold confirmation but for overbought conditions.
        """
        try:
            symbol = tick.symbol
            
            # Check trend filter if enabled
            if self.use_trend_filter:
                trend_ma = self.trend_ma[symbol].get_value()
                if trend_ma and tick.mid > trend_ma * 1.005:  # Price significantly above trend
                    self.metrics['trend_filtered_signals'] += 1
                    return False
            
            # Check RSI momentum (RSI should be stabilizing or turning down)
            rsi_history = self.rsi_history[symbol]
            if len(rsi_history) >= 3:
                recent_rsi_slope = rsi_history[-1] - rsi_history[-3]
                if recent_rsi_slope > 5:  # RSI still rising rapidly
                    return False
            
            # Check for recent similar signals
            if self._has_recent_signal(symbol, 'SELL', 300):  # 5 minutes
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error confirming overbought signal: {str(e)}")
            return False
    
    def _has_recent_signal(self, symbol: str, signal_type: str, seconds: int) -> bool:
        """Check if similar signal was recently generated."""
        try:
            last_signal_time = self.position_states[symbol]['last_signal_time']
            if last_signal_time is None:
                return False
            
            time_diff = (datetime.now() - last_signal_time).total_seconds()
            return time_diff < seconds
            
        except Exception as e:
            logger.error(f"Error checking recent signals: {str(e)}")
            return False
    
    def _create_oversold_buy_signal(self, tick: Tick, current_rsi: float) -> Optional[StrategySignal]:
        """
        Create validated buy signal for oversold conditions.
        
        Educational Note:
        This method demonstrates how to create buy signals with
        appropriate position sizing and risk management for RSI strategies.
        """
        try:
            symbol = tick.symbol
            
            # Calculate signal strength based on RSI level
            strength = self._calculate_oversold_strength(current_rsi)
            
            if strength < self.min_signal_strength:
                return None
            
            # Calculate position size based on RSI extremity
            position_size = self._calculate_rsi_position_size(tick, current_rsi, 'BUY')
            
            # Create signal with comprehensive metadata
            signal = self.create_signal(
                tick=tick,
                signal_type='BUY',
                strength=strength,
                quantity=position_size,
                metadata={
                    'rsi_value': current_rsi,
                    'rsi_level': self._get_rsi_level(current_rsi),
                    'signal_quality': 'EXTREME' if current_rsi <= self.extreme_oversold else 'STANDARD',
                    'entry_reason': 'RSI_OVERSOLD_MEAN_REVERSION',
                    'trend_ma': self.trend_ma[symbol].get_value(),
                    'price_vs_trend': tick.mid / self.trend_ma[symbol].get_value() if self.trend_ma[symbol].get_value() else 1.0,
                    'rsi_momentum': self._calculate_rsi_momentum(symbol),
                    'strategy_version': '1.0'
                }
            )
            
            # Update position state
            self.position_states[symbol]['last_signal_type'] = 'BUY'
            self.position_states[symbol]['last_signal_time'] = tick.timestamp
            self.position_states[symbol]['signal_count'] += 1
            self.position_states[symbol]['reversal_start_time'] = tick.timestamp
            
            # Update metrics
            self.metrics['oversold_signals'] += 1
            if current_rsi <= self.extreme_oversold:
                self.metrics['extreme_signals'] += 1
            
            # Record educational example
            self._record_mean_reversion_example(signal, 'OVERSOLD', current_rsi)
            
            logger.info(f"BUY signal created for {symbol}: RSI={current_rsi:.2f}, "
                       f"strength={strength:.2f}, size={position_size:.2f}")
            
            return signal
            
        except Exception as e:
            error_msg = f"Error creating oversold buy signal: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            return None
    
    def _create_overbought_sell_signal(self, tick: Tick, current_rsi: float) -> Optional[StrategySignal]:
        """
        Create validated sell signal for overbought conditions.
        
        Educational Note:
        This method mirrors the buy signal creation but for overbought conditions.
        """
        try:
            symbol = tick.symbol
            
            # Calculate signal strength based on RSI level
            strength = self._calculate_overbought_strength(current_rsi)
            
            if strength < self.min_signal_strength:
                return None
            
            # Calculate position size based on RSI extremity
            position_size = self._calculate_rsi_position_size(tick, current_rsi, 'SELL')
            
            # Create signal with comprehensive metadata
            signal = self.create_signal(
                tick=tick,
                signal_type='SELL',
                strength=strength,
                quantity=position_size,
                metadata={
                    'rsi_value': current_rsi,
                    'rsi_level': self._get_rsi_level(current_rsi),
                    'signal_quality': 'EXTREME' if current_rsi >= self.extreme_overbought else 'STANDARD',
                    'entry_reason': 'RSI_OVERBOUGHT_MEAN_REVERSION',
                    'trend_ma': self.trend_ma[symbol].get_value(),
                    'price_vs_trend': tick.mid / self.trend_ma[symbol].get_value() if self.trend_ma[symbol].get_value() else 1.0,
                    'rsi_momentum': self._calculate_rsi_momentum(symbol),
                    'strategy_version': '1.0'
                }
            )
            
            # Update position state
            self.position_states[symbol]['last_signal_type'] = 'SELL'
            self.position_states[symbol]['last_signal_time'] = tick.timestamp
            self.position_states[symbol]['signal_count'] += 1
            self.position_states[symbol]['reversal_start_time'] = tick.timestamp
            
            # Update metrics
            self.metrics['overbought_signals'] += 1
            if current_rsi >= self.extreme_overbought:
                self.metrics['extreme_signals'] += 1
            
            # Record educational example
            self._record_mean_reversion_example(signal, 'OVERBOUGHT', current_rsi)
            
            logger.info(f"SELL signal created for {symbol}: RSI={current_rsi:.2f}, "
                       f"strength={strength:.2f}, size={position_size:.2f}")
            
            return signal
            
        except Exception as e:
            error_msg = f"Error creating overbought sell signal: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            return None
    
    def _calculate_oversold_strength(self, rsi_value: float) -> float:
        """Calculate signal strength for oversold conditions."""
        try:
            if rsi_value <= self.extreme_oversold:
                # Extreme oversold: very high strength
                return 0.9 + (0.1 * (self.extreme_oversold - rsi_value) / self.extreme_oversold)
            else:
                # Standard oversold: moderate strength
                base_strength = 0.5
                oversold_factor = (self.oversold_level - rsi_value) / self.oversold_level
                return base_strength + (0.3 * oversold_factor)
            
        except Exception as e:
            logger.error(f"Error calculating oversold strength: {str(e)}")
            return 0.5
    
    def _calculate_overbought_strength(self, rsi_value: float) -> float:
        """Calculate signal strength for overbought conditions."""
        try:
            if rsi_value >= self.extreme_overbought:
                # Extreme overbought: very high strength
                return 0.9 + (0.1 * (rsi_value - self.extreme_overbought) / (100 - self.extreme_overbought))
            else:
                # Standard overbought: moderate strength
                base_strength = 0.5
                overbought_factor = (rsi_value - self.overbought_level) / (100 - self.overbought_level)
                return base_strength + (0.3 * overbought_factor)
            
        except Exception as e:
            logger.error(f"Error calculating overbought strength: {str(e)}")
            return 0.5
    
    def _calculate_rsi_position_size(self, tick: Tick, rsi_value: float, signal_type: str) -> float:
        """
        Calculate position size based on RSI extremity.
        
        Educational Note:
        This method demonstrates how to adjust position sizing
        based on signal strength in mean reversion strategies.
        """
        try:
            base_size = self.config.risk_management.get('max_position_size', 1.0)
            
            # Calculate RSI extremity factor
            if signal_type == 'BUY':
                extremity = max(0, (self.oversold_level - rsi_value) / self.oversold_level)
            else:
                extremity = max(0, (rsi_value - self.overbought_level) / (100 - self.overbought_level))
            
            # Adjust size based on extremity (more extreme = larger position)
            size_multiplier = 0.5 + (0.5 * extremity)
            
            # Apply multiplier
            adjusted_size = base_size * size_multiplier
            
            return min(adjusted_size, base_size)
            
        except Exception as e:
            logger.error(f"Error calculating RSI position size: {str(e)}")
            return 1.0
    
    def _get_rsi_level(self, rsi_value: float) -> str:
        """Get RSI level classification."""
        if rsi_value <= self.extreme_oversold:
            return 'EXTREME_OVERSOLD'
        elif rsi_value <= self.oversold_level:
            return 'OVERSOLD'
        elif rsi_value >= self.extreme_overbought:
            return 'EXTREME_OVERBOUGHT'
        elif rsi_value >= self.overbought_level:
            return 'OVERBOUGHT'
        else:
            return 'NEUTRAL'
    
    def _calculate_rsi_momentum(self, symbol: str) -> float:
        """Calculate RSI momentum for signal confirmation."""
        try:
            rsi_history = self.rsi_history[symbol]
            if len(rsi_history) < 3:
                return 0.0
            
            # Calculate 3-period RSI slope
            return rsi_history[-1] - rsi_history[-3]
            
        except Exception as e:
            logger.error(f"Error calculating RSI momentum: {str(e)}")
            return 0.0
    
    def _check_exit_conditions(self, tick: Tick, current_rsi: float) -> Optional[StrategySignal]:
        """
        Check for position exit conditions based on RSI.
        
        Educational Note:
        This method demonstrates systematic exit logic for mean reversion strategies.
        """
        try:
            symbol = tick.symbol
            position = self.position_states[symbol]['position']
            
            if not position or position.is_closed:
                return None
            
            entry_rsi = self.position_states[symbol]['entry_rsi']
            exit_reason = None
            
            # Exit long position when RSI returns to neutral or becomes overbought
            if position.is_long:
                if current_rsi >= self.neutral_upper:
                    exit_reason = "RSI_NEUTRAL_ZONE_EXIT"
                elif current_rsi >= self.overbought_level:
                    exit_reason = "RSI_OVERBOUGHT_EXIT"
            
            # Exit short position when RSI returns to neutral or becomes oversold
            elif position.is_short:
                if current_rsi <= self.neutral_lower:
                    exit_reason = "RSI_NEUTRAL_ZONE_EXIT"
                elif current_rsi <= self.oversold_level:
                    exit_reason = "RSI_OVERSOLD_EXIT"
            
            if exit_reason:
                # Calculate reversal time
                reversal_time = 0
                if self.position_states[symbol]['reversal_start_time']:
                    reversal_time = (tick.timestamp - 
                                   self.position_states[symbol]['reversal_start_time']).total_seconds()
                
                # Create exit signal
                signal = StrategySignal(
                    timestamp=tick.timestamp,
                    symbol=symbol,
                    signal_type='CLOSE',
                    strength=1.0,
                    price=tick.mid,
                    quantity=abs(position.quantity),
                    metadata={
                        'exit_reason': exit_reason,
                        'entry_rsi': entry_rsi,
                        'exit_rsi': current_rsi,
                        'rsi_change': current_rsi - entry_rsi if entry_rsi else 0,
                        'reversal_time_seconds': reversal_time,
                        'position_pnl': position.unrealized_pnl,
                        'mean_reversion_success': self._assess_mean_reversion_success(position, current_rsi),
                        'strategy_version': '1.0'
                    }
                )
                
                # Update metrics
                if 'NEUTRAL_ZONE' in exit_reason:
                    self.metrics['neutral_zone_exits'] += 1
                
                # Record educational example
                self._record_exit_example(signal, exit_reason, position, current_rsi)
                
                logger.info(f"EXIT signal created for {symbol}: {exit_reason}, "
                           f"RSI={current_rsi:.2f}")
                
                return signal
            
            return None
            
        except Exception as e:
            error_msg = f"Error checking exit conditions: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
            return None
    
    def _assess_mean_reversion_success(self, position: Position, exit_rsi: float) -> bool:
        """Assess if mean reversion was successful."""
        try:
            # Mean reversion is successful if:
            # 1. Position is profitable
            # 2. RSI moved from extreme to neutral
            return position.unrealized_pnl > 0 and 40 <= exit_rsi <= 60
            
        except Exception as e:
            logger.error(f"Error assessing mean reversion success: {str(e)}")
            return False
    
    def _validate_signal(self, signal: StrategySignal) -> bool:
        """Validate RSI signal before execution."""
        try:
            # Standard signal validation
            if signal.strength < 0.1 or signal.strength > 1.0:
                return False
            
            if signal.quantity <= 0:
                return False
            
            if signal.price <= 0:
                return False
            
            # RSI-specific validation
            if signal.metadata and 'rsi_value' in signal.metadata:
                rsi_value = signal.metadata['rsi_value']
                if not (0 <= rsi_value <= 100):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {str(e)}")
            return False
    
    def on_order_filled(self, order: Order) -> None:
        """Handle order execution events."""
        try:
            self.metrics['signals_executed'] += 1
            symbol = order.symbol
            
            # Update position state
            if symbol in self.position_states:
                self.position_states[symbol]['entry_time'] = order.filled_at
                self.position_states[symbol]['entry_price'] = order.avg_fill_price
                
                # Store entry RSI value
                current_rsi = self.rsi[symbol].get_value()
                self.position_states[symbol]['entry_rsi'] = current_rsi
                
                # Track maximum adverse RSI for risk analysis
                self.position_states[symbol]['max_adverse_rsi'] = current_rsi
            
            logger.info(f"RSI Order filled: {order.side.value} {order.filled_quantity} lots "
                       f"of {symbol} at {order.avg_fill_price:.5f}")
            
        except Exception as e:
            error_msg = f"Error handling order fill: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
    
    def on_position_update(self, position: Position) -> None:
        """Handle position updates."""
        try:
            symbol = position.symbol
            
            # Update position tracking
            self.position_states[symbol]['position'] = position
            
            # Update performance metrics
            if position.is_closed:
                # Position closed - analyze mean reversion success
                entry_rsi = self.position_states[symbol]['entry_rsi']
                if entry_rsi:
                    # Update analytics
                    rsi_level = self._get_rsi_level(entry_rsi)
                    if rsi_level in self.analytics['signal_quality_by_rsi_level'][symbol]:
                        self.analytics['signal_quality_by_rsi_level'][symbol][rsi_level]['count'] += 1
                        if position.total_pnl > 0:
                            self.analytics['signal_quality_by_rsi_level'][symbol][rsi_level]['wins'] += 1
                
                # Update win/loss counts
                if position.total_pnl > 0:
                    self.winning_signals += 1
                else:
                    self.losing_signals += 1
                
                # Reset position state
                self.position_states[symbol]['entry_time'] = None
                self.position_states[symbol]['entry_price'] = None
                self.position_states[symbol]['entry_rsi'] = None
                
                logger.info(f"RSI Position closed: {symbol} P&L: {position.total_pnl:.2f}")
            else:
                # Position updated - track adverse RSI movement
                current_rsi = self.rsi[symbol].get_value()
                if current_rsi and self.position_states[symbol]['max_adverse_rsi']:
                    if position.is_long and current_rsi < self.position_states[symbol]['max_adverse_rsi']:
                        self.position_states[symbol]['max_adverse_rsi'] = current_rsi
                    elif position.is_short and current_rsi > self.position_states[symbol]['max_adverse_rsi']:
                        self.position_states[symbol]['max_adverse_rsi'] = current_rsi
                
                logger.info(f"RSI Position updated: {symbol} P&L: {position.unrealized_pnl:.2f}")
            
        except Exception as e:
            error_msg = f"Error handling position update: {str(e)}"
            logger.error(error_msg)
            self.metrics['errors'].append(error_msg)
    
    def _record_mean_reversion_example(self, signal: StrategySignal, condition: str, rsi_value: float) -> None:
        """Record mean reversion example for educational purposes."""
        example = {
            'condition': condition,
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'rsi_value': rsi_value,
            'signal_type': signal.signal_type,
            'strength': signal.strength,
            'quantity': signal.quantity,
            'price': signal.price,
            'metadata': signal.metadata
        }
        
        self.educational_data['mean_reversion_examples'].append(example)
    
    def _record_exit_example(self, signal: StrategySignal, exit_reason: str, position: Position, exit_rsi: float) -> None:
        """Record exit example for educational purposes."""
        example = {
            'exit_reason': exit_reason,
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'exit_rsi': exit_rsi,
            'entry_rsi': signal.metadata.get('entry_rsi', 0),
            'position_pnl': position.unrealized_pnl,
            'reversal_time': signal.metadata.get('reversal_time_seconds', 0),
            'mean_reversion_success': signal.metadata.get('mean_reversion_success', False)
        }
        
        self.educational_data['risk_management_examples'].append(example)
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get comprehensive RSI strategy metrics."""
        base_metrics = self.get_performance_metrics()
        
        # Calculate RSI-specific metrics
        total_signals = self.metrics['signals_generated']
        execution_rate = self.metrics['signals_executed'] / max(total_signals, 1)
        
        # Calculate mean reversion success rate
        successful_reversions = sum(1 for example in self.educational_data['risk_management_examples'] 
                                  if example['mean_reversion_success'])
        total_reversions = len(self.educational_data['risk_management_examples'])
        mean_reversion_rate = successful_reversions / max(total_reversions, 1)
        
        strategy_metrics = {
            **base_metrics,
            **self.metrics,
            'execution_rate': execution_rate,
            'mean_reversion_success_rate': mean_reversion_rate,
            'rsi_distribution': self.analytics['rsi_distribution'],
            'signal_quality_by_rsi_level': self.analytics['signal_quality_by_rsi_level'],
            'strategy_type': 'RSIMeanReversion',
            'parameters': {
                'rsi_period': self.rsi_period,
                'oversold_level': self.oversold_level,
                'overbought_level': self.overbought_level,
                'extreme_oversold': self.extreme_oversold,
                'extreme_overbought': self.extreme_overbought,
                'use_trend_filter': self.use_trend_filter,
                'trend_ma_period': self.trend_ma_period
            },
            'educational_data': self.educational_data
        }
        
        return strategy_metrics
    
    def finalize(self) -> None:
        """Finalize RSI strategy execution."""
        super().finalize()
        
        # Calculate final analytics
        total_signals = self.metrics['signals_generated']
        
        # Update analytics
        self.analytics['mean_reversion_success_rate'] = (
            sum(1 for example in self.educational_data['risk_management_examples'] 
                if example['mean_reversion_success']) / 
            max(len(self.educational_data['risk_management_examples']), 1)
        )
        
        # Log final statistics
        logger.info(f"RSI Strategy finalized - Total signals: {total_signals}, "
                   f"Oversold: {self.metrics['oversold_signals']}, "
                   f"Overbought: {self.metrics['overbought_signals']}, "
                   f"Mean reversion rate: {self.analytics['mean_reversion_success_rate']:.2%}")
        
        # Update final performance metrics
        self.state.performance.update({
            'final_rsi_metrics': self.get_strategy_metrics(),
            'rsi_signal_distribution': {
                'oversold': self.metrics['oversold_signals'],
                'overbought': self.metrics['overbought_signals'],
                'extreme': self.metrics['extreme_signals']
            },
            'mean_reversion_analytics': self.analytics
        })


def create_rsi_test_strategy_config() -> StrategyConfig:
    """Create standardized RSI configuration for testing."""
    return StrategyConfig(
        name="RSI_Mean_Reversion_Test_Strategy",
        description="Professional RSI Mean Reversion Strategy for Framework Testing",
        parameters={
            'rsi_period': 14,
            'oversold_level': 30,
            'overbought_level': 70,
            'extreme_oversold': 20,
            'extreme_overbought': 80,
            'neutral_lower': 40,
            'neutral_upper': 60,
            'use_trend_filter': True,
            'trend_ma_period': 50,
            'confirmation_bars': 2,
            'min_signal_strength': 0.4,
            'min_spread_pips': 0.5,
            'max_spread_pips': 3.0
        },
        risk_management={
            'max_position_size': 1.0,
            'stop_loss_pips': 40,
            'take_profit_pips': 80,
            'max_daily_loss': 2500,
            'max_drawdown': 0.12,
            'risk_per_trade': 600
        }
    )


def run_rsi_strategy_test(data_path: str = "./data") -> Dict[str, Any]:
    """Run complete RSI strategy test with validation."""
    from engine.backtest_engine import BacktestEngine
    from analysis.performance_analyzer import PerformanceAnalyzer
    
    # Test configuration
    backtest_config = BacktestConfig(
        start_date=datetime(2025, 7, 3),
        end_date=datetime(2025, 7, 5),
        symbols=['EURUSD', 'EURJPY', 'GBPNZD'],
        
        # Execution settings
        spread_markup=0.5,
        slippage_model='linear',
        max_slippage=1.0,
        execution_delay_min=0.05,
        execution_delay_max=0.2,
        
        # Account settings
        initial_balance=100000.0,
        leverage=100.0,
        commission_per_lot=7.0,
        
        # Risk settings
        max_position_size=2.0,
        margin_requirement=0.01,
        
        # Data quality
        interpolate_missing_ticks=True,
        max_gap_seconds=30.0
    )
    
    try:
        # Create and run strategy
        strategy_config = create_rsi_test_strategy_config()
        strategy = RSIMeanReversionTestStrategy(strategy_config, backtest_config)
        
        engine = BacktestEngine(backtest_config, data_path)
        engine.add_strategy(strategy)
        
        result = engine.run_backtest()
        
        # Analyze results
        analyzer = PerformanceAnalyzer(result)
        performance_report = analyzer.generate_summary_report()
        
        # Get strategy metrics
        strategy_metrics = strategy.get_strategy_metrics()
        
        return {
            'success': True,
            'backtest_result': result,
            'performance_report': performance_report,
            'strategy_metrics': strategy_metrics,
            'test_summary': {
                'total_signals': strategy_metrics['signals_generated'],
                'oversold_signals': strategy_metrics['oversold_signals'],
                'overbought_signals': strategy_metrics['overbought_signals'],
                'mean_reversion_rate': strategy_metrics['mean_reversion_success_rate'],
                'execution_rate': strategy_metrics['execution_rate'],
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


# Example usage
if __name__ == "__main__":
    # Run the RSI strategy test
    test_results = run_rsi_strategy_test()
    
    if test_results['success']:
        print("✅ RSI Mean Reversion Test Strategy: SUCCESS")
        print(f"Total Signals: {test_results['test_summary']['total_signals']}")
        print(f"Oversold Signals: {test_results['test_summary']['oversold_signals']}")
        print(f"Overbought Signals: {test_results['test_summary']['overbought_signals']}")
        print(f"Mean Reversion Rate: {test_results['test_summary']['mean_reversion_rate']:.2%}")
        print(f"Final Balance: ${test_results['test_summary']['final_balance']:,.2f}")
        print(f"Total Return: {test_results['test_summary']['total_return']:.2%}")
    else:
        print("❌ RSI Mean Reversion Test Strategy: FAILED")
        print(f"Error: {test_results['error']}")