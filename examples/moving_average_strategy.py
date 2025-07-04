"""
Example Moving Average Crossover Strategy.
Demonstrates how to implement a trading strategy using the backtesting framework.
"""

from typing import List, Optional
import logging

from core.data_structures import Tick, Order, Position, OrderSide
from strategy.strategy_interface import (
    TradingStrategy, StrategyConfig, StrategySignal, 
    MovingAverage, BacktestConfig
)


logger = logging.getLogger(__name__)


class MovingAverageCrossoverStrategy(TradingStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Generates buy signals when fast MA crosses above slow MA,
    and sell signals when fast MA crosses below slow MA.
    """
    
    def __init__(self, config: StrategyConfig, backtest_config: BacktestConfig):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            config: Strategy configuration
            backtest_config: Backtesting configuration
        """
        super().__init__(config, backtest_config)
        
        # Strategy parameters
        self.fast_period = config.parameters.get('fast_period', 10)
        self.slow_period = config.parameters.get('slow_period', 20)
        self.min_spread_pips = config.parameters.get('min_spread_pips', 2)
        self.max_spread_pips = config.parameters.get('max_spread_pips', 5)
        
        # Technical indicators
        self.fast_ma: dict = {}  # By symbol
        self.slow_ma: dict = {}  # By symbol
        
        # State tracking
        self.last_signals: dict = {}  # By symbol
        self.current_positions: dict = {}  # By symbol
        
        logger.info(f"Moving Average strategy initialized: Fast={self.fast_period}, Slow={self.slow_period}")
    
    def initialize(self) -> None:
        """Initialize the strategy."""
        # Initialize indicators for each symbol
        for symbol in self.backtest_config.symbols:
            self.fast_ma[symbol] = MovingAverage(self.fast_period)
            self.slow_ma[symbol] = MovingAverage(self.slow_period)
            self.last_signals[symbol] = None
            self.current_positions[symbol] = None
        
        logger.info("Moving Average strategy initialized")
    
    def on_tick(self, tick: Tick) -> List[StrategySignal]:
        """
        Process a new tick and generate trading signals.
        
        Args:
            tick: New market tick
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Add tick to history
        self.add_tick_to_history(tick)
        
        # Check spread conditions
        if not self._is_spread_acceptable(tick):
            return signals
        
        # Update indicators
        mid_price = tick.mid
        fast_ma_value = self.fast_ma[tick.symbol].update(mid_price)
        slow_ma_value = self.slow_ma[tick.symbol].update(mid_price)
        
        # Check if indicators are ready
        if not (self.fast_ma[tick.symbol].is_ready and self.slow_ma[tick.symbol].is_ready):
            return signals
        
        # Get previous values for crossover detection
        fast_history = self.fast_ma[tick.symbol].get_history(2)
        slow_history = self.slow_ma[tick.symbol].get_history(2)
        
        if len(fast_history) < 2 or len(slow_history) < 2:
            return signals
        
        # Detect crossovers
        prev_fast = fast_history[-2]
        curr_fast = fast_history[-1]
        prev_slow = slow_history[-2]
        curr_slow = slow_history[-1]
        
        # Check for bullish crossover (fast MA crosses above slow MA)
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            if not self._has_position(tick.symbol, OrderSide.BUY):
                signal = self._create_buy_signal(tick)
                if signal:
                    signals.append(signal)
        
        # Check for bearish crossover (fast MA crosses below slow MA)
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            if not self._has_position(tick.symbol, OrderSide.SELL):
                signal = self._create_sell_signal(tick)
                if signal:
                    signals.append(signal)
        
        # Check for position exit conditions
        exit_signal = self._check_exit_conditions(tick)
        if exit_signal:
            signals.append(exit_signal)
        
        return signals
    
    def _is_spread_acceptable(self, tick: Tick) -> bool:
        """
        Check if the spread is acceptable for trading.
        
        Args:
            tick: Current tick
            
        Returns:
            True if spread is acceptable
        """
        spread_pips = tick.spread / self._get_pip_value(tick.symbol)
        return self.min_spread_pips <= spread_pips <= self.max_spread_pips
    
    def _has_position(self, symbol: str, side: OrderSide) -> bool:
        """
        Check if we have a position in the specified direction.
        
        Args:
            symbol: Symbol to check
            side: Position side
            
        Returns:
            True if position exists
        """
        position = self.current_positions.get(symbol)
        if not position or position.is_closed:
            return False
        
        if side == OrderSide.BUY:
            return position.is_long
        else:
            return position.is_short
    
    def _create_buy_signal(self, tick: Tick) -> Optional[StrategySignal]:
        """
        Create a buy signal.
        
        Args:
            tick: Current tick
            
        Returns:
            Buy signal or None
        """
        # Calculate position size
        risk_amount = self.config.risk_management.get('risk_per_trade', 100)
        position_size = self.calculate_position_size(tick, risk_amount)
        
        # Calculate signal strength based on MA separation
        fast_ma = self.fast_ma[tick.symbol].get_value()
        slow_ma = self.slow_ma[tick.symbol].get_value()
        
        if fast_ma is None or slow_ma is None:
            return None
        
        # Signal strength based on MA separation
        separation = abs(fast_ma - slow_ma) / self._get_pip_value(tick.symbol)
        strength = min(separation / 20.0, 1.0)  # Normalize to 0-1
        
        signal = self.create_signal(
            tick=tick,
            signal_type='BUY',
            strength=strength,
            quantity=position_size,
            metadata={
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'ma_separation_pips': separation,
                'entry_reason': 'MA_CROSSOVER_BULLISH'
            }
        )
        
        self.last_signals[tick.symbol] = signal
        return signal
    
    def _create_sell_signal(self, tick: Tick) -> Optional[StrategySignal]:
        """
        Create a sell signal.
        
        Args:
            tick: Current tick
            
        Returns:
            Sell signal or None
        """
        # Calculate position size
        risk_amount = self.config.risk_management.get('risk_per_trade', 100)
        position_size = self.calculate_position_size(tick, risk_amount)
        
        # Calculate signal strength based on MA separation
        fast_ma = self.fast_ma[tick.symbol].get_value()
        slow_ma = self.slow_ma[tick.symbol].get_value()
        
        if fast_ma is None or slow_ma is None:
            return None
        
        # Signal strength based on MA separation
        separation = abs(fast_ma - slow_ma) / self._get_pip_value(tick.symbol)
        strength = min(separation / 20.0, 1.0)  # Normalize to 0-1
        
        signal = self.create_signal(
            tick=tick,
            signal_type='SELL',
            strength=strength,
            quantity=position_size,
            metadata={
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'ma_separation_pips': separation,
                'entry_reason': 'MA_CROSSOVER_BEARISH'
            }
        )
        
        self.last_signals[tick.symbol] = signal
        return signal
    
    def _check_exit_conditions(self, tick: Tick) -> Optional[StrategySignal]:
        """
        Check if we should exit current position.
        
        Args:
            tick: Current tick
            
        Returns:
            Exit signal or None
        """
        position = self.current_positions.get(tick.symbol)
        if not position or position.is_closed:
            return None
        
        # Check for opposite crossover
        fast_history = self.fast_ma[tick.symbol].get_history(2)
        slow_history = self.slow_ma[tick.symbol].get_history(2)
        
        if len(fast_history) < 2 or len(slow_history) < 2:
            return None
        
        prev_fast = fast_history[-2]
        curr_fast = fast_history[-1]
        prev_slow = slow_history[-2]
        curr_slow = slow_history[-1]
        
        should_exit = False
        exit_reason = ""
        
        if position.is_long:
            # Exit long position on bearish crossover
            if prev_fast >= prev_slow and curr_fast < curr_slow:
                should_exit = True
                exit_reason = "MA_CROSSOVER_BEARISH"
        
        elif position.is_short:
            # Exit short position on bullish crossover
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                should_exit = True
                exit_reason = "MA_CROSSOVER_BULLISH"
        
        if should_exit:
            return StrategySignal(
                timestamp=tick.timestamp,
                symbol=tick.symbol,
                signal_type='CLOSE',
                strength=1.0,
                price=tick.mid,
                quantity=abs(position.quantity),
                metadata={'exit_reason': exit_reason}
            )
        
        return None
    
    def on_order_filled(self, order: Order) -> None:
        """
        Handle order fill events.
        
        Args:
            order: Filled order
        """
        logger.info(f"Order filled: {order.side.value} {order.filled_quantity} lots of {order.symbol} at {order.avg_fill_price:.5f}")
    
    def on_position_update(self, position: Position) -> None:
        """
        Handle position updates.
        
        Args:
            position: Updated position
        """
        # Update our position tracking
        self.current_positions[position.symbol] = position
        
        # Update performance metrics
        if position.is_closed:
            if position.total_pnl > 0:
                self.winning_signals += 1
            else:
                self.losing_signals += 1
        
        # Log position update
        if position.is_closed:
            logger.info(f"Position closed: {position.symbol} P&L: {position.total_pnl:.2f}")
        else:
            logger.info(f"Position updated: {position.symbol} {position.quantity} lots, Unrealized P&L: {position.unrealized_pnl:.2f}")
    
    def finalize(self) -> None:
        """Finalize the strategy."""
        super().finalize()
        
        # Calculate additional performance metrics
        self.state.performance.update({
            'fast_ma_period': self.fast_period,
            'slow_ma_period': self.slow_period,
            'signals_per_symbol': {symbol: len([s for s in self.state.signals if s.symbol == symbol]) for symbol in self.backtest_config.symbols}
        })
        
        logger.info("Moving Average strategy finalized")


class MovingAverageRSIStrategy(TradingStrategy):
    """
    Moving Average + RSI Strategy.
    
    Combines MA crossover with RSI confirmation for better signal quality.
    """
    
    def __init__(self, config: StrategyConfig, backtest_config: BacktestConfig):
        """Initialize the MA + RSI strategy."""
        super().__init__(config, backtest_config)
        
        # Strategy parameters
        self.fast_period = config.parameters.get('fast_period', 10)
        self.slow_period = config.parameters.get('slow_period', 20)
        self.rsi_period = config.parameters.get('rsi_period', 14)
        self.rsi_overbought = config.parameters.get('rsi_overbought', 70)
        self.rsi_oversold = config.parameters.get('rsi_oversold', 30)
        
        # Technical indicators
        self.fast_ma: dict = {}
        self.slow_ma: dict = {}
        self.rsi: dict = {}
        
        # State tracking
        self.current_positions: dict = {}
        
        logger.info(f"MA + RSI strategy initialized: MA({self.fast_period},{self.slow_period}), RSI({self.rsi_period})")
    
    def initialize(self) -> None:
        """Initialize the strategy."""
        from strategy.strategy_interface import RSI
        
        for symbol in self.backtest_config.symbols:
            self.fast_ma[symbol] = MovingAverage(self.fast_period)
            self.slow_ma[symbol] = MovingAverage(self.slow_period)
            self.rsi[symbol] = RSI(self.rsi_period)
            self.current_positions[symbol] = None
        
        logger.info("MA + RSI strategy initialized")
    
    def on_tick(self, tick: Tick) -> List[StrategySignal]:
        """Process tick and generate signals."""
        signals = []
        
        # Add tick to history
        self.add_tick_to_history(tick)
        
        # Update indicators
        mid_price = tick.mid
        fast_ma_value = self.fast_ma[tick.symbol].update(mid_price)
        slow_ma_value = self.slow_ma[tick.symbol].update(mid_price)
        rsi_value = self.rsi[tick.symbol].update(mid_price)
        
        # Check if all indicators are ready
        if not (self.fast_ma[tick.symbol].is_ready and 
                self.slow_ma[tick.symbol].is_ready and 
                self.rsi[tick.symbol].is_ready):
            return signals
        
        # Get previous values for crossover detection
        fast_history = self.fast_ma[tick.symbol].get_history(2)
        slow_history = self.slow_ma[tick.symbol].get_history(2)
        
        if len(fast_history) < 2 or len(slow_history) < 2:
            return signals
        
        # Detect crossovers
        prev_fast = fast_history[-2]
        curr_fast = fast_history[-1]
        prev_slow = slow_history[-2]
        curr_slow = slow_history[-1]
        
        # Check for bullish crossover with RSI confirmation
        if (prev_fast <= prev_slow and curr_fast > curr_slow and 
            rsi_value < self.rsi_overbought and 
            not self._has_position(tick.symbol, OrderSide.BUY)):
            
            signal = self._create_confirmed_buy_signal(tick, rsi_value)
            if signal:
                signals.append(signal)
        
        # Check for bearish crossover with RSI confirmation
        elif (prev_fast >= prev_slow and curr_fast < curr_slow and 
              rsi_value > self.rsi_oversold and 
              not self._has_position(tick.symbol, OrderSide.SELL)):
            
            signal = self._create_confirmed_sell_signal(tick, rsi_value)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _has_position(self, symbol: str, side: OrderSide) -> bool:
        """Check if we have a position in the specified direction."""
        position = self.current_positions.get(symbol)
        if not position or position.is_closed:
            return False
        
        if side == OrderSide.BUY:
            return position.is_long
        else:
            return position.is_short
    
    def _create_confirmed_buy_signal(self, tick: Tick, rsi_value: float) -> Optional[StrategySignal]:
        """Create a buy signal with RSI confirmation."""
        # Higher signal strength when RSI is more oversold
        rsi_strength = max(0, (self.rsi_oversold - rsi_value) / self.rsi_oversold)
        
        # Calculate position size
        risk_amount = self.config.risk_management.get('risk_per_trade', 100)
        position_size = self.calculate_position_size(tick, risk_amount)
        
        signal = self.create_signal(
            tick=tick,
            signal_type='BUY',
            strength=min(rsi_strength + 0.5, 1.0),  # Boost strength with RSI
            quantity=position_size,
            metadata={
                'rsi_value': rsi_value,
                'rsi_strength': rsi_strength,
                'entry_reason': 'MA_CROSSOVER_BULLISH_RSI_CONFIRMED'
            }
        )
        
        return signal
    
    def _create_confirmed_sell_signal(self, tick: Tick, rsi_value: float) -> Optional[StrategySignal]:
        """Create a sell signal with RSI confirmation."""
        # Higher signal strength when RSI is more overbought
        rsi_strength = max(0, (rsi_value - self.rsi_overbought) / (100 - self.rsi_overbought))
        
        # Calculate position size
        risk_amount = self.config.risk_management.get('risk_per_trade', 100)
        position_size = self.calculate_position_size(tick, risk_amount)
        
        signal = self.create_signal(
            tick=tick,
            signal_type='SELL',
            strength=min(rsi_strength + 0.5, 1.0),  # Boost strength with RSI
            quantity=position_size,
            metadata={
                'rsi_value': rsi_value,
                'rsi_strength': rsi_strength,
                'entry_reason': 'MA_CROSSOVER_BEARISH_RSI_CONFIRMED'
            }
        )
        
        return signal
    
    def on_order_filled(self, order: Order) -> None:
        """Handle order fill events."""
        logger.info(f"MA+RSI Order filled: {order.side.value} {order.filled_quantity} lots of {order.symbol}")
    
    def on_position_update(self, position: Position) -> None:
        """Handle position updates."""
        self.current_positions[position.symbol] = position
        
        if position.is_closed:
            if position.total_pnl > 0:
                self.winning_signals += 1
            else:
                self.losing_signals += 1