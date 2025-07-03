"""
Main Backtesting Engine for tick-by-tick simulation.
Coordinates all components and executes the complete backtesting process.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.data_structures import (
    Tick, Order, Position, Trade, BacktestConfig, BacktestResult,
    OrderType, OrderSide, MarketEvent
)
from data.dukascopy_loader import DukascopyDataLoader
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from strategy.strategy_interface import TradingStrategy, StrategySignal


logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Professional backtesting engine with tick-by-tick simulation.
    Provides precise execution modeling and comprehensive performance analysis.
    """
    
    def __init__(self, config: BacktestConfig, data_path: str):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Backtesting configuration
            data_path: Path to Dukascopy data directory
        """
        self.config = config
        self.data_path = data_path
        
        # Initialize components
        self.data_loader = DukascopyDataLoader(data_path, config)
        self.order_manager = OrderManager(config)
        self.position_manager = PositionManager(config, config.initial_balance)
        
        # Strategy management
        self.strategies: List[TradingStrategy] = []
        self.strategy_results: Dict[str, Any] = {}
        
        # Event tracking
        self.market_events: List[MarketEvent] = []
        self.current_tick: Optional[Tick] = None
        self.tick_count = 0
        self.processed_ticks = 0
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.execution_stats = {
            'total_ticks': 0,
            'orders_created': 0,
            'orders_filled': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'trades_completed': 0
        }
        
        # Event callbacks
        self.on_tick_processed: Optional[Callable[[Tick], None]] = None
        self.on_order_filled: Optional[Callable[[Order], None]] = None
        self.on_position_opened: Optional[Callable[[Position], None]] = None
        self.on_position_closed: Optional[Callable[[Position, Trade], None]] = None
        self.on_progress_update: Optional[Callable[[float, Dict[str, Any]], None]] = None
        
        # Setup internal event handlers
        self._setup_event_handlers()
        
        logger.info(f"Backtesting engine initialized for {len(config.symbols)} symbols")
    
    def _setup_event_handlers(self) -> None:
        """Setup internal event handlers between components."""
        # Order manager callbacks
        self.order_manager.on_order_filled = self._handle_order_filled
        self.order_manager.on_order_rejected = self._handle_order_rejected
        
        # Position manager callbacks
        self.position_manager.on_position_opened = self._handle_position_opened
        self.position_manager.on_position_closed = self._handle_position_closed
        self.position_manager.on_margin_call = self._handle_margin_call
    
    def add_strategy(self, strategy: TradingStrategy) -> None:
        """
        Add a trading strategy to the backtesting engine.
        
        Args:
            strategy: Trading strategy to add
        """
        # Setup strategy callbacks
        strategy.on_signal_generated = self._handle_strategy_signal
        strategy.on_position_opened = self._handle_strategy_position_opened
        strategy.on_position_closed = self._handle_strategy_position_closed
        
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.config.name}")
    
    def run_backtest(self) -> BacktestResult:
        """
        Execute the complete backtesting process.
        
        Returns:
            Comprehensive backtesting results
        """
        logger.info("Starting backtesting process...")
        self.start_time = datetime.now()
        
        # Initialize result object
        result = BacktestResult(
            config=self.config,
            backtest_start_time=self.start_time
        )
        
        try:
            # Validate data availability
            self._validate_data_availability()
            
            # Initialize strategies
            self._initialize_strategies()
            
            # Execute main backtesting loop
            self._execute_backtesting_loop()
            
            # Finalize strategies
            self._finalize_strategies()
            
            # Generate comprehensive results
            result = self._generate_results()
            
            logger.info("Backtesting completed successfully")
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            raise
        
        finally:
            self.end_time = datetime.now()
            if result:
                result.backtest_end_time = self.end_time
        
        return result
    
    def _validate_data_availability(self) -> None:
        """Validate that required data is available."""
        logger.info("Validating data availability...")
        
        for symbol in self.config.symbols:
            if not self.data_loader.verify_data_integrity(
                symbol, self.config.start_date, self.config.end_date
            ):
                raise ValueError(f"Insufficient data for {symbol}")
        
        logger.info("Data validation completed")
    
    def _initialize_strategies(self) -> None:
        """Initialize all trading strategies."""
        logger.info("Initializing trading strategies...")
        
        for strategy in self.strategies:
            strategy.initialize()
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    def _execute_backtesting_loop(self) -> None:
        """Execute the main backtesting loop with tick-by-tick simulation."""
        logger.info("Starting tick-by-tick simulation...")
        
        # Load tick data for all symbols
        tick_generator = self.data_loader.load_multiple_symbols(
            self.config.symbols,
            self.config.start_date,
            self.config.end_date
        )
        
        # Process each tick
        for tick in tick_generator:
            self._process_tick(tick)
            self.processed_ticks += 1
            
            # Update progress periodically
            if self.processed_ticks % 10000 == 0:
                self._update_progress()
        
        logger.info(f"Processed {self.processed_ticks} ticks")
    
    def _process_tick(self, tick: Tick) -> None:
        """
        Process a single tick through the entire system.
        
        Args:
            tick: Tick to process
        """
        self.current_tick = tick
        self.tick_count += 1
        
        # Update position manager with market data
        self.position_manager.update_market_data(tick)
        
        # Process pending orders
        executed_orders = self.order_manager.process_tick(tick)
        
        # Update strategies with tick data
        for strategy in self.strategies:
            strategy.add_tick_to_history(tick)
            
            # Generate signals
            signals = strategy.on_tick(tick)
            
            # Process signals
            for signal in signals:
                if strategy.validate_signal(signal):
                    self._process_strategy_signal(signal)
        
        # Create market event
        event = MarketEvent(
            timestamp=tick.timestamp,
            event_type='tick',
            symbol=tick.symbol,
            data={'tick': tick}
        )
        self.market_events.append(event)
        
        # Call callback if set
        if self.on_tick_processed:
            self.on_tick_processed(tick)
    
    def _process_strategy_signal(self, signal: StrategySignal) -> None:
        """
        Process a trading signal by creating appropriate orders.
        
        Args:
            signal: Trading signal to process
        """
        try:
            if signal.signal_type == 'BUY':
                order = self.order_manager.create_order(
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=signal.quantity,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
                self.execution_stats['orders_created'] += 1
                
            elif signal.signal_type == 'SELL':
                order = self.order_manager.create_order(
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=signal.quantity,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
                self.execution_stats['orders_created'] += 1
                
            elif signal.signal_type == 'CLOSE':
                # Close existing position
                position = self.position_manager.get_position(signal.symbol)
                if position and not position.is_closed:
                    close_side = OrderSide.SELL if position.is_long else OrderSide.BUY
                    close_quantity = abs(position.quantity)
                    
                    order = self.order_manager.create_order(
                        symbol=signal.symbol,
                        side=close_side,
                        order_type=OrderType.MARKET,
                        quantity=close_quantity
                    )
                    self.execution_stats['orders_created'] += 1
                    
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _handle_order_filled(self, order: Order) -> None:
        """Handle order fill events."""
        # Update position manager
        trade = self.position_manager.process_order_fill(order)
        
        # Update statistics
        self.execution_stats['orders_filled'] += 1
        if trade:
            self.execution_stats['trades_completed'] += 1
        
        # Notify strategies
        for strategy in self.strategies:
            strategy.on_order_filled(order)
        
        # Call callback if set
        if self.on_order_filled:
            self.on_order_filled(order)
    
    def _handle_order_rejected(self, order: Order, reason: str) -> None:
        """Handle order rejection events."""
        logger.warning(f"Order rejected: {order.id} - {reason}")
    
    def _handle_position_opened(self, position: Position) -> None:
        """Handle position opened events."""
        self.execution_stats['positions_opened'] += 1
        
        # Notify strategies
        for strategy in self.strategies:
            strategy.on_position_update(position)
        
        # Call callback if set
        if self.on_position_opened:
            self.on_position_opened(position)
    
    def _handle_position_closed(self, position: Position, trade: Trade) -> None:
        """Handle position closed events."""
        self.execution_stats['positions_closed'] += 1
        
        # Notify strategies
        for strategy in self.strategies:
            strategy.on_position_update(position)
        
        # Call callback if set
        if self.on_position_closed:
            self.on_position_closed(position, trade)
    
    def _handle_margin_call(self, margin_level: float) -> None:
        """Handle margin call events."""
        logger.critical(f"Margin call triggered! Margin level: {margin_level:.2%}")
    
    def _handle_strategy_signal(self, signal: StrategySignal) -> None:
        """Handle strategy signal generation."""
        logger.debug(f"Strategy signal: {signal.signal_type} {signal.symbol} at {signal.timestamp}")
    
    def _handle_strategy_position_opened(self, position: Position) -> None:
        """Handle strategy position opened events."""
        logger.debug(f"Strategy position opened: {position.symbol}")
    
    def _handle_strategy_position_closed(self, position: Position, trade: Trade) -> None:
        """Handle strategy position closed events."""
        logger.debug(f"Strategy position closed: {position.symbol} P&L: {trade.net_pnl:.2f}")
    
    def _update_progress(self) -> None:
        """Update progress and call progress callback."""
        if self.current_tick is None:
            return
        
        # Calculate progress
        total_duration = (self.config.end_date - self.config.start_date).total_seconds()
        current_duration = (self.current_tick.timestamp - self.config.start_date).total_seconds()
        progress = min(current_duration / total_duration, 1.0)
        
        # Prepare status info
        status_info = {
            'processed_ticks': self.processed_ticks,
            'current_time': self.current_tick.timestamp,
            'account_balance': self.position_manager.current_balance,
            'account_equity': self.position_manager.equity,
            'active_positions': len(self.position_manager.get_all_positions()),
            'pending_orders': len(self.order_manager.get_pending_orders()),
            'total_trades': len(self.position_manager.trades)
        }
        
        # Call callback if set
        if self.on_progress_update:
            self.on_progress_update(progress, status_info)
    
    def _finalize_strategies(self) -> None:
        """Finalize all trading strategies."""
        logger.info("Finalizing strategies...")
        
        for strategy in self.strategies:
            strategy.finalize()
            
            # Store strategy results
            self.strategy_results[strategy.config.name] = {
                'config': strategy.config,
                'performance': strategy.get_performance_metrics(),
                'final_state': strategy.state
            }
    
    def _generate_results(self) -> BacktestResult:
        """
        Generate comprehensive backtesting results.
        
        Returns:
            Complete backtesting results
        """
        logger.info("Generating backtesting results...")
        
        # Get all trades
        trades = self.position_manager.trades
        
        # Calculate performance metrics
        total_return = self._calculate_total_return()
        max_drawdown = self.position_manager.max_drawdown
        sharpe_ratio = self._calculate_sharpe_ratio(trades)
        sortino_ratio = self._calculate_sortino_ratio(trades)
        profit_factor = self._calculate_profit_factor(trades)
        
        # Calculate trade statistics
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = sum(t.net_pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.net_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Get execution statistics
        exec_stats = self.order_manager.get_execution_statistics()
        
        # Create result object
        result = BacktestResult(
            config=self.config,
            trades=trades,
            positions=self.position_manager.closed_positions,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_commission=exec_stats.get('total_commission', 0),
            total_slippage=exec_stats.get('total_slippage_pips', 0),
            avg_execution_delay=exec_stats.get('avg_execution_delay_seconds', 0),
            var_95=self._calculate_var_95(trades),
            expected_shortfall=self._calculate_expected_shortfall(trades),
            backtest_start_time=self.start_time,
            backtest_end_time=self.end_time
        )
        
        return result
    
    def _calculate_total_return(self) -> float:
        """Calculate total return as percentage."""
        return (self.position_manager.equity - self.config.initial_balance) / self.config.initial_balance
    
    def _calculate_sharpe_ratio(self, trades: List[Trade]) -> float:
        """Calculate Sharpe ratio."""
        if not trades:
            return 0.0
        
        returns = [t.net_pnl / self.config.initial_balance for t in trades]
        avg_return = sum(returns) / len(returns)
        
        if len(returns) < 2:
            return 0.0
        
        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5
        
        return avg_return / std_dev if std_dev > 0 else 0.0
    
    def _calculate_sortino_ratio(self, trades: List[Trade]) -> float:
        """Calculate Sortino ratio."""
        if not trades:
            return 0.0
        
        returns = [t.net_pnl / self.config.initial_balance for t in trades]
        avg_return = sum(returns) / len(returns)
        
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf') if avg_return > 0 else 0.0
        
        downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
        downside_deviation = downside_variance ** 0.5
        
        return avg_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0.0
        
        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_var_95(self, trades: List[Trade]) -> float:
        """Calculate 95% Value at Risk."""
        if not trades:
            return 0.0
        
        returns = sorted([t.net_pnl for t in trades])
        var_index = int(len(returns) * 0.05)
        
        return returns[var_index] if var_index < len(returns) else 0.0
    
    def _calculate_expected_shortfall(self, trades: List[Trade]) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if not trades:
            return 0.0
        
        var_95 = self._calculate_var_95(trades)
        tail_losses = [t.net_pnl for t in trades if t.net_pnl <= var_95]
        
        return sum(tail_losses) / len(tail_losses) if tail_losses else 0.0
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current backtesting status."""
        return {
            'processed_ticks': self.processed_ticks,
            'current_time': self.current_tick.timestamp if self.current_tick else None,
            'account_summary': self.position_manager.get_account_summary(),
            'position_summary': self.position_manager.get_position_summary(),
            'execution_stats': self.execution_stats,
            'strategy_count': len(self.strategies),
            'is_running': self.start_time is not None and self.end_time is None
        }
    
    def reset(self) -> None:
        """Reset the backtesting engine."""
        self.order_manager.reset()
        self.position_manager.reset()
        
        for strategy in self.strategies:
            strategy.reset()
        
        self.market_events.clear()
        self.strategy_results.clear()
        self.current_tick = None
        self.tick_count = 0
        self.processed_ticks = 0
        self.start_time = None
        self.end_time = None
        
        # Reset execution stats
        self.execution_stats = {
            'total_ticks': 0,
            'orders_created': 0,
            'orders_filled': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'trades_completed': 0
        }
        
        logger.info("Backtesting engine reset")