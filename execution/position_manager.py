"""
Position Management System for backtesting.
Handles position tracking, P&L calculations, and risk management.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from collections import defaultdict

from core.data_structures import (
    Position, Order, OrderSide, Trade, Tick, ExecutionType, BacktestConfig
)


logger = logging.getLogger(__name__)


class PositionManager:
    """
    Professional position management system with precise P&L tracking.
    Handles position sizing, margin calculations, and risk management.
    """
    
    def __init__(self, config: BacktestConfig, initial_balance: float):
        """
        Initialize the position manager.
        
        Args:
            config: Backtesting configuration
            initial_balance: Starting account balance
        """
        self.config = config
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity = initial_balance
        self.margin_used = 0.0
        self.free_margin = initial_balance
        
        # Position tracking
        self.positions: Dict[str, Position] = {}  # By symbol
        self.closed_positions: List[Position] = []
        self.trades: List[Trade] = []
        
        # Market data cache
        self.current_prices: Dict[str, Tick] = {}
        self.pip_values = self._calculate_pip_values()
        
        # Risk management
        self.max_drawdown = 0.0
        self.peak_equity = initial_balance
        
        # Callbacks
        self.on_position_opened: Optional[Callable[[Position], None]] = None
        self.on_position_closed: Optional[Callable[[Position, Trade], None]] = None
        self.on_margin_call: Optional[Callable[[float], None]] = None
    
    def _calculate_pip_values(self) -> Dict[str, float]:
        """
        Calculate pip values for all symbols.
        
        Returns:
            Dictionary mapping symbols to pip values
        """
        pip_values = {}
        
        for symbol in self.config.symbols:
            if 'JPY' in symbol:
                pip_values[symbol] = 0.01
            else:
                pip_values[symbol] = 0.0001
        
        return pip_values
    
    def update_market_data(self, tick: Tick) -> None:
        """
        Update market data and recalculate all positions.
        
        Args:
            tick: Latest market tick
        """
        self.current_prices[tick.symbol] = tick
        
        # Update unrealized P&L for all positions
        if tick.symbol in self.positions:
            position = self.positions[tick.symbol]
            if not position.is_closed:
                self._update_position_pnl(position, tick)
        
        # Update equity and margin calculations
        self._update_equity()
        self._update_margin()
        
        # Check for margin call
        if self._is_margin_call():
            self._handle_margin_call()
    
    def process_order_fill(self, order: Order) -> Optional[Trade]:
        """
        Process a filled order and update positions.
        
        Args:
            order: Filled order
            
        Returns:
            Trade object if position was closed, None otherwise
        """
        symbol = order.symbol
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                side=order.side,
                quantity=0.0,
                avg_price=0.0,
                opened_at=order.filled_at or datetime.now()
            )
        
        position = self.positions[symbol]
        
        # Process the order fill
        trade = self._process_order_fill(position, order)
        
        # Update balance with commission
        self.current_balance -= order.commission
        
        # Log the position update
        logger.info(f"Position updated for {symbol}: {position.quantity} lots at {position.avg_price:.5f}")
        
        return trade
    
    def _process_order_fill(self, position: Position, order: Order) -> Optional[Trade]:
        """
        Process an order fill against a position.
        
        Args:
            position: Position to update
            order: Filled order
            
        Returns:
            Trade object if position was closed, None otherwise
        """
        order_quantity = order.filled_quantity
        order_price = order.avg_fill_price
        
        # Determine if this is opening or closing the position
        if position.quantity == 0:
            # Opening new position
            return self._open_position(position, order)
        
        elif (position.is_long and order.side == OrderSide.SELL) or \
             (position.is_short and order.side == OrderSide.BUY):
            # Closing or reducing position
            return self._close_position(position, order)
        
        else:
            # Adding to position
            return self._add_to_position(position, order)
    
    def _open_position(self, position: Position, order: Order) -> None:
        """
        Open a new position.
        
        Args:
            position: Position to open
            order: Opening order
        """
        position.side = order.side
        position.quantity = order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity
        position.avg_price = order.avg_fill_price
        position.opened_at = order.filled_at or datetime.now()
        position.commission_paid += order.commission
        
        # Initialize MFE/MAE
        position.max_favorable_excursion = 0.0
        position.max_adverse_excursion = 0.0
        
        # Call callback
        if self.on_position_opened:
            self.on_position_opened(position)
        
        logger.info(f"Opened position: {position.quantity} lots of {position.symbol} at {position.avg_price:.5f}")
    
    def _add_to_position(self, position: Position, order: Order) -> None:
        """
        Add to an existing position.
        
        Args:
            position: Position to add to
            order: Adding order
        """
        # Calculate new average price
        total_value = position.quantity * position.avg_price
        order_value = order.filled_quantity * order.avg_fill_price
        
        if order.side == OrderSide.BUY:
            new_quantity = position.quantity + order.filled_quantity
            new_value = total_value + order_value
        else:
            new_quantity = position.quantity - order.filled_quantity
            new_value = total_value - order_value
        
        if new_quantity != 0:
            position.avg_price = new_value / new_quantity
        
        position.quantity = new_quantity
        position.commission_paid += order.commission
        
        logger.info(f"Added to position: {position.quantity} lots of {position.symbol} at avg {position.avg_price:.5f}")
    
    def _close_position(self, position: Position, order: Order) -> Optional[Trade]:
        """
        Close or reduce a position.
        
        Args:
            position: Position to close/reduce
            order: Closing order
            
        Returns:
            Trade object if position was closed
        """
        # Calculate quantities
        closing_quantity = min(abs(position.quantity), order.filled_quantity)
        
        # Calculate P&L for the closed portion
        if position.is_long:
            pnl = (order.avg_fill_price - position.avg_price) * closing_quantity
        else:
            pnl = (position.avg_price - order.avg_fill_price) * closing_quantity
        
        # Convert to account currency
        pip_value = self.pip_values[position.symbol]
        pnl_pips = pnl / pip_value
        pnl_account_currency = pnl_pips * closing_quantity * 100000 * pip_value  # Standard lot conversion
        
        # Update position
        if abs(position.quantity) <= closing_quantity:
            # Position fully closed
            position.realized_pnl += pnl_account_currency
            position.commission_paid += order.commission
            position.closed_at = order.filled_at or datetime.now()
            
            # Create trade record
            trade = Trade(
                symbol=position.symbol,
                side=position.side,
                quantity=abs(position.quantity),
                entry_price=position.avg_price,
                exit_price=order.avg_fill_price,
                entry_time=position.opened_at,
                exit_time=position.closed_at,
                gross_pnl=pnl_account_currency,
                commission=position.commission_paid,
                net_pnl=pnl_account_currency - position.commission_paid,
                entry_execution_type=ExecutionType.MARKET,
                exit_execution_type=ExecutionType.MARKET,
                entry_slippage=0.0,
                exit_slippage=order.slippage,
                max_favorable_excursion=position.max_favorable_excursion,
                max_adverse_excursion=position.max_adverse_excursion
            )
            
            # Update balance
            self.current_balance += pnl_account_currency
            
            # Move to closed positions
            self.closed_positions.append(position)
            self.trades.append(trade)
            
            # Reset position
            position.quantity = 0.0
            position.unrealized_pnl = 0.0
            
            # Call callback
            if self.on_position_closed:
                self.on_position_closed(position, trade)
            
            logger.info(f"Closed position: {trade.quantity} lots of {trade.symbol} for {trade.net_pnl:.2f} profit")
            
            return trade
        
        else:
            # Position partially closed
            position.quantity = position.quantity - closing_quantity * (1 if position.is_long else -1)
            position.realized_pnl += pnl_account_currency
            position.commission_paid += order.commission
            
            # Update balance
            self.current_balance += pnl_account_currency
            
            logger.info(f"Partially closed position: {closing_quantity} lots of {position.symbol} for {pnl_account_currency:.2f} profit")
            
            return None
    
    def _update_position_pnl(self, position: Position, tick: Tick) -> None:
        """
        Update position unrealized P&L.
        
        Args:
            position: Position to update
            tick: Current market tick
        """
        if position.is_closed:
            return
        
        # Get current price
        current_price = tick.bid if position.is_long else tick.ask
        
        # Calculate unrealized P&L
        pip_value = self.pip_values[position.symbol]
        position.update_unrealized_pnl(current_price, pip_value)
    
    def _update_equity(self) -> None:
        """Update account equity with unrealized P&L."""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.equity = self.current_balance + unrealized_pnl
        
        # Update peak equity and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def _update_margin(self) -> None:
        """Update margin calculations."""
        self.margin_used = 0.0
        
        for position in self.positions.values():
            if not position.is_closed:
                # Calculate margin requirement
                position_value = abs(position.quantity) * 100000  # Standard lot value
                margin_required = position_value * self.config.margin_requirement
                self.margin_used += margin_required
        
        self.free_margin = self.equity - self.margin_used
    
    def _is_margin_call(self) -> bool:
        """Check if margin call conditions are met."""
        if self.margin_used == 0:
            return False
        
        margin_level = self.equity / self.margin_used
        return margin_level < self.config.margin_call_level
    
    def _handle_margin_call(self) -> None:
        """Handle margin call by closing positions."""
        logger.warning(f"Margin call triggered! Equity: {self.equity:.2f}, Margin used: {self.margin_used:.2f}")
        
        # Call callback
        if self.on_margin_call:
            self.on_margin_call(self.equity / self.margin_used)
        
        # Close positions with worst unrealized P&L first
        positions_to_close = sorted(
            [pos for pos in self.positions.values() if not pos.is_closed],
            key=lambda x: x.unrealized_pnl
        )
        
        for position in positions_to_close:
            if not self._is_margin_call():
                break
            
            # Create market order to close position
            logger.info(f"Force closing position: {position.symbol} due to margin call")
            # Note: This would typically trigger an order through the order manager
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position object or None if no position
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """
        Get all active positions.
        
        Returns:
            List of active positions
        """
        return [pos for pos in self.positions.values() if not pos.is_closed]
    
    def get_net_position(self, symbol: str) -> float:
        """
        Get net position size for a symbol.
        
        Args:
            symbol: Symbol to get net position for
            
        Returns:
            Net position size in lots
        """
        position = self.positions.get(symbol)
        return position.quantity if position else 0.0
    
    def can_open_position(self, symbol: str, quantity: float) -> bool:
        """
        Check if a position can be opened given margin requirements.
        
        Args:
            symbol: Symbol to check
            quantity: Quantity to open
            
        Returns:
            True if position can be opened
        """
        # Calculate margin required for new position
        position_value = quantity * 100000  # Standard lot value
        margin_required = position_value * self.config.margin_requirement
        
        # Check if we have enough free margin
        return self.free_margin >= margin_required
    
    def get_account_summary(self) -> dict:
        """
        Get account summary with current balance, equity, and margin info.
        
        Returns:
            Dictionary with account summary
        """
        return {
            'balance': self.current_balance,
            'equity': self.equity,
            'margin_used': self.margin_used,
            'free_margin': self.free_margin,
            'margin_level': self.equity / self.margin_used if self.margin_used > 0 else float('inf'),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'realized_pnl': self.current_balance - self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'active_positions': len(self.get_all_positions()),
            'total_trades': len(self.trades)
        }
    
    def get_position_summary(self) -> dict:
        """
        Get summary of all positions.
        
        Returns:
            Dictionary with position summary
        """
        active_positions = self.get_all_positions()
        
        total_unrealized = sum(pos.unrealized_pnl for pos in active_positions)
        total_realized = sum(pos.realized_pnl for pos in self.closed_positions)
        
        return {
            'active_positions': len(active_positions),
            'closed_positions': len(self.closed_positions),
            'total_unrealized_pnl': total_unrealized,
            'total_realized_pnl': total_realized,
            'total_pnl': total_unrealized + total_realized,
            'positions_by_symbol': {pos.symbol: pos.quantity for pos in active_positions}
        }
    
    def reset(self) -> None:
        """Reset the position manager to initial state."""
        self.current_balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin_used = 0.0
        self.free_margin = self.initial_balance
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        self.positions.clear()
        self.closed_positions.clear()
        self.trades.clear()
        self.current_prices.clear()
        
        logger.info("Position manager reset")