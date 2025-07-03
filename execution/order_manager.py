"""
Order Management System for backtesting.
Handles order creation, execution, and lifecycle management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import random
import uuid

from core.data_structures import (
    Order, OrderType, OrderSide, OrderStatus, ExecutionType,
    Tick, Position, Trade, BacktestConfig
)


logger = logging.getLogger(__name__)


class OrderManager:
    """
    Professional order management system with realistic execution simulation.
    Handles market, limit, and stop orders with slippage and delays.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize the order manager.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.orders: Dict[str, Order] = {}
        self.pending_orders: Dict[str, List[Order]] = defaultdict(list)  # By symbol
        self.executed_orders: List[Order] = []
        
        # Execution callbacks
        self.on_order_filled: Optional[Callable[[Order], None]] = None
        self.on_order_rejected: Optional[Callable[[Order, str], None]] = None
        
        # Slippage and delay models
        self.slippage_model = self._create_slippage_model()
        self.delay_model = self._create_delay_model()
    
    def _create_slippage_model(self) -> Callable[[Order, Tick], float]:
        """
        Create slippage model based on configuration.
        
        Returns:
            Function that calculates slippage for an order
        """
        def linear_slippage(order: Order, tick: Tick) -> float:
            """Linear slippage model based on order size."""
            base_slippage = self.config.max_slippage * 0.3  # 30% of max as base
            size_factor = min(order.quantity / 10.0, 1.0)  # Scale with position size
            return base_slippage + (self.config.max_slippage - base_slippage) * size_factor
        
        def random_slippage(order: Order, tick: Tick) -> float:
            """Random slippage model."""
            return random.uniform(0, self.config.max_slippage)
        
        def fixed_slippage(order: Order, tick: Tick) -> float:
            """Fixed slippage model."""
            return self.config.max_slippage * 0.5
        
        models = {
            'linear': linear_slippage,
            'random': random_slippage,
            'fixed': fixed_slippage
        }
        
        return models.get(self.config.slippage_model, linear_slippage)
    
    def _create_delay_model(self) -> Callable[[Order], float]:
        """
        Create execution delay model.
        
        Returns:
            Function that calculates execution delay for an order
        """
        def calculate_delay(order: Order) -> float:
            """Calculate execution delay based on order type and size."""
            if order.order_type == OrderType.MARKET:
                # Market orders have shorter delays
                min_delay = self.config.execution_delay_min
                max_delay = self.config.execution_delay_max * 0.5
            else:
                # Limit/stop orders can have longer delays
                min_delay = self.config.execution_delay_min * 2
                max_delay = self.config.execution_delay_max
            
            # Add randomness
            return random.uniform(min_delay, max_delay)
        
        return calculate_delay
    
    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                    quantity: float, price: Optional[float] = None,
                    stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Currency pair symbol
            side: Buy or sell
            order_type: Market, limit, or stop
            quantity: Order quantity in lots
            price: Order price (for limit/stop orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Created order object
        """
        # Validate order parameters
        if quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if quantity > self.config.max_position_size:
            raise ValueError(f"Order quantity exceeds maximum position size: {self.config.max_position_size}")
        
        if order_type in [OrderType.LIMIT, OrderType.STOP] and price is None:
            raise ValueError(f"{order_type.value} orders require a price")
        
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            created_at=datetime.now(),
            execution_delay=self.delay_model(Order(symbol=symbol, side=side, order_type=order_type, quantity=quantity))
        )
        
        # Add to tracking
        self.orders[order.id] = order
        self.pending_orders[symbol].append(order)
        
        logger.info(f"Created {order.order_type.value} order: {order.id} for {quantity} lots of {symbol}")
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if order was cancelled successfully
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            return False
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        
        # Remove from pending orders
        self.pending_orders[order.symbol] = [
            o for o in self.pending_orders[order.symbol] if o.id != order_id
        ]
        
        logger.info(f"Cancelled order: {order_id}")
        
        return True
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None,
                    new_quantity: Optional[float] = None,
                    new_stop_loss: Optional[float] = None,
                    new_take_profit: Optional[float] = None) -> bool:
        """
        Modify a pending order.
        
        Args:
            order_id: Order ID to modify
            new_price: New order price
            new_quantity: New order quantity
            new_stop_loss: New stop loss price
            new_take_profit: New take profit price
            
        Returns:
            True if order was modified successfully
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            return False
        
        # Update order parameters
        if new_price is not None:
            order.price = new_price
        if new_quantity is not None:
            if new_quantity <= 0 or new_quantity > self.config.max_position_size:
                return False
            order.quantity = new_quantity
        if new_stop_loss is not None:
            order.stop_loss = new_stop_loss
        if new_take_profit is not None:
            order.take_profit = new_take_profit
        
        logger.info(f"Modified order: {order_id}")
        
        return True
    
    def process_tick(self, tick: Tick) -> List[Order]:
        """
        Process a market tick and execute eligible orders.
        
        Args:
            tick: Market tick data
            
        Returns:
            List of executed orders
        """
        executed_orders = []
        
        # Process pending orders for this symbol
        pending_symbol_orders = self.pending_orders[tick.symbol].copy()
        
        for order in pending_symbol_orders:
            if self._should_execute_order(order, tick):
                # Execute the order
                executed_order = self._execute_order(order, tick)
                if executed_order:
                    executed_orders.append(executed_order)
        
        return executed_orders
    
    def _should_execute_order(self, order: Order, tick: Tick) -> bool:
        """
        Determine if an order should be executed given the current tick.
        
        Args:
            order: Order to evaluate
            tick: Current market tick
            
        Returns:
            True if order should be executed
        """
        # Check if enough time has passed for execution delay
        time_since_created = (tick.timestamp - order.created_at).total_seconds()
        if time_since_created < order.execution_delay:
            return False
        
        # Check execution conditions based on order type
        if order.order_type == OrderType.MARKET:
            return True
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                # Buy limit: execute when ask price <= limit price
                return tick.ask <= order.price
            else:
                # Sell limit: execute when bid price >= limit price
                return tick.bid >= order.price
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                # Buy stop: execute when ask price >= stop price
                return tick.ask >= order.price
            else:
                # Sell stop: execute when bid price <= stop price
                return tick.bid <= order.price
        
        return False
    
    def _execute_order(self, order: Order, tick: Tick) -> Optional[Order]:
        """
        Execute an order at the current market conditions.
        
        Args:
            order: Order to execute
            tick: Current market tick
            
        Returns:
            Executed order or None if execution failed
        """
        try:
            # Calculate execution price
            execution_price = self._calculate_execution_price(order, tick)
            
            # Apply slippage
            slippage = self.slippage_model(order, tick)
            if order.side == OrderSide.BUY:
                execution_price += slippage * self._get_pip_value(order.symbol)
            else:
                execution_price -= slippage * self._get_pip_value(order.symbol)
            
            # Apply spread markup
            spread_markup = self.config.spread_markup * self._get_pip_value(order.symbol)
            if order.side == OrderSide.BUY:
                execution_price += spread_markup
            else:
                execution_price -= spread_markup
            
            # Calculate commission
            commission = self.config.commission_per_lot * order.quantity
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = execution_price
            order.commission = commission
            order.filled_at = tick.timestamp
            order.slippage = slippage
            
            # Remove from pending orders
            self.pending_orders[order.symbol] = [
                o for o in self.pending_orders[order.symbol] if o.id != order.id
            ]
            
            # Add to executed orders
            self.executed_orders.append(order)
            
            # Call callback if set
            if self.on_order_filled:
                self.on_order_filled(order)
            
            logger.info(f"Executed order: {order.id} at {execution_price:.5f} with {slippage:.1f} pips slippage")
            
            return order
        
        except Exception as e:
            logger.error(f"Error executing order {order.id}: {e}")
            
            # Mark order as rejected
            order.status = OrderStatus.REJECTED
            
            # Remove from pending orders
            self.pending_orders[order.symbol] = [
                o for o in self.pending_orders[order.symbol] if o.id != order.id
            ]
            
            # Call callback if set
            if self.on_order_rejected:
                self.on_order_rejected(order, str(e))
            
            return None
    
    def _calculate_execution_price(self, order: Order, tick: Tick) -> float:
        """
        Calculate the execution price for an order.
        
        Args:
            order: Order to execute
            tick: Current market tick
            
        Returns:
            Execution price
        """
        if order.order_type == OrderType.MARKET:
            # Market orders execute at current market price
            return tick.ask if order.side == OrderSide.BUY else tick.bid
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders execute at the limit price or better
            if order.side == OrderSide.BUY:
                return min(order.price, tick.ask)
            else:
                return max(order.price, tick.bid)
        
        elif order.order_type == OrderType.STOP:
            # Stop orders execute at market price once triggered
            return tick.ask if order.side == OrderSide.BUY else tick.bid
        
        return tick.mid  # Fallback
    
    def _get_pip_value(self, symbol: str) -> float:
        """
        Get pip value for a currency pair.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            Pip value
        """
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all pending orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of pending orders
        """
        if symbol:
            return self.pending_orders[symbol].copy()
        
        all_pending = []
        for symbol_orders in self.pending_orders.values():
            all_pending.extend(symbol_orders)
        
        return all_pending
    
    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """
        Get an order by its ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        return self.orders.get(order_id)
    
    def get_execution_statistics(self) -> dict:
        """
        Get order execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self.executed_orders:
            return {}
        
        total_orders = len(self.executed_orders)
        total_slippage = sum(order.slippage for order in self.executed_orders)
        total_commission = sum(order.commission for order in self.executed_orders)
        
        avg_slippage = total_slippage / total_orders
        avg_commission = total_commission / total_orders
        
        # Calculate average execution delay
        avg_delay = sum(order.execution_delay for order in self.executed_orders) / total_orders
        
        return {
            'total_orders_executed': total_orders,
            'total_slippage_pips': total_slippage,
            'avg_slippage_pips': avg_slippage,
            'total_commission': total_commission,
            'avg_commission_per_order': avg_commission,
            'avg_execution_delay_seconds': avg_delay
        }
    
    def reset(self) -> None:
        """Reset the order manager state."""
        self.orders.clear()
        self.pending_orders.clear()
        self.executed_orders.clear()
        
        logger.info("Order manager reset")