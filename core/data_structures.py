"""
Core data structures for the backtesting engine.
Defines tick data, orders, positions, and market events.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid


class OrderType(Enum):
    """Order types supported by the backtesting engine."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionType(Enum):
    """How an order was executed."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SLIPPAGE = "slippage"


@dataclass
class Tick:
    """
    Individual tick data point with precise timestamp.
    Represents raw market data from Dukascopy.
    """
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    is_interpolated: bool = False
    
    @property
    def mid(self) -> float:
        """Calculate mid-price between bid and ask."""
        return (self.bid + self.ask) / 2.0
    
    @property
    def spread(self) -> float:
        """Calculate natural spread in pips."""
        return self.ask - self.bid


@dataclass
class Order:
    """
    Order representation with all necessary fields for backtesting.
    Supports market, limit, and stop orders with SL/TP.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0  # In lots
    price: Optional[float] = None  # For limit/stop orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Execution details
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # Backtesting specific
    execution_delay: float = 0.0  # In seconds
    slippage: float = 0.0  # In pips
    
    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_pending(self) -> bool:
        """Check if order is still pending execution."""
        return self.status == OrderStatus.PENDING


@dataclass
class Position:
    """
    Position tracking with precise P&L calculation.
    Handles position sizing, margin, and running P&L.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0  # Net position in lots
    avg_price: float = 0.0  # Average entry price
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Position management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Timestamps
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    
    # Backtesting metrics
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE
    commission_paid: float = 0.0
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    @property
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return abs(self.quantity) < 1e-8
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    def update_unrealized_pnl(self, current_price: float, pip_value: float) -> None:
        """
        Update unrealized P&L based on current market price.
        
        Args:
            current_price: Current market price
            pip_value: Value of 1 pip in account currency
        """
        if self.is_closed:
            self.unrealized_pnl = 0.0
            return
        
        price_diff = current_price - self.avg_price
        if self.is_short:
            price_diff = -price_diff
        
        # Convert price difference to pips and then to account currency
        pips_diff = price_diff / pip_value
        self.unrealized_pnl = pips_diff * self.quantity * 10000 * pip_value
        
        # Update MFE/MAE
        if self.unrealized_pnl > self.max_favorable_excursion:
            self.max_favorable_excursion = self.unrealized_pnl
        if self.unrealized_pnl < self.max_adverse_excursion:
            self.max_adverse_excursion = self.unrealized_pnl


@dataclass
class Trade:
    """
    Completed trade record for performance analysis.
    Created when a position is fully or partially closed.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: datetime = field(default_factory=datetime.now)
    
    # P&L and metrics
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    
    # Execution details
    entry_execution_type: ExecutionType = ExecutionType.MARKET
    exit_execution_type: ExecutionType = ExecutionType.MARKET
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0
    
    # Risk metrics
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    @property
    def duration(self) -> float:
        """Trade duration in seconds."""
        return (self.exit_time - self.entry_time).total_seconds()
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_pnl > 0


@dataclass
class MarketEvent:
    """
    Market event for event-driven backtesting.
    Represents tick updates, order fills, and other market events.
    """
    timestamp: datetime
    event_type: str
    symbol: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting parameters.
    Defines all simulation settings and risk parameters.
    """
    # Data settings
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    
    # Execution settings
    spread_markup: float = 0.0  # Additional spread in pips
    slippage_model: str = "linear"  # linear, random, fixed
    max_slippage: float = 2.0  # Maximum slippage in pips
    execution_delay_min: float = 0.0  # Minimum delay in seconds
    execution_delay_max: float = 0.5  # Maximum delay in seconds
    
    # Account settings
    initial_balance: float = 100000.0
    currency: str = "USD"
    leverage: float = 100.0
    commission_per_lot: float = 7.0  # Commission per standard lot
    
    # Risk settings
    max_position_size: float = 10.0  # Maximum position size in lots
    margin_requirement: float = 0.01  # Margin requirement (1% for 100:1 leverage)
    margin_call_level: float = 0.5  # Margin call at 50%
    
    # Data quality settings
    interpolate_missing_ticks: bool = True
    max_gap_seconds: float = 60.0  # Maximum gap to interpolate
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        
        if self.leverage <= 0:
            raise ValueError("Leverage must be positive")
        
        if not self.symbols:
            raise ValueError("At least one symbol must be specified")


@dataclass
class BacktestResult:
    """
    Complete backtesting results with performance metrics.
    Contains all trades, positions, and statistical analysis.
    """
    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)
    
    # Performance metrics
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Execution statistics
    total_commission: float = 0.0
    total_slippage: float = 0.0
    avg_execution_delay: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    expected_shortfall: float = 0.0
    
    # Timing
    backtest_start_time: datetime = field(default_factory=datetime.now)
    backtest_end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> float:
        """Backtest duration in seconds."""
        if self.backtest_end_time:
            return (self.backtest_end_time - self.backtest_start_time).total_seconds()
        return 0.0