"""
Professional Backtesting Suite for Algorithmic Trading.

A comprehensive backtesting framework designed for tick-by-tick simulation
of trading strategies with realistic execution modeling.

Features:
- Tick-by-tick precision simulation
- Dukascopy data integration
- Realistic spread, slippage, and delay modeling
- Professional order and position management
- Comprehensive performance analysis
- Strategy template framework
- Risk management tools

Author: Claude
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude"

# Core imports
from core.data_structures import (
    Tick, Order, Position, Trade, BacktestConfig, BacktestResult,
    OrderType, OrderSide, OrderStatus, ExecutionType
)

# Engine imports
from engine.backtest_engine import BacktestEngine

# Strategy framework
from strategy.strategy_interface import (
    TradingStrategy, StrategyConfig, StrategySignal,
    TechnicalIndicator, MovingAverage, RSI
)

# Data handling
from data.dukascopy_loader import DukascopyDataLoader

# Execution components
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager

# Analysis tools
from analysis.performance_analyzer import PerformanceAnalyzer

__all__ = [
    # Core data structures
    'Tick', 'Order', 'Position', 'Trade', 'BacktestConfig', 'BacktestResult',
    'OrderType', 'OrderSide', 'OrderStatus', 'ExecutionType',
    
    # Main engine
    'BacktestEngine',
    
    # Strategy framework
    'TradingStrategy', 'StrategyConfig', 'StrategySignal',
    'TechnicalIndicator', 'MovingAverage', 'RSI',
    
    # Data and execution
    'DukascopyDataLoader', 'OrderManager', 'PositionManager',
    
    # Analysis
    'PerformanceAnalyzer'
]