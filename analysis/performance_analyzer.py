"""
Performance Analysis and Reporting Module.
Provides comprehensive analysis of backtesting results and strategy performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import asdict
import json

from core.data_structures import Trade, Position, BacktestResult, BacktestConfig


logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.
    Provides detailed metrics, risk analysis, and reporting capabilities.
    """
    
    def __init__(self, result: BacktestResult):
        """
        Initialize performance analyzer.
        
        Args:
            result: Backtesting result to analyze
        """
        self.result = result
        self.trades_df = self._create_trades_dataframe()
        self.equity_curve = self._calculate_equity_curve()
        self.monthly_returns = self._calculate_monthly_returns()
        self.risk_metrics = self._calculate_risk_metrics()
        
        logger.info(f"Performance analyzer initialized with {len(result.trades)} trades")
    
    def _create_trades_dataframe(self) -> pd.DataFrame:
        """
        Create pandas DataFrame from trades for easier analysis.
        
        Returns:
            DataFrame with trade data
        """
        if not self.result.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.result.trades:
            trade_dict = asdict(trade)
            trades_data.append(trade_dict)
        
        df = pd.DataFrame(trades_data)
        
        # Convert datetime columns
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        # Calculate additional metrics
        df['duration_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
        df['return_pct'] = df['net_pnl'] / self.result.config.initial_balance * 100
        df['pips'] = np.where(
            df['symbol'].str.contains('JPY'),
            (df['exit_price'] - df['entry_price']) * 100,
            (df['exit_price'] - df['entry_price']) * 10000
        )
        
        # Adjust pips for short positions
        df.loc[df['side'] == 'SELL', 'pips'] *= -1
        
        return df
    
    def _calculate_equity_curve(self) -> pd.Series:
        """
        Calculate equity curve over time.
        
        Returns:
            Series with equity values indexed by time
        """
        if self.trades_df.empty:
            return pd.Series(dtype=float)
        
        # Create time series of equity
        equity_data = []
        running_balance = self.result.config.initial_balance
        
        for _, trade in self.trades_df.iterrows():
            running_balance += trade['net_pnl']
            equity_data.append({
                'timestamp': trade['exit_time'],
                'equity': running_balance,
                'pnl': trade['net_pnl']
            })
        
        equity_df = pd.DataFrame(equity_data)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        return equity_df.set_index('timestamp')['equity']
    
    def _calculate_monthly_returns(self) -> pd.Series:
        """
        Calculate monthly returns.
        
        Returns:
            Series with monthly returns
        """
        if self.equity_curve.empty:
            return pd.Series(dtype=float)
        
        # Resample to monthly frequency
        monthly_equity = self.equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        return monthly_returns
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        if self.trades_df.empty:
            return {}
        
        returns = self.trades_df['return_pct'].values
        
        # Basic statistics
        total_return = sum(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected shortfall (conditional VaR)
        es_95 = np.mean(returns[returns <= var_95])
        es_99 = np.mean(returns[returns <= var_99])
        
        # Maximum consecutive losses
        max_consecutive_losses = self._calculate_max_consecutive_losses()
        
        # Drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics()
        
        # Calmar ratio
        calmar_ratio = (total_return / 100) / abs(drawdown_metrics['max_drawdown']) if drawdown_metrics['max_drawdown'] < 0 else 0
        
        return {
            'total_return_pct': total_return,
            'avg_return_pct': avg_return,
            'volatility_pct': std_return,
            'var_95_pct': var_95,
            'var_99_pct': var_99,
            'expected_shortfall_95_pct': es_95,
            'expected_shortfall_99_pct': es_99,
            'max_consecutive_losses': max_consecutive_losses,
            'calmar_ratio': calmar_ratio,
            **drawdown_metrics
        }
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses."""
        if self.trades_df.empty:
            return 0
        
        consecutive_losses = 0
        max_consecutive = 0
        
        for _, trade in self.trades_df.iterrows():
            if trade['net_pnl'] < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive
    
    def _calculate_drawdown_metrics(self) -> Dict[str, float]:
        """Calculate drawdown metrics."""
        if self.equity_curve.empty:
            return {'max_drawdown': 0, 'avg_drawdown': 0, 'drawdown_duration_days': 0}
        
        # Calculate running maximum
        running_max = self.equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (self.equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Drawdown duration
        drawdown_duration = self._calculate_drawdown_duration(drawdown)
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'drawdown_duration_days': drawdown_duration
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> float:
        """Calculate average drawdown duration."""
        if drawdown.empty:
            return 0
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        start_idx = None
        for idx, is_dd in in_drawdown.items():
            if is_dd and start_idx is None:
                start_idx = idx
            elif not is_dd and start_idx is not None:
                drawdown_periods.append((start_idx, idx))
                start_idx = None
        
        # Handle case where drawdown continues to end
        if start_idx is not None:
            drawdown_periods.append((start_idx, drawdown.index[-1]))
        
        # Calculate durations
        durations = []
        for start, end in drawdown_periods:
            duration = (end - start).days
            durations.append(duration)
        
        return np.mean(durations) if durations else 0
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Returns:
            Dictionary with complete analysis
        """
        # Basic performance metrics
        basic_metrics = {
            'start_date': self.result.config.start_date.isoformat(),
            'end_date': self.result.config.end_date.isoformat(),
            'initial_balance': self.result.config.initial_balance,
            'final_balance': self.result.config.initial_balance + sum(t.net_pnl for t in self.result.trades),
            'total_return': self.result.total_return,
            'total_trades': self.result.total_trades,
            'winning_trades': self.result.winning_trades,
            'losing_trades': self.result.losing_trades,
            'win_rate': self.result.win_rate,
            'profit_factor': self.result.profit_factor,
            'sharpe_ratio': self.result.sharpe_ratio,
            'sortino_ratio': self.result.sortino_ratio,
            'max_drawdown': self.result.max_drawdown
        }
        
        # Trade analysis
        trade_analysis = self._analyze_trades()
        
        # Time analysis
        time_analysis = self._analyze_time_patterns()
        
        # Risk analysis
        risk_analysis = self.risk_metrics
        
        # Execution analysis
        execution_analysis = {
            'total_commission': self.result.total_commission,
            'avg_commission_per_trade': self.result.total_commission / self.result.total_trades if self.result.total_trades > 0 else 0,
            'total_slippage_pips': self.result.total_slippage,
            'avg_slippage_per_trade': self.result.total_slippage / self.result.total_trades if self.result.total_trades > 0 else 0,
            'avg_execution_delay': self.result.avg_execution_delay
        }
        
        # Performance by symbol
        symbol_analysis = self._analyze_by_symbol()
        
        return {
            'basic_metrics': basic_metrics,
            'trade_analysis': trade_analysis,
            'time_analysis': time_analysis,
            'risk_analysis': risk_analysis,
            'execution_analysis': execution_analysis,
            'symbol_analysis': symbol_analysis,
            'monthly_returns': self.monthly_returns.to_dict() if not self.monthly_returns.empty else {}
        }
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade characteristics."""
        if self.trades_df.empty:
            return {}
        
        winning_trades = self.trades_df[self.trades_df['net_pnl'] > 0]
        losing_trades = self.trades_df[self.trades_df['net_pnl'] < 0]
        
        return {
            'avg_winner': winning_trades['net_pnl'].mean() if not winning_trades.empty else 0,
            'avg_loser': losing_trades['net_pnl'].mean() if not losing_trades.empty else 0,
            'largest_winner': self.trades_df['net_pnl'].max(),
            'largest_loser': self.trades_df['net_pnl'].min(),
            'avg_trade_duration_minutes': self.trades_df['duration_minutes'].mean(),
            'avg_winning_duration_minutes': winning_trades['duration_minutes'].mean() if not winning_trades.empty else 0,
            'avg_losing_duration_minutes': losing_trades['duration_minutes'].mean() if not losing_trades.empty else 0,
            'avg_pips_per_trade': self.trades_df['pips'].mean(),
            'total_pips': self.trades_df['pips'].sum(),
            'best_trade_pips': self.trades_df['pips'].max(),
            'worst_trade_pips': self.trades_df['pips'].min()
        }
    
    def _analyze_time_patterns(self) -> Dict[str, Any]:
        """Analyze time-based patterns."""
        if self.trades_df.empty:
            return {}
        
        # Add time features
        df = self.trades_df.copy()
        df['hour'] = df['entry_time'].dt.hour
        df['day_of_week'] = df['entry_time'].dt.dayofweek
        df['month'] = df['entry_time'].dt.month
        
        # Performance by hour
        hourly_performance = df.groupby('hour')['net_pnl'].agg(['count', 'mean', 'sum']).to_dict()
        
        # Performance by day of week
        daily_performance = df.groupby('day_of_week')['net_pnl'].agg(['count', 'mean', 'sum']).to_dict()
        
        # Performance by month
        monthly_performance = df.groupby('month')['net_pnl'].agg(['count', 'mean', 'sum']).to_dict()
        
        return {
            'hourly_performance': hourly_performance,
            'daily_performance': daily_performance,
            'monthly_performance': monthly_performance,
            'best_hour': df.groupby('hour')['net_pnl'].sum().idxmax(),
            'worst_hour': df.groupby('hour')['net_pnl'].sum().idxmin(),
            'best_day': df.groupby('day_of_week')['net_pnl'].sum().idxmax(),
            'worst_day': df.groupby('day_of_week')['net_pnl'].sum().idxmin()
        }
    
    def _analyze_by_symbol(self) -> Dict[str, Any]:
        """Analyze performance by symbol."""
        if self.trades_df.empty:
            return {}
        
        symbol_stats = {}
        
        for symbol in self.trades_df['symbol'].unique():
            symbol_trades = self.trades_df[self.trades_df['symbol'] == symbol]
            
            symbol_stats[symbol] = {
                'total_trades': len(symbol_trades),
                'winning_trades': len(symbol_trades[symbol_trades['net_pnl'] > 0]),
                'losing_trades': len(symbol_trades[symbol_trades['net_pnl'] < 0]),
                'win_rate': len(symbol_trades[symbol_trades['net_pnl'] > 0]) / len(symbol_trades),
                'total_pnl': symbol_trades['net_pnl'].sum(),
                'avg_pnl': symbol_trades['net_pnl'].mean(),
                'total_pips': symbol_trades['pips'].sum(),
                'avg_pips': symbol_trades['pips'].mean(),
                'best_trade': symbol_trades['net_pnl'].max(),
                'worst_trade': symbol_trades['net_pnl'].min()
            }
        
        return symbol_stats
    
    def export_detailed_report(self, filename: str, format: str = 'json') -> None:
        """
        Export detailed report to file.
        
        Args:
            filename: Output filename
            format: Export format ('json', 'csv', 'excel')
        """
        summary = self.generate_summary_report()
        
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        elif format == 'csv':
            if not self.trades_df.empty:
                self.trades_df.to_csv(filename, index=False)
        
        elif format == 'excel':
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([summary['basic_metrics']])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Trades sheet
                if not self.trades_df.empty:
                    self.trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Monthly returns sheet
                if not self.monthly_returns.empty:
                    self.monthly_returns.to_excel(writer, sheet_name='Monthly Returns')
                
                # Risk metrics sheet
                risk_df = pd.DataFrame([summary['risk_analysis']])
                risk_df.to_excel(writer, sheet_name='Risk Metrics', index=False)
        
        logger.info(f"Report exported to {filename}")
    
    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.generate_summary_report()
        
        print("\n" + "="*60)
        print("BACKTESTING RESULTS SUMMARY")
        print("="*60)
        
        # Basic metrics
        basic = summary['basic_metrics']
        print(f"\nPERFORMANCE OVERVIEW:")
        print(f"  Period: {basic['start_date'][:10]} to {basic['end_date'][:10]}")
        print(f"  Initial Balance: ${basic['initial_balance']:,.2f}")
        print(f"  Final Balance: ${basic['final_balance']:,.2f}")
        print(f"  Total Return: {basic['total_return']:.2%}")
        print(f"  Max Drawdown: {basic['max_drawdown']:.2%}")
        
        # Trade statistics
        print(f"\nTRADE STATISTICS:")
        print(f"  Total Trades: {basic['total_trades']}")
        print(f"  Winning Trades: {basic['winning_trades']}")
        print(f"  Losing Trades: {basic['losing_trades']}")
        print(f"  Win Rate: {basic['win_rate']:.2%}")
        print(f"  Profit Factor: {basic['profit_factor']:.2f}")
        
        # Risk metrics
        print(f"\nRISK METRICS:")
        print(f"  Sharpe Ratio: {basic['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {basic['sortino_ratio']:.2f}")
        
        # Execution metrics
        exec_metrics = summary['execution_analysis']
        print(f"\nEXECUTION ANALYSIS:")
        print(f"  Total Commission: ${exec_metrics['total_commission']:.2f}")
        print(f"  Total Slippage: {exec_metrics['total_slippage_pips']:.1f} pips")
        print(f"  Avg Execution Delay: {exec_metrics['avg_execution_delay']:.3f} seconds")
        
        print("="*60)
    
    def get_trade_statistics(self) -> Dict[str, float]:
        """Get basic trade statistics."""
        if self.trades_df.empty:
            return {}
        
        return {
            'total_trades': len(self.trades_df),
            'winning_trades': len(self.trades_df[self.trades_df['net_pnl'] > 0]),
            'losing_trades': len(self.trades_df[self.trades_df['net_pnl'] < 0]),
            'win_rate': len(self.trades_df[self.trades_df['net_pnl'] > 0]) / len(self.trades_df),
            'total_pnl': self.trades_df['net_pnl'].sum(),
            'avg_pnl': self.trades_df['net_pnl'].mean(),
            'best_trade': self.trades_df['net_pnl'].max(),
            'worst_trade': self.trades_df['net_pnl'].min(),
            'total_pips': self.trades_df['pips'].sum(),
            'avg_pips': self.trades_df['pips'].mean(),
            'profit_factor': self.trades_df[self.trades_df['net_pnl'] > 0]['net_pnl'].sum() / abs(self.trades_df[self.trades_df['net_pnl'] < 0]['net_pnl'].sum()) if len(self.trades_df[self.trades_df['net_pnl'] < 0]) > 0 else float('inf')
        }