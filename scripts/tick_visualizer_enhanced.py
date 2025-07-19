#!/usr/bin/env python3
"""
Enhanced Tick Data Visualizer with Gap Analysis
Professional visualization for massive tick datasets with data quality analysis.

Author: Lorenzo Albanese (alblor)
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import warnings

# Add backtester to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_structures import BacktestConfig, Tick
from data.data_loader_factory import DataLoaderFactory

# Suppress all matplotlib warnings
warnings.filterwarnings('ignore')
os.environ['MPLCONFIGDIR'] = '/tmp'

class EnhancedTickViewer:
    """
    Enhanced tick data viewer with gap analysis and data quality metrics.
    
    Features:
    - Handles millions of ticks efficiently
    - Real-time navigation without lag
    - Clear gap visualization instead of interpolation
    - Comprehensive data quality analysis
    - Professional gap indicators with timing information
    - CSV file range display
    - No font dependencies or crashes
    """
    
    def __init__(self, ticks: List[Tick], symbol: str):
        # Configure matplotlib with minimal overhead
        import matplotlib
        matplotlib.use('TkAgg', force=True)
        matplotlib.rcParams.update({
            'interactive': False,
            'axes.unicode_minus': False,
            'font.size': 9,
            'figure.max_open_warning': 0
        })
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        self.plt = plt
        self.np = np
        
        # Data management
        self.all_ticks = ticks
        self.symbol = symbol
        self.total_ticks = len(ticks)
        
        # Analyze data quality and gaps
        self.data_quality = self._analyze_data_quality()
        self.gaps = self._detect_gaps()
        self.csv_ranges = self._get_csv_file_ranges()
        
        print(f"\nData Quality Analysis for {symbol}:")
        print(f"  Real ticks: {self.data_quality['real_ticks']:,} ({self.data_quality['real_percentage']:.1f}%)")
        print(f"  Interpolated: {self.data_quality['interpolated_ticks']:,} ({self.data_quality['interpolated_percentage']:.1f}%)")
        print(f"  Data gaps (>5min): {len(self.gaps)}")
        print(f"  Quality score: {self.data_quality['quality_score']:.1f}%")
        
        if self.gaps:
            total_gap_hours = sum(gap['duration_hours'] for gap in self.gaps)
            print(f"  Total gap time: {total_gap_hours:.1f} hours")
        
        if self.csv_ranges:
            print(f"  CSV files: {len(self.csv_ranges)}")
            for csv_range in self.csv_ranges:
                print(f"    {csv_range['file']}: {csv_range['start_date']} to {csv_range['end_date']}")
        
        # Efficient windowing for large datasets
        self.window_size = 5000  # Ticks to keep in memory
        self.display_size = 1000  # Ticks to display on chart
        self.current_position = self.total_ticks // 2  # Start in middle
        
        # Load initial window
        self._load_current_window()
        
        # Navigation
        self.movement_step = 1
        self.movement_modes = ['1 tick', '10 ticks', '100 ticks', '1000 ticks', '10% jump']
        self.movement_mode_idx = 1  # Start with 10 ticks
        
        # GUI components
        self.fig = None
        self.ax_chart = None
        self.ax_info = None
        self.current_lines = []
        self.updating = False
    
    def _analyze_data_quality(self):
        """Analyze the quality of tick data."""
        if not self.all_ticks:
            return {
                'real_ticks': 0,
                'interpolated_ticks': 0,
                'real_percentage': 0.0,
                'interpolated_percentage': 0.0,
                'quality_score': 0.0,
                'first_tick': None,
                'last_tick': None
            }
        
        real_ticks = sum(1 for tick in self.all_ticks if not tick.is_interpolated)
        interpolated_ticks = sum(1 for tick in self.all_ticks if tick.is_interpolated)
        
        real_percentage = (real_ticks / self.total_ticks) * 100
        interpolated_percentage = (interpolated_ticks / self.total_ticks) * 100
        
        # Quality score based on real data percentage
        quality_score = real_percentage
        
        return {
            'real_ticks': real_ticks,
            'interpolated_ticks': interpolated_ticks,
            'real_percentage': real_percentage,
            'interpolated_percentage': interpolated_percentage,
            'quality_score': quality_score,
            'first_tick': self.all_ticks[0].timestamp if self.all_ticks else None,
            'last_tick': self.all_ticks[-1].timestamp if self.all_ticks else None
        }
    
    def _detect_gaps(self):
        """Detect significant gaps in tick data."""
        gaps = []
        if len(self.all_ticks) < 2:
            return gaps
        
        # Define significant gap threshold (more than 5 minutes)
        gap_threshold_seconds = 300
        
        for i in range(len(self.all_ticks) - 1):
            current_tick = self.all_ticks[i]
            next_tick = self.all_ticks[i + 1]
            
            # Skip if current tick is interpolated (already part of a gap)
            if current_tick.is_interpolated:
                continue
            
            time_diff = (next_tick.timestamp - current_tick.timestamp).total_seconds()
            
            if time_diff > gap_threshold_seconds:
                gaps.append({
                    'start_tick': i,
                    'end_tick': i + 1,
                    'start_time': current_tick.timestamp,
                    'end_time': next_tick.timestamp,
                    'duration_seconds': time_diff,
                    'duration_minutes': time_diff / 60,
                    'duration_hours': time_diff / 3600
                })
        
        return gaps
    
    def _get_csv_file_ranges(self):
        """Get the date ranges from CSV files."""
        try:
            data_path = Path('./data') / self.symbol.upper()
            if not data_path.exists():
                return []
            
            csv_files = list(data_path.glob('*.csv'))
            ranges = []
            
            for csv_file in csv_files:
                filename = csv_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    try:
                        start_date = parts[1]
                        end_date = parts[2]
                        ranges.append({
                            'file': csv_file.name,
                            'start_date': start_date,
                            'end_date': end_date
                        })
                    except:
                        pass
            
            return ranges
        except:
            return []
    
    def _load_current_window(self):
        """Load a window of data around current position."""
        start = max(0, self.current_position - self.window_size // 2)
        end = min(self.total_ticks, start + self.window_size)
        start = max(0, end - self.window_size)  # Adjust if near end
        
        self.window_ticks = self.all_ticks[start:end]
        self.window_start = start
        self.local_position = self.current_position - start
    
    def _get_display_data(self):
        """Get data for chart display."""
        center = self.local_position
        half_display = self.display_size // 2
        
        start = max(0, center - half_display)
        end = min(len(self.window_ticks), center + half_display)
        
        display_ticks = self.window_ticks[start:end]
        display_center = center - start
        
        return display_ticks, display_center
    
    def _find_gaps_in_display(self, display_ticks):
        """Find gaps within the current display window."""
        gaps = []
        gap_threshold_seconds = 300  # 5 minutes
        
        for i in range(len(display_ticks) - 1):
            current_tick = display_ticks[i]
            next_tick = display_ticks[i + 1]
            
            time_diff = (next_tick.timestamp - current_tick.timestamp).total_seconds()
            
            if time_diff > gap_threshold_seconds:
                gaps.append({
                    'start_time': current_tick.timestamp,
                    'end_time': next_tick.timestamp,
                    'start_price': current_tick.mid,
                    'end_price': next_tick.mid,
                    'duration_minutes': time_diff / 60
                })
        
        return gaps
    
    def _update_chart_fast(self):
        """Ultra-fast chart update with gap visualization."""
        if self.updating:
            return
        self.updating = True
        
        try:
            display_ticks, center_idx = self._get_display_data()
            
            if not display_ticks:
                return
            
            # Clear previous lines efficiently
            for line in self.current_lines:
                line.remove()
            self.current_lines.clear()
            
            # Prepare data arrays
            times = [tick.timestamp for tick in display_ticks]
            bids = [tick.bid for tick in display_ticks]
            asks = [tick.ask for tick in display_ticks]
            
            # Separate real and interpolated data
            real_indices = [i for i, tick in enumerate(display_ticks) if not tick.is_interpolated]
            interp_indices = [i for i, tick in enumerate(display_ticks) if tick.is_interpolated]
            
            # Plot real data
            if real_indices:
                real_times = [times[i] for i in real_indices]
                real_bids = [bids[i] for i in real_indices]
                real_asks = [asks[i] for i in real_indices]
                
                lines1 = self.ax_chart.plot(real_times, real_bids, 'b-', linewidth=1.2, label='Bid (Real)', alpha=0.8)
                lines2 = self.ax_chart.plot(real_times, real_asks, 'r-', linewidth=1.2, label='Ask (Real)', alpha=0.8)
                self.current_lines.extend(lines1 + lines2)
            
            # Show interpolated data with dashed lines (if any)
            if interp_indices:
                interp_times = [times[i] for i in interp_indices]
                interp_bids = [bids[i] for i in interp_indices]
                interp_asks = [asks[i] for i in interp_indices]
                
                lines3 = self.ax_chart.plot(interp_times, interp_bids, 'b--', linewidth=0.8, label='Bid (Interpolated)', alpha=0.5)
                lines4 = self.ax_chart.plot(interp_times, interp_asks, 'r--', linewidth=0.8, label='Ask (Interpolated)', alpha=0.5)
                self.current_lines.extend(lines3 + lines4)
            
            # Visualize gaps with special indicators
            display_gaps = self._find_gaps_in_display(display_ticks)
            for gap in display_gaps:
                # Draw gap indicator with red shaded area
                gap_rect = self.ax_chart.axvspan(
                    gap['start_time'], gap['end_time'], 
                    alpha=0.3, color='red', label=f'Gap ({gap["duration_minutes"]:.1f}min)'
                )
                self.current_lines.append(gap_rect)
                
                # Add gap text annotation
                gap_center_time = gap['start_time'] + (gap['end_time'] - gap['start_time']) / 2
                gap_center_price = (gap['start_price'] + gap['end_price']) / 2
                
                gap_text = self.ax_chart.annotate(
                    f'GAP\n{gap["duration_minutes"]:.1f}min',
                    xy=(gap_center_time, gap_center_price),
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                    fontsize=8, color='white', weight='bold'
                )
                self.current_lines.append(gap_text)
            
            # Current position marker
            if 0 <= center_idx < len(display_ticks):
                current_tick = display_ticks[center_idx]
                color = 'orange' if current_tick.is_interpolated else 'yellow'
                style = '--' if current_tick.is_interpolated else '-'
                
                line5 = self.ax_chart.axvline(current_tick.timestamp, color=color, linewidth=2, linestyle=style, alpha=0.8)
                self.current_lines.append(line5)
                
                # Year separators
                self._add_year_lines(display_ticks)
            
            # Update axes ranges efficiently
            if times:
                self.ax_chart.set_xlim(times[0], times[-1])
                self.ax_chart.set_ylim(min(bids + asks) * 0.9999, max(bids + asks) * 1.0001)
            
            # Update info panel
            self._update_info_fast()
            
        finally:
            self.updating = False
    
    def _add_year_lines(self, display_ticks):
        """Add year separator lines efficiently."""
        if len(display_ticks) < 2:
            return
        
        current_year = display_ticks[0].timestamp.year
        for tick in display_ticks[1:]:
            if tick.timestamp.year != current_year:
                line = self.ax_chart.axvline(tick.timestamp, color='green', linewidth=1.5, alpha=0.6)
                self.current_lines.append(line)
                current_year = tick.timestamp.year
    
    def _update_info_fast(self):
        """Fast info panel update with enhanced data quality information."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Get current tick
        if self.local_position < len(self.window_ticks):
            current_tick = self.window_ticks[self.local_position]
        else:
            return
        
        spread = (current_tick.ask - current_tick.bid) * 10000
        current_mode = self.movement_modes[self.movement_mode_idx]
        
        # Count interpolated in current window
        interp_count = sum(1 for tick in self.window_ticks if tick.is_interpolated)
        interp_pct = (interp_count / len(self.window_ticks)) * 100 if self.window_ticks else 0
        
        # Find nearby gaps
        nearby_gaps = [gap for gap in self.gaps if abs(gap['start_tick'] - (self.current_position)) < 1000]
        
        status = "INTERPOLATED" if current_tick.is_interpolated else "REAL DATA"
        status_color = "red" if current_tick.is_interpolated else "green"
        
        # Build comprehensive info text
        data_range_text = ""
        if self.data_quality['first_tick'] and self.data_quality['last_tick']:
            data_range_text = f"""
DATA RANGE:
From: {self.data_quality['first_tick'].strftime('%Y-%m-%d %H:%M:%S')}
To: {self.data_quality['last_tick'].strftime('%Y-%m-%d %H:%M:%S')}"""
        
        csv_files_text = ""
        if self.csv_ranges:
            csv_files_text = f"""
CSV FILES: {len(self.csv_ranges)} files"""
        
        info_text = f"""SYMBOL: {self.symbol}
POSITION: {self.current_position + 1:,} / {self.total_ticks:,}
PROGRESS: {(self.current_position / self.total_ticks) * 100:.1f}%

CURRENT TICK:
Time: {current_tick.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Bid: {current_tick.bid:.5f}
Ask: {current_tick.ask:.5f}
Spread: {spread:.2f} pips
Status: {status}

DATA QUALITY:
Real ticks: {self.data_quality['real_percentage']:.1f}%
Quality score: {self.data_quality['quality_score']:.1f}%
Data gaps: {len(self.gaps)} (>5min){data_range_text}{csv_files_text}

WINDOW INFO:
Loaded: {len(self.window_ticks):,} ticks
Interpolated: {interp_count:,} ({interp_pct:.1f}%)
Nearby gaps: {len(nearby_gaps)}

NAVIGATION:
Mode: {current_mode}
Left/Right: Navigate
Up/Down: Change mode
Space: Jump to middle
Home/End: Start/End
Esc: Exit"""
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=7, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        
        # Status indicator
        self.ax_info.text(0.5, 0.02, status, transform=self.ax_info.transAxes,
                         fontsize=10, fontweight='bold', ha='center',
                         color=status_color, bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor="white", edgecolor=status_color, linewidth=2))
    
    def _navigate(self, direction: int):
        """Navigate efficiently with automatic window reloading."""
        if self.movement_mode_idx == 4:  # 10% jump
            step = max(1, self.total_ticks // 10)
        else:
            step = 10 ** self.movement_mode_idx
        
        new_position = self.current_position + (direction * step)
        new_position = max(0, min(new_position, self.total_ticks - 1))
        
        # Check if we need to reload window
        local_pos = new_position - self.window_start
        if local_pos < 0 or local_pos >= len(self.window_ticks):
            self.current_position = new_position
            self._load_current_window()
        else:
            self.current_position = new_position
            self.local_position = local_pos
    
    def _on_key_press(self, event):
        """Handle keyboard input efficiently."""
        if not event.key:
            return
        
        if event.key == 'right':
            self._navigate(1)
        elif event.key == 'left':
            self._navigate(-1)
        elif event.key == 'up':
            self.movement_mode_idx = (self.movement_mode_idx + 1) % len(self.movement_modes)
        elif event.key == 'down':
            self.movement_mode_idx = (self.movement_mode_idx - 1) % len(self.movement_modes)
        elif event.key == 'home':
            self.current_position = 0
            self._load_current_window()
        elif event.key == 'end':
            self.current_position = self.total_ticks - 1
            self._load_current_window()
        elif event.key == ' ':  # Space - jump to middle
            self.current_position = self.total_ticks // 2
            self._load_current_window()
        elif event.key == 'escape':
            self.plt.close('all')
            return
        
        self._update_chart_fast()
        self.fig.canvas.draw_idle()  # Use draw_idle for better performance
    
    def run(self):
        """Launch the enhanced viewer."""
        print(f"\nEnhanced Tick Viewer - {self.symbol}")
        print(f"Dataset: {self.total_ticks:,} ticks")
        print(f"Window size: {self.window_size:,} ticks")
        print("Use arrow keys to navigate, ESC to exit")
        print("Red shaded areas indicate data gaps")
        
        try:
            # Create figure with enhanced layout
            self.fig, (self.ax_chart, self.ax_info) = self.plt.subplots(
                1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [3, 1]}
            )
            
            # Configure chart
            self.ax_chart.set_title(f'{self.symbol} - Enhanced Tick Data Viewer with Gap Analysis', fontsize=14, pad=15)
            self.ax_chart.grid(True, alpha=0.3)
            self.ax_chart.set_facecolor('#fafafa')
            self.ax_chart.set_xlabel('Time', fontsize=10)
            self.ax_chart.set_ylabel('Price', fontsize=10)
            
            # Configure info panel
            self.ax_info.set_xlim(0, 1)
            self.ax_info.set_ylim(0, 1)
            self.ax_info.axis('off')
            
            # Set window title
            try:
                self.fig.canvas.manager.set_window_title(f'Enhanced Tick Viewer - {self.symbol}')
            except:
                pass
            
            # Connect events
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
            # Initial display
            self._update_chart_fast()
            
            print("✓ Enhanced viewer ready! Use keyboard to navigate.")
            print("  - Red areas show data gaps")
            print("  - Green lines mark year boundaries") 
            print("  - Yellow line shows current position")
            self.plt.show()
            
        except Exception as e:
            print(f"Error: {e}")
            self._show_text_preview()
    
    def _show_text_preview(self):
        """Fallback text preview with gap information."""
        print(f"\nBASIC PREVIEW - {self.symbol}")
        print(f"Total ticks: {self.total_ticks:,}")
        print(f"Data quality: {self.data_quality['quality_score']:.1f}%")
        print(f"Data gaps: {len(self.gaps)}")
        
        if self.gaps:
            print("\nLargest gaps:")
            sorted_gaps = sorted(self.gaps, key=lambda x: x['duration_hours'], reverse=True)
            for i, gap in enumerate(sorted_gaps[:5]):
                print(f"  {i+1}. {gap['start_time']} -> {gap['end_time']} ({gap['duration_hours']:.1f}h)")
        
        if self.all_ticks:
            print("\nFirst 5 ticks:")
            for i, tick in enumerate(self.all_ticks[:5]):
                status = "INTERP" if tick.is_interpolated else "REAL"
                print(f"{i+1}: {tick.timestamp} | {tick.bid:.5f}/{tick.ask:.5f} | {status}")


def main():
    """CLI entry point."""
    if len(sys.argv) != 2:
        print("Usage: python tick_visualizer_enhanced.py SYMBOL")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    # Load data without interpolation to see real gaps
    try:
        config = BacktestConfig(
            start_date=datetime(2000, 1, 1),
            end_date=datetime(2100, 1, 1),
            symbols=[symbol],
            initial_balance=100000,
            interpolate_missing_ticks=False  # Don't interpolate to see real gaps
        )
        
        data_loader = DataLoaderFactory.create_loader('./data', config)
        ticks = list(data_loader.load_symbol_data(symbol, config.start_date, config.end_date))
        
        if not ticks:
            print(f"No data found for {symbol}")
            return
        
        print(f"Loaded {len(ticks):,} ticks for {symbol}")
        
        # Launch enhanced viewer
        viewer = EnhancedTickViewer(ticks, symbol)
        viewer.run()
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()