#!/usr/bin/env python3
"""
Log Monitoring Dashboard for NightScan

This script provides a real-time monitoring dashboard for log rotation and disk usage.
It displays comprehensive information about:
- Log file status and sizes
- Disk usage and alerts
- Recent maintenance operations
- System health metrics
- Alert history

Usage:
    # Start dashboard
    python scripts/log_monitor_dashboard.py

    # Dashboard with custom refresh rate
    python scripts/log_monitor_dashboard.py --interval 10

    # Export health report
    python scripts/log_monitor_dashboard.py --export health_report.json

    # Generate summary report
    python scripts/log_monitor_dashboard.py --report

    # Check specific log files
    python scripts/log_monitor_dashboard.py --files "*.log"
"""

import os
import sys
import time
import json
import curses
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from log_rotation_manager import get_log_rotation_manager
from log_utils import get_log_file_info, get_disk_usage


class LogMonitorDashboard:
    """Real-time log monitoring dashboard."""
    
    def __init__(self, refresh_interval: int = 5):
        """Initialize dashboard.
        
        Args:
            refresh_interval: Refresh interval in seconds
        """
        self.refresh_interval = refresh_interval
        self.manager = get_log_rotation_manager()
        self.running = True
        self.show_help = False
        
    def run_curses_dashboard(self, stdscr):
        """Run the curses-based dashboard.
        
        Args:
            stdscr: Curses screen object
        """
        # Configure curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(100) # Refresh timeout
        
        # Color pairs
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Healthy
            curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
            curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)     # Critical
            curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Info
            curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Header
        
        last_update = 0
        
        while self.running:
            current_time = time.time()
            
            # Update data if refresh interval has passed
            if current_time - last_update >= self.refresh_interval:
                try:
                    health_data = self.manager.get_health_status()
                    history_data = self.manager.get_rotation_history(limit=10)
                    last_update = current_time
                except Exception as e:
                    health_data = {'error': str(e)}
                    history_data = []
            
            # Clear screen
            stdscr.clear()
            
            # Handle keyboard input
            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                self.running = False
                break
            elif key == ord('h') or key == ord('H'):
                self.show_help = not self.show_help
            elif key == ord('r') or key == ord('R'):
                # Force refresh
                last_update = 0
                continue
            
            # Display content
            if self.show_help:
                self._draw_help(stdscr)
            else:
                self._draw_dashboard(stdscr, health_data, history_data)
            
            # Refresh screen
            stdscr.refresh()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)
    
    def _draw_help(self, stdscr):
        """Draw help screen."""
        height, width = stdscr.getmaxyx()
        
        help_text = [
            "NightScan Log Monitor Dashboard - Help",
            "=" * 40,
            "",
            "Key Commands:",
            "  q/Q     - Quit dashboard",
            "  h/H     - Toggle help screen",
            "  r/R     - Force refresh",
            "",
            "Dashboard Sections:",
            "  System Status  - Overall health and environment",
            "  Disk Usage     - Space utilization and alerts",
            "  Log Files      - Individual file status",
            "  Recent Ops     - Maintenance operation history",
            "",
            "Status Indicators:",
            "  ✅ Healthy    - All systems normal",
            "  ⚠️ Warning    - Attention needed",
            "  ❌ Critical   - Immediate action required",
            "",
            "Press 'h' to return to dashboard"
        ]
        
        for i, line in enumerate(help_text):
            if i < height - 1:
                stdscr.addstr(i, 0, line[:width-1])
    
    def _draw_dashboard(self, stdscr, health_data: Dict[str, Any], history_data: List[Dict[str, Any]]):
        """Draw the main dashboard."""
        height, width = stdscr.getmaxyx()
        
        if 'error' in health_data:
            self._draw_error(stdscr, health_data['error'])
            return
        
        row = 0
        
        # Header
        header = f"NightScan Log Monitor Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self._add_colored_line(stdscr, row, 0, header, 5)
        row += 1
        
        self._add_line(stdscr, row, 0, "=" * min(len(header), width-1))
        row += 2
        
        # System Status
        row = self._draw_system_status(stdscr, row, health_data)
        row += 1
        
        # Disk Usage
        row = self._draw_disk_usage(stdscr, row, health_data.get('disk_status', {}))
        row += 1
        
        # Log Files Status
        row = self._draw_log_files(stdscr, row, health_data.get('log_files', {}))
        row += 1
        
        # Recent Operations
        row = self._draw_recent_operations(stdscr, row, history_data)
        
        # Footer
        footer_row = height - 2
        if footer_row > row:
            footer = f"Press 'h' for help, 'r' to refresh, 'q' to quit | Refresh: {self.refresh_interval}s"
            self._add_line(stdscr, footer_row, 0, footer[:width-1])
    
    def _draw_system_status(self, stdscr, start_row: int, health_data: Dict[str, Any]) -> int:
        """Draw system status section."""
        row = start_row
        
        self._add_line(stdscr, row, 0, "📊 System Status")
        row += 1
        
        overall_status = health_data.get('overall_status', 'unknown')
        environment = health_data.get('environment', 'unknown')
        
        # Status indicator
        status_color = 1 if overall_status == 'healthy' else (2 if overall_status == 'degraded' else 3)
        status_symbol = "✅" if overall_status == 'healthy' else ("⚠️" if overall_status == 'degraded' else "❌")
        
        self._add_colored_line(stdscr, row, 2, f"{status_symbol} Status: {overall_status.upper()}", status_color)
        row += 1
        
        self._add_line(stdscr, row, 2, f"Environment: {environment}")
        row += 1
        
        # Log configuration
        log_config = health_data.get('log_config', {})
        if log_config:
            self._add_line(stdscr, row, 2, f"Log Directory: {log_config.get('log_dir', 'unknown')}")
            row += 1
            self._add_line(stdscr, row, 2, f"Max File Size: {log_config.get('max_file_size_mb', 0):.1f}MB")
            row += 1
            self._add_line(stdscr, row, 2, f"Retention: {log_config.get('retention_days', 0)} days")
            row += 1
        
        return row
    
    def _draw_disk_usage(self, stdscr, start_row: int, disk_status: Dict[str, Any]) -> int:
        """Draw disk usage section."""
        row = start_row
        
        self._add_line(stdscr, row, 0, "💾 Disk Usage")
        row += 1
        
        if disk_status.get('error'):
            self._add_colored_line(stdscr, row, 2, f"❌ Error: {disk_status['error']}", 3)
            return row + 1
        
        usage_percent = disk_status.get('disk_usage_percent', 0)
        free_gb = disk_status.get('disk_free_gb', 0)
        total_gb = disk_status.get('disk_total_gb', 0)
        log_size_mb = disk_status.get('log_dir_size_mb', 0)
        
        # Usage bar
        bar_width = 30
        filled = int((usage_percent / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        usage_color = 1 if usage_percent < 80 else (2 if usage_percent < 90 else 3)
        
        self._add_colored_line(stdscr, row, 2, f"Usage: [{bar}] {usage_percent:.1f}%", usage_color)
        row += 1
        
        self._add_line(stdscr, row, 2, f"Free Space: {free_gb:.1f}GB / {total_gb:.1f}GB")
        row += 1
        
        self._add_line(stdscr, row, 2, f"Log Directory Size: {log_size_mb:.1f}MB")
        row += 1
        
        # Alert status
        if disk_status.get('alert'):
            alert = disk_status['alert']
            alert_color = 2 if alert['severity'] == 'warning' else 3
            self._add_colored_line(stdscr, row, 2, f"🚨 Alert: {alert['severity'].upper()} - {alert['recommended_action']}", alert_color)
            row += 1
        
        return row
    
    def _draw_log_files(self, stdscr, start_row: int, log_files: Dict[str, Any]) -> int:
        """Draw log files section."""
        row = start_row
        
        self._add_line(stdscr, row, 0, "📄 Log Files")
        row += 1
        
        if log_files.get('error'):
            self._add_colored_line(stdscr, row, 2, f"❌ Error: {log_files['error']}", 3)
            return row + 1
        
        total_files = log_files.get('total_files', 0)
        total_size_mb = log_files.get('total_size_mb', 0)
        all_accessible = log_files.get('all_accessible', False)
        
        access_color = 1 if all_accessible else 3
        access_symbol = "✅" if all_accessible else "❌"
        
        self._add_colored_line(stdscr, row, 2, f"{access_symbol} Accessibility: {'All files accessible' if all_accessible else 'Some files inaccessible'}", access_color)
        row += 1
        
        self._add_line(stdscr, row, 2, f"Total Files: {total_files}")
        row += 1
        
        self._add_line(stdscr, row, 2, f"Total Size: {total_size_mb:.1f}MB")
        row += 1
        
        # Show top files by size
        files = log_files.get('files', [])
        if files:
            self._add_line(stdscr, row, 2, "Top Files:")
            row += 1
            
            # Sort by size and show top 5
            sorted_files = sorted(files, key=lambda f: f.get('info', {}).get('size_mb', 0), reverse=True)
            for file_info in sorted_files[:5]:
                name = file_info.get('name', 'unknown')
                size_mb = file_info.get('info', {}).get('size_mb', 0)
                self._add_line(stdscr, row, 4, f"{name}: {size_mb:.1f}MB")
                row += 1
                
                # Stop if we're running out of space
                if row >= stdscr.getmaxyx()[0] - 5:
                    break
        
        return row
    
    def _draw_recent_operations(self, stdscr, start_row: int, history_data: List[Dict[str, Any]]) -> int:
        """Draw recent operations section."""
        row = start_row
        
        self._add_line(stdscr, row, 0, "🔄 Recent Operations")
        row += 1
        
        if not history_data:
            self._add_line(stdscr, row, 2, "No recent operations")
            return row + 1
        
        for operation in history_data[-5:]:  # Show last 5 operations
            timestamp = operation.get('timestamp', 'unknown')
            op_type = operation.get('operation', 'unknown')
            success = operation.get('success', False)
            files_processed = operation.get('files_processed', 0)
            space_freed = operation.get('space_freed_mb', 0)
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
            
            status_symbol = "✅" if success else "❌"
            status_color = 1 if success else 3
            
            summary = f"{status_symbol} {time_str} {op_type}: {files_processed} files, {space_freed:.1f}MB freed"
            self._add_colored_line(stdscr, row, 2, summary, status_color)
            row += 1
            
            # Stop if we're running out of space
            if row >= stdscr.getmaxyx()[0] - 2:
                break
        
        return row
    
    def _draw_error(self, stdscr, error_message: str):
        """Draw error screen."""
        height, width = stdscr.getmaxyx()
        
        error_lines = [
            "❌ Error occurred while fetching data",
            "",
            f"Error: {error_message}",
            "",
            "Press 'r' to retry or 'q' to quit"
        ]
        
        start_row = height // 2 - len(error_lines) // 2
        
        for i, line in enumerate(error_lines):
            self._add_colored_line(stdscr, start_row + i, 0, line[:width-1], 3)
    
    def _add_line(self, stdscr, row: int, col: int, text: str):
        """Add a line of text to the screen."""
        try:
            height, width = stdscr.getmaxyx()
            if row < height and col < width:
                stdscr.addstr(row, col, text[:width-col-1])
        except curses.error:
            pass  # Ignore screen boundary errors
    
    def _add_colored_line(self, stdscr, row: int, col: int, text: str, color_pair: int):
        """Add a colored line of text to the screen."""
        try:
            height, width = stdscr.getmaxyx()
            if row < height and col < width:
                if curses.has_colors():
                    stdscr.addstr(row, col, text[:width-col-1], curses.color_pair(color_pair))
                else:
                    stdscr.addstr(row, col, text[:width-col-1])
        except curses.error:
            pass  # Ignore screen boundary errors
    
    def export_health_report(self, output_file: str) -> bool:
        """Export comprehensive health report to JSON file.
        
        Args:
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            health_data = self.manager.get_health_status()
            history_data = self.manager.get_rotation_history(limit=50)
            validation_data = self.manager.validate_configuration()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'health_status': health_data,
                'operation_history': history_data,
                'configuration_validation': validation_data,
                'system_info': {
                    'environment': os.environ.get('NIGHTSCAN_ENV', 'development'),
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Health report exported to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export health report: {e}")
            return False
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report.
        
        Returns:
            Summary report as string
        """
        try:
            health_data = self.manager.get_health_status()
            history_data = self.manager.get_rotation_history(limit=10)
            
            lines = [
                "NightScan Log Rotation Summary Report",
                "=" * 40,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "SYSTEM STATUS:",
                f"  Overall Status: {health_data.get('overall_status', 'unknown').upper()}",
                f"  Environment: {health_data.get('environment', 'unknown')}",
                "",
                "DISK USAGE:",
            ]
            
            disk_status = health_data.get('disk_status', {})
            if disk_status.get('error'):
                lines.append(f"  Error: {disk_status['error']}")
            else:
                lines.extend([
                    f"  Usage: {disk_status.get('disk_usage_percent', 0):.1f}%",
                    f"  Free Space: {disk_status.get('disk_free_gb', 0):.1f}GB",
                    f"  Log Directory Size: {disk_status.get('log_dir_size_mb', 0):.1f}MB"
                ])
                
                if disk_status.get('alert'):
                    alert = disk_status['alert']
                    lines.append(f"  Alert: {alert['severity'].upper()} - {alert['recommended_action']}")
            
            lines.extend([
                "",
                "LOG FILES:",
            ])
            
            log_files = health_data.get('log_files', {})
            if log_files.get('error'):
                lines.append(f"  Error: {log_files['error']}")
            else:
                lines.extend([
                    f"  Total Files: {log_files.get('total_files', 0)}",
                    f"  Total Size: {log_files.get('total_size_mb', 0):.1f}MB",
                    f"  All Accessible: {'Yes' if log_files.get('all_accessible', False) else 'No'}"
                ])
            
            lines.extend([
                "",
                "RECENT OPERATIONS:"
            ])
            
            if not history_data:
                lines.append("  No recent operations")
            else:
                for operation in history_data[-5:]:
                    timestamp = operation.get('timestamp', 'unknown')
                    op_type = operation.get('operation', 'unknown')
                    success = operation.get('success', False)
                    files_processed = operation.get('files_processed', 0)
                    space_freed = operation.get('space_freed_mb', 0)
                    
                    status = "SUCCESS" if success else "FAILED"
                    lines.append(f"  {timestamp[:19]} {op_type}: {status} ({files_processed} files, {space_freed:.1f}MB)")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error generating summary report: {e}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NightScan Log Monitor Dashboard")
    
    parser.add_argument('--interval', type=int, default=5,
                       help='Dashboard refresh interval in seconds (default: 5)')
    parser.add_argument('--export', 
                       help='Export health report to JSON file')
    parser.add_argument('--report', action='store_true',
                       help='Generate and display summary report')
    parser.add_argument('--files', 
                       help='Check specific log files (glob pattern)')
    parser.add_argument('--environment',
                       help='Override environment setting')
    
    args = parser.parse_args()
    
    # Set environment if specified
    if args.environment:
        os.environ['NIGHTSCAN_ENV'] = args.environment
    
    # Initialize dashboard
    dashboard = LogMonitorDashboard(refresh_interval=args.interval)
    
    try:
        if args.export:
            # Export health report
            success = dashboard.export_health_report(args.export)
            sys.exit(0 if success else 1)
            
        elif args.report:
            # Generate summary report
            report = dashboard.generate_summary_report()
            print(report)
            sys.exit(0)
            
        elif args.files:
            # Check specific files
            from pathlib import Path
            
            log_dir = dashboard.manager.log_config.log_dir
            files = list(log_dir.glob(args.files))
            
            print(f"Log files matching '{args.files}':")
            print("-" * 40)
            
            for file_path in files:
                info = get_log_file_info(str(file_path))
                if info.get('error'):
                    print(f"❌ {file_path.name}: {info['error']}")
                else:
                    print(f"✅ {file_path.name}: {info['size_mb']:.1f}MB, modified {info['modified_time']}")
            
            sys.exit(0)
            
        else:
            # Run interactive dashboard
            print("Starting NightScan Log Monitor Dashboard...")
            print("Press 'h' for help, 'q' to quit")
            time.sleep(1)
            
            curses.wrapper(dashboard.run_curses_dashboard)
            
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()