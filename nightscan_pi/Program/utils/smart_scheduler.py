"""Smart scheduling system for energy-efficient NightScanPi operations."""
from __future__ import annotations

from datetime import datetime, timedelta
import os
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional

from . import sun_times

logger = logging.getLogger(__name__)

# Camera timing configuration
CAMERA_WINDOW_MINUTES = int(os.getenv("NIGHTSCAN_CAMERA_WINDOW", "30"))  # 30 min before/after sunset/sunrise
SUN_FILE = Path(os.getenv("NIGHTSCAN_SUN_FILE", str(Path.home() / "sun_times.json")))

# WiFi timeout when enabled
WIFI_AUTO_TIMEOUT = int(os.getenv("NIGHTSCAN_WIFI_TIMEOUT", "600"))  # 10 minutes default

# Process management
MAIN_PROCESS_PID_FILE = Path("/tmp/nightscan_main.pid")
WIFI_PROCESS_PID_FILE = Path("/tmp/nightscan_wifi.pid")


class EnergyScheduler:
    """Intelligent energy scheduler for NightScanPi components."""
    
    def __init__(self):
        self.camera_window = timedelta(minutes=CAMERA_WINDOW_MINUTES)
        
    def get_sun_times(self, day: datetime.date) -> tuple[datetime, datetime]:
        """Get sunrise and sunset times for a given day."""
        _, sunrise, sunset = sun_times.get_or_update_sun_times(SUN_FILE, day)
        return sunrise, sunset
        
    def get_camera_periods(self, now: datetime) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
        """Get camera active periods for current day."""
        day = now.date()
        sunrise, sunset = self.get_sun_times(day)
        
        # Evening period: 30 min before sunset to sunset
        evening_start = sunset - self.camera_window
        evening_end = sunset
        
        # Morning period: sunrise to 30 min after sunrise
        morning_start = sunrise
        morning_end = sunrise + self.camera_window
        
        return (evening_start, evening_end), (morning_start, morning_end)
        
    def is_camera_period(self, now: Optional[datetime] = None) -> bool:
        """Check if camera should be active now."""
        if now is None:
            now = datetime.now()
            
        (evening_start, evening_end), (morning_start, morning_end) = self.get_camera_periods(now)
        
        # Check if in today's periods
        if evening_start <= now <= evening_end or morning_start <= now <= morning_end:
            return True
            
        # Check yesterday's morning period (in case we're early morning)
        yesterday = now.date() - timedelta(days=1)
        yesterday_sunrise, _ = self.get_sun_times(yesterday)
        yesterday_morning_end = yesterday_sunrise + self.camera_window
        
        if yesterday_sunrise <= now <= yesterday_morning_end:
            return True
            
        return False
        
    def is_audio_only_period(self, now: Optional[datetime] = None) -> bool:
        """Check if only audio detection should be active (no camera)."""
        if now is None:
            now = datetime.now()
            
        # Audio is active during night hours but outside camera windows
        hour = now.hour
        is_night = hour >= 18 or hour <= 10
        
        return is_night and not self.is_camera_period(now)
        
    def should_system_sleep(self, now: Optional[datetime] = None) -> bool:
        """Check if entire system should sleep (day time)."""
        if now is None:
            now = datetime.now()
            
        hour = now.hour
        # Sleep during day hours (11 AM to 5 PM)
        return 11 <= hour < 17
        
    def next_camera_period(self, now: Optional[datetime] = None) -> datetime:
        """Get the next time camera should be active."""
        if now is None:
            now = datetime.now()
            
        (evening_start, evening_end), (morning_start, morning_end) = self.get_camera_periods(now)
        
        # If before evening period today
        if now < evening_start:
            return evening_start
            
        # If between evening and morning periods
        if evening_end < now < morning_start:
            return morning_start
            
        # If after morning period, get tomorrow's evening
        if now > morning_end:
            tomorrow = now.date() + timedelta(days=1)
            tomorrow_sunrise, tomorrow_sunset = self.get_sun_times(tomorrow)
            return tomorrow_sunset - self.camera_window
            
        # Currently in a camera period
        return now
        
    def get_operation_mode(self, now: Optional[datetime] = None) -> str:
        """Get current operation mode."""
        if now is None:
            now = datetime.now()
            
        if self.should_system_sleep(now):
            return "sleep"
        elif self.is_camera_period(now):
            return "camera_active"
        elif self.is_audio_only_period(now):
            return "audio_only"
        else:
            return "minimal"


class WiFiManager:
    """Smart WiFi management for on-demand activation."""
    
    def __init__(self):
        self.status_file = Path("/tmp/wifi_active.status")
        
    def is_wifi_active(self) -> bool:
        """Check if WiFi is currently active."""
        try:
            if not self.status_file.exists():
                return False
            timestamp = float(self.status_file.read_text().strip())
            # Check if still within timeout period
            return (time.time() - timestamp) < WIFI_AUTO_TIMEOUT
        except (ValueError, FileNotFoundError):
            return False
            
    def activate_wifi(self, duration_minutes: Optional[int] = None) -> bool:
        """Activate WiFi for specified duration."""
        try:
            # Bring up WiFi interface
            result = subprocess.run(["sudo", "ifconfig", "wlan0", "up"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to bring up WiFi: {result.stderr}")
                return False
                
            # Start WiFi service
            result = subprocess.run(["sudo", "systemctl", "start", "nightscan-wifi"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"WiFi service start warning: {result.stderr}")
                
            # Record activation time
            self.status_file.write_text(str(time.time()))
            
            # Schedule auto-deactivation
            timeout = duration_minutes * 60 if duration_minutes else WIFI_AUTO_TIMEOUT
            subprocess.run(["sudo", "systemctl", "start", f"nightscan-wifi-timeout@{timeout}"], 
                          check=False)
            
            logger.info(f"WiFi activated for {timeout // 60} minutes")
            return True
            
        except Exception as e:
            logger.error(f"WiFi activation failed: {e}")
            return False
            
    def deactivate_wifi(self) -> bool:
        """Deactivate WiFi and associated services."""
        try:
            # Stop WiFi service
            subprocess.run(["sudo", "systemctl", "stop", "nightscan-wifi"], check=False)
            
            # Bring down WiFi interface
            result = subprocess.run(["sudo", "ifconfig", "wlan0", "down"], 
                                  capture_output=True, text=True)
            
            # Remove status file
            self.status_file.unlink(missing_ok=True)
            
            logger.info("WiFi deactivated")
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"WiFi deactivation failed: {e}")
            return False
            
    def extend_wifi_session(self, additional_minutes: int = 10) -> bool:
        """Extend current WiFi session."""
        if not self.is_wifi_active():
            return False
            
        try:
            # Update timestamp
            self.status_file.write_text(str(time.time()))
            logger.info(f"WiFi session extended by {additional_minutes} minutes")
            return True
        except Exception as e:
            logger.error(f"Failed to extend WiFi session: {e}")
            return False


class ProcessManager:
    """Manage NightScanPi processes based on energy schedule."""
    
    def __init__(self):
        self.scheduler = EnergyScheduler()
        self.wifi_manager = WiFiManager()
        
    def start_main_detection(self) -> bool:
        """Start main detection process."""
        try:
            if MAIN_PROCESS_PID_FILE.exists():
                logger.info("Main detection already running")
                return True
                
            # Start main process
            process = subprocess.Popen([
                "python3", "-m", "NightScanPi.Program.main"
            ], cwd="/home/pi/nightscan")
            
            MAIN_PROCESS_PID_FILE.write_text(str(process.pid))
            logger.info(f"Started main detection process (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start main detection: {e}")
            return False
            
    def stop_main_detection(self) -> bool:
        """Stop main detection process."""
        try:
            if not MAIN_PROCESS_PID_FILE.exists():
                return True
                
            pid = int(MAIN_PROCESS_PID_FILE.read_text().strip())
            subprocess.run(["sudo", "kill", "-TERM", str(pid)], check=False)
            
            # Wait for graceful shutdown
            time.sleep(2)
            
            # Force kill if still running
            try:
                subprocess.run(["sudo", "kill", "-KILL", str(pid)], check=False)
            except:
                pass
                
            MAIN_PROCESS_PID_FILE.unlink(missing_ok=True)
            logger.info("Stopped main detection process")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop main detection: {e}")
            return False
            
    def get_system_status(self) -> dict:
        """Get comprehensive system status."""
        now = datetime.now()
        mode = self.scheduler.get_operation_mode(now)
        
        return {
            "timestamp": now.isoformat(),
            "operation_mode": mode,
            "camera_active": self.scheduler.is_camera_period(now),
            "audio_only": self.scheduler.is_audio_only_period(now),
            "should_sleep": self.scheduler.should_system_sleep(now),
            "wifi_active": self.wifi_manager.is_wifi_active(),
            "main_process_running": MAIN_PROCESS_PID_FILE.exists(),
            "next_camera_period": self.scheduler.next_camera_period(now).isoformat(),
            "camera_periods_today": {
                "evening": [
                    period[0].isoformat(), period[1].isoformat()
                ] for period in [self.scheduler.get_camera_periods(now)]
            }
        }


# Global instances
energy_scheduler = EnergyScheduler()
wifi_manager = WiFiManager()
process_manager = ProcessManager()


def get_energy_scheduler() -> EnergyScheduler:
    """Get global energy scheduler instance."""
    return energy_scheduler


def get_wifi_manager() -> WiFiManager:
    """Get global WiFi manager instance."""
    return wifi_manager


def get_process_manager() -> ProcessManager:
    """Get global process manager instance."""
    return process_manager


if __name__ == "__main__":
    # Test the scheduler
    scheduler = EnergyScheduler()
    now = datetime.now()
    
    print(f"Current time: {now}")
    print(f"Operation mode: {scheduler.get_operation_mode(now)}")
    print(f"Camera period: {scheduler.is_camera_period(now)}")
    print(f"Audio only: {scheduler.is_audio_only_period(now)}")
    print(f"Should sleep: {scheduler.should_system_sleep(now)}")
    print(f"Next camera period: {scheduler.next_camera_period(now)}")