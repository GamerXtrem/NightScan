#!/usr/bin/env python3
"""
Tests for Smart Scheduler module.

Tests energy-efficient scheduling based on sunrise/sunset and system management.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from datetime import datetime, time, timedelta
import json
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_scheduler import SmartScheduler, OperationMode


class TestSmartScheduler(unittest.TestCase):
    """Test cases for Smart Scheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock sunrise/sunset times
        self.mock_sunrise = time(6, 30)  # 6:30 AM
        self.mock_sunset = time(19, 30)  # 7:30 PM
        
        # Create scheduler with test location
        self.scheduler = SmartScheduler(
            latitude=45.5,
            longitude=-73.6,
            timezone='America/Montreal'
        )
        
        # Mock astral sun calculation
        self.sun_patcher = patch('smart_scheduler.sun')
        self.mock_sun = self.sun_patcher.start()
        
        # Set up sun mock to return our test times
        mock_sun_times = {
            'sunrise': datetime.combine(datetime.today(), self.mock_sunrise),
            'sunset': datetime.combine(datetime.today(), self.mock_sunset)
        }
        self.mock_sun.return_value = mock_sun_times
    
    def tearDown(self):
        """Clean up after tests."""
        self.sun_patcher.stop()
    
    def test_initialization(self):
        """Test scheduler initialization."""
        self.assertEqual(self.scheduler.latitude, 45.5)
        self.assertEqual(self.scheduler.longitude, -73.6)
        self.assertEqual(self.scheduler.timezone, 'America/Montreal')
        self.assertEqual(self.scheduler.current_mode, OperationMode.ADAPTIVE)
        self.assertTrue(self.scheduler.camera_active)
    
    def test_sunrise_sunset_calculation(self):
        """Test sunrise/sunset time calculation."""
        sunrise, sunset = self.scheduler.get_sun_times()
        
        self.assertEqual(sunrise, self.mock_sunrise)
        self.assertEqual(sunset, self.mock_sunset)
        
        # Verify astral was called with correct parameters
        self.mock_sun.assert_called()
    
    def test_is_night_time_detection(self):
        """Test night time detection."""
        # Test night time (before sunrise)
        with patch('smart_scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.combine(datetime.today(), time(5, 0))
            mock_datetime.combine = datetime.combine
            self.assertTrue(self.scheduler.is_night_time())
        
        # Test day time
        with patch('smart_scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.combine(datetime.today(), time(12, 0))
            mock_datetime.combine = datetime.combine
            self.assertFalse(self.scheduler.is_night_time())
        
        # Test night time (after sunset)
        with patch('smart_scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.combine(datetime.today(), time(21, 0))
            mock_datetime.combine = datetime.combine
            self.assertTrue(self.scheduler.is_night_time())
    
    def test_operation_mode_determination(self):
        """Test operation mode based on time of day."""
        # Night mode
        with patch.object(self.scheduler, 'is_night_time', return_value=True):
            mode = self.scheduler.get_current_operation_mode()
            self.assertEqual(mode, OperationMode.NIGHT)
        
        # Day mode
        with patch.object(self.scheduler, 'is_night_time', return_value=False):
            mode = self.scheduler.get_current_operation_mode()
            self.assertEqual(mode, OperationMode.DAY)
        
        # Override with ECO mode
        self.scheduler.set_mode(OperationMode.ECO)
        mode = self.scheduler.get_current_operation_mode()
        self.assertEqual(mode, OperationMode.ECO)
    
    def test_camera_scheduling(self):
        """Test camera activation scheduling."""
        # Night time - camera should be active
        with patch.object(self.scheduler, 'is_night_time', return_value=True):
            self.scheduler.update_camera_state()
            self.assertTrue(self.scheduler.camera_active)
        
        # Day time in adaptive mode - camera should be inactive
        with patch.object(self.scheduler, 'is_night_time', return_value=False):
            self.scheduler.update_camera_state()
            self.assertFalse(self.scheduler.camera_active)
        
        # ALWAYS_ON mode - camera should always be active
        self.scheduler.set_mode(OperationMode.ALWAYS_ON)
        self.scheduler.update_camera_state()
        self.assertTrue(self.scheduler.camera_active)
    
    @patch('subprocess.run')
    def test_wifi_management(self, mock_subprocess):
        """Test WiFi enable/disable functionality."""
        # Enable WiFi
        self.scheduler.enable_wifi()
        mock_subprocess.assert_called_with(
            ['sudo', 'rfkill', 'unblock', 'wifi'],
            capture_output=True,
            text=True
        )
        self.assertTrue(self.scheduler.wifi_enabled)
        
        # Disable WiFi
        mock_subprocess.reset_mock()
        self.scheduler.disable_wifi()
        mock_subprocess.assert_called_with(
            ['sudo', 'rfkill', 'block', 'wifi'],
            capture_output=True,
            text=True
        )
        self.assertFalse(self.scheduler.wifi_enabled)
    
    @patch('subprocess.run')
    def test_wifi_on_demand(self, mock_subprocess):
        """Test WiFi on-demand activation."""
        # Set up successful command execution
        mock_subprocess.return_value.returncode = 0
        
        # Activate WiFi on demand
        success = self.scheduler.activate_wifi_on_demand(duration_minutes=30)
        
        self.assertTrue(success)
        self.assertTrue(self.scheduler.wifi_enabled)
        
        # Verify timer was set
        self.assertIsNotNone(self.scheduler.wifi_timer)
    
    def test_process_management(self):
        """Test process lifecycle management."""
        # Mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        
        # Start process
        self.scheduler.managed_processes['test_process'] = mock_process
        
        # Check if running
        is_running = self.scheduler.is_process_running('test_process')
        self.assertTrue(is_running)
        
        # Stop process
        self.scheduler.stop_process('test_process')
        mock_process.terminate.assert_called_once()
    
    @patch('subprocess.Popen')
    def test_start_night_capture(self, mock_popen):
        """Test starting night capture process."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        self.scheduler.start_night_capture()
        
        # Verify process was started with correct command
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertIn('python3', args[0])
        self.assertIn('night_capture.py', args[1])
        
        # Verify process is tracked
        self.assertIn('night_capture', self.scheduler.managed_processes)
    
    def test_schedule_configuration(self):
        """Test schedule configuration."""
        # Default schedule
        schedule = self.scheduler.get_schedule_config()
        self.assertIn('mode', schedule)
        self.assertIn('sunrise', schedule)
        self.assertIn('sunset', schedule)
        self.assertIn('camera_active_hours', schedule)
        
        # Custom schedule
        custom_hours = [(time(22, 0), time(6, 0))]
        self.scheduler.set_camera_hours(custom_hours)
        
        schedule = self.scheduler.get_schedule_config()
        self.assertEqual(len(schedule['camera_active_hours']), 1)
        self.assertEqual(schedule['camera_active_hours'][0], ('22:00', '06:00'))
    
    def test_eco_mode_behavior(self):
        """Test ECO mode resource management."""
        self.scheduler.set_mode(OperationMode.ECO)
        
        # In ECO mode, WiFi should be disabled by default
        with patch.object(self.scheduler, 'disable_wifi') as mock_disable:
            self.scheduler.apply_eco_settings()
            mock_disable.assert_called_once()
        
        # Camera should only be active during specific hours
        with patch.object(self.scheduler, 'is_night_time', return_value=False):
            self.scheduler.update_camera_state()
            self.assertFalse(self.scheduler.camera_active)
    
    def test_system_status_reporting(self):
        """Test system status reporting."""
        status = self.scheduler.get_system_status()
        
        self.assertIn('mode', status)
        self.assertIn('camera_active', status)
        self.assertIn('wifi_enabled', status)
        self.assertIn('active_processes', status)
        self.assertIn('sunrise', status)
        self.assertIn('sunset', status)
        self.assertIn('is_night', status)
        self.assertIn('timestamp', status)
    
    def test_schedule_persistence(self):
        """Test saving and loading schedule configuration."""
        # Configure custom settings
        self.scheduler.set_mode(OperationMode.NIGHT_ONLY)
        custom_hours = [(time(21, 0), time(5, 0))]
        self.scheduler.set_camera_hours(custom_hours)
        
        # Save configuration
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            self.scheduler.save_schedule('/tmp/test_schedule.json')
            
            # Verify JSON was written
            mock_file.write.assert_called()
            written_data = mock_file.write.call_args[0][0]
            config = json.loads(written_data)
            
            self.assertEqual(config['mode'], 'night_only')
            self.assertEqual(len(config['camera_hours']), 1)
    
    def test_load_schedule_configuration(self):
        """Test loading schedule from file."""
        test_config = {
            'mode': 'eco',
            'camera_hours': [['22:00', '06:00']],
            'wifi_schedule': 'on_demand'
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(test_config))):
            self.scheduler.load_schedule('/tmp/test_schedule.json')
            
            self.assertEqual(self.scheduler.current_mode, OperationMode.ECO)
            self.assertEqual(len(self.scheduler.camera_hours), 1)
    
    def test_adaptive_mode_transitions(self):
        """Test adaptive mode transitions based on conditions."""
        self.scheduler.set_mode(OperationMode.ADAPTIVE)
        
        # Simulate day to night transition
        with patch.object(self.scheduler, 'is_night_time', side_effect=[False, True]):
            # Day time
            self.scheduler.update_camera_state()
            self.assertFalse(self.scheduler.camera_active)
            
            # Night time
            self.scheduler.update_camera_state()
            self.assertTrue(self.scheduler.camera_active)
    
    @patch('subprocess.run')
    def test_system_optimization(self, mock_subprocess):
        """Test system optimization for power saving."""
        mock_subprocess.return_value.returncode = 0
        
        self.scheduler.optimize_system_resources()
        
        # Verify optimization commands were run
        calls = mock_subprocess.call_args_list
        
        # Check for CPU governor setting
        cpu_gov_call = any('cpufreq-set' in str(call) for call in calls)
        self.assertTrue(cpu_gov_call)
    
    def test_error_handling_in_sun_calculation(self):
        """Test error handling when sun calculation fails."""
        # Mock sun calculation failure
        self.mock_sun.side_effect = Exception("Location error")
        
        # Should return default times
        sunrise, sunset = self.scheduler.get_sun_times()
        
        # Default times
        self.assertEqual(sunrise, time(6, 0))
        self.assertEqual(sunset, time(18, 0))
    
    def test_concurrent_process_management(self):
        """Test managing multiple processes concurrently."""
        # Mock multiple processes
        processes = {
            'capture': MagicMock(),
            'upload': MagicMock(),
            'analysis': MagicMock()
        }
        
        for name, proc in processes.items():
            proc.poll.return_value = None  # All running
            self.scheduler.managed_processes[name] = proc
        
        # Get active processes
        active = self.scheduler.get_active_processes()
        self.assertEqual(len(active), 3)
        
        # Stop all processes
        self.scheduler.stop_all_processes()
        
        for proc in processes.values():
            proc.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()