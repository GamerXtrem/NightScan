#!/usr/bin/env python3
"""
Tests for IR Night Vision System module.

Tests IR camera control, LED management, and day/night detection.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from datetime import datetime, time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock hardware modules before import
sys.modules['RPi'] = MagicMock()
sys.modules['RPi.GPIO'] = MagicMock()
sys.modules['picamera2'] = MagicMock()
sys.modules['cv2'] = MagicMock()

from ir_night_vision import IRNightVision, DayNightMode


class TestIRNightVision(unittest.TestCase):
    """Test cases for IR Night Vision system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock GPIO
        self.gpio_mock = sys.modules['RPi.GPIO']
        self.gpio_mock.BCM = 'BCM'
        self.gpio_mock.OUT = 'OUT'
        self.gpio_mock.HIGH = 1
        self.gpio_mock.LOW = 0
        
        # Mock camera
        self.camera_mock = MagicMock()
        sys.modules['picamera2'].Picamera2.return_value = self.camera_mock
        
        # Mock cv2
        self.cv2_mock = sys.modules['cv2']
        
        # Create instance with mocked components
        self.ir_system = IRNightVision(
            ir_cut_pin=17,
            ir_led_pin=18,
            ambient_threshold=30
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Reset GPIO mock
        self.gpio_mock.reset_mock()
    
    def test_initialization(self):
        """Test IR system initialization."""
        # Verify GPIO setup was called
        self.gpio_mock.setmode.assert_called_with('BCM')
        self.gpio_mock.setup.assert_any_call(17, 'OUT')  # IR-CUT pin
        self.gpio_mock.setup.assert_any_call(18, 'OUT')  # IR LED pin
        
        # Verify PWM setup
        self.gpio_mock.PWM.assert_called_with(18, 1000)  # 1kHz PWM
        
        # Verify initial state
        self.assertEqual(self.ir_system.current_mode, DayNightMode.AUTO)
        self.assertFalse(self.ir_system.is_night_mode)
    
    def test_day_mode_activation(self):
        """Test switching to day mode."""
        self.ir_system.set_day_mode()
        
        # Verify IR-CUT filter is enabled (HIGH)
        self.gpio_mock.output.assert_any_call(17, 1)
        
        # Verify IR LEDs are off
        self.ir_system.pwm.stop.assert_called()
        
        # Verify state
        self.assertFalse(self.ir_system.is_night_mode)
        self.assertEqual(self.ir_system.current_mode, DayNightMode.DAY)
    
    def test_night_mode_activation(self):
        """Test switching to night mode."""
        self.ir_system.set_night_mode(led_brightness=75)
        
        # Verify IR-CUT filter is disabled (LOW)
        self.gpio_mock.output.assert_any_call(17, 0)
        
        # Verify IR LEDs are on with correct brightness
        self.ir_system.pwm.start.assert_called_with(75)
        
        # Verify state
        self.assertTrue(self.ir_system.is_night_mode)
        self.assertEqual(self.ir_system.current_mode, DayNightMode.NIGHT)
        self.assertEqual(self.ir_system.led_brightness, 75)
    
    def test_led_brightness_limits(self):
        """Test LED brightness value limits."""
        # Test maximum brightness
        self.ir_system.set_night_mode(led_brightness=150)
        self.ir_system.pwm.start.assert_called_with(100)  # Clamped to 100
        
        # Test minimum brightness
        self.ir_system.set_night_mode(led_brightness=-10)
        self.ir_system.pwm.start.assert_called_with(0)  # Clamped to 0
    
    def test_manual_mode_override(self):
        """Test manual mode overrides auto detection."""
        # Set to manual night mode
        self.ir_system.set_night_mode()
        self.assertEqual(self.ir_system.current_mode, DayNightMode.NIGHT)
        
        # Try auto detection - should not change
        with patch.object(self.ir_system, 'detect_ambient_light', return_value=100):
            self.ir_system.update_auto_mode()
            self.assertTrue(self.ir_system.is_night_mode)  # Still night
    
    @patch('numpy.array')
    def test_ambient_light_detection(self, mock_np_array):
        """Test ambient light detection from image."""
        # Mock bright image
        mock_image = MagicMock()
        mock_np_array.return_value = mock_image
        self.cv2_mock.cvtColor.return_value = mock_image
        
        # Mock mean calculation for bright image
        with patch('numpy.mean', return_value=150):
            brightness = self.ir_system.detect_ambient_light(mock_image)
            self.assertEqual(brightness, 150)
            self.cv2_mock.cvtColor.assert_called_once()
        
        # Mock dark image
        with patch('numpy.mean', return_value=20):
            brightness = self.ir_system.detect_ambient_light(mock_image)
            self.assertEqual(brightness, 20)
    
    def test_auto_mode_switching(self):
        """Test automatic day/night switching."""
        # Set to auto mode
        self.ir_system.set_auto_mode()
        self.assertEqual(self.ir_system.current_mode, DayNightMode.AUTO)
        
        # Mock dark conditions
        with patch.object(self.ir_system, 'detect_ambient_light', return_value=20):
            mock_image = MagicMock()
            self.ir_system.update_auto_mode(mock_image)
            self.assertTrue(self.ir_system.is_night_mode)
        
        # Mock bright conditions
        with patch.object(self.ir_system, 'detect_ambient_light', return_value=80):
            self.ir_system.update_auto_mode(mock_image)
            self.assertFalse(self.ir_system.is_night_mode)
    
    def test_hysteresis_in_auto_mode(self):
        """Test hysteresis prevents rapid switching."""
        self.ir_system.set_auto_mode()
        
        # Start in day mode
        self.assertFalse(self.ir_system.is_night_mode)
        
        # Light level just below threshold - should switch to night
        with patch.object(self.ir_system, 'detect_ambient_light', return_value=29):
            self.ir_system.update_auto_mode(MagicMock())
            self.assertTrue(self.ir_system.is_night_mode)
        
        # Light level just above threshold - should stay in night (hysteresis)
        with patch.object(self.ir_system, 'detect_ambient_light', return_value=31):
            self.ir_system.update_auto_mode(MagicMock())
            self.assertTrue(self.ir_system.is_night_mode)  # Still night
        
        # Light level well above threshold - should switch to day
        with patch.object(self.ir_system, 'detect_ambient_light', return_value=45):
            self.ir_system.update_auto_mode(MagicMock())
            self.assertFalse(self.ir_system.is_night_mode)
    
    def test_time_based_switching(self):
        """Test time-based day/night switching."""
        # Test night time
        with patch('ir_night_vision.datetime') as mock_datetime:
            mock_datetime.now.return_value.time.return_value = time(22, 30)  # 10:30 PM
            is_night = self.ir_system.is_night_time()
            self.assertTrue(is_night)
        
        # Test day time
        with patch('ir_night_vision.datetime') as mock_datetime:
            mock_datetime.now.return_value.time.return_value = time(14, 30)  # 2:30 PM
            is_night = self.ir_system.is_night_time()
            self.assertFalse(is_night)
        
        # Test edge case - sunset time
        with patch('ir_night_vision.datetime') as mock_datetime:
            mock_datetime.now.return_value.time.return_value = time(19, 0)  # 7:00 PM
            is_night = self.ir_system.is_night_time()
            self.assertTrue(is_night)
    
    def test_capture_with_ir(self):
        """Test image capture with IR settings."""
        # Mock camera capture
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera_mock.capture_array.return_value = mock_image
        
        # Capture in night mode
        self.ir_system.set_night_mode(led_brightness=80)
        image = self.ir_system.capture_image()
        
        # Verify camera was configured for IR
        config = self.camera_mock.create_preview_configuration.call_args[1]
        self.assertEqual(config.get('main', {}).get('format', ''), 'RGB888')
        
        # Verify image was captured
        self.camera_mock.capture_array.assert_called_once()
        self.assertIsNotNone(image)
    
    def test_adaptive_led_brightness(self):
        """Test adaptive LED brightness based on conditions."""
        # Test low ambient light - high LED brightness
        brightness = self.ir_system.calculate_adaptive_brightness(ambient_light=10)
        self.assertEqual(brightness, 90)  # High brightness for very dark
        
        # Test medium ambient light - medium LED brightness
        brightness = self.ir_system.calculate_adaptive_brightness(ambient_light=25)
        self.assertEqual(brightness, 60)  # Medium brightness
        
        # Test high ambient light - low LED brightness
        brightness = self.ir_system.calculate_adaptive_brightness(ambient_light=40)
        self.assertEqual(brightness, 30)  # Low brightness for twilight
    
    def test_cleanup(self):
        """Test cleanup releases resources."""
        # Create new instance to test full lifecycle
        ir_system = IRNightVision()
        
        # Perform cleanup
        ir_system.cleanup()
        
        # Verify PWM was stopped
        ir_system.pwm.stop.assert_called()
        
        # Verify GPIO was cleaned up
        self.gpio_mock.cleanup.assert_called()
        
        # Verify camera was closed
        ir_system.camera.close.assert_called()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with IRNightVision() as ir_system:
            # Use the system
            ir_system.set_night_mode()
            self.assertTrue(ir_system.is_night_mode)
        
        # Verify cleanup was called after context exit
        self.gpio_mock.cleanup.assert_called()
    
    def test_get_status(self):
        """Test status reporting."""
        # Set specific state
        self.ir_system.set_night_mode(led_brightness=70)
        
        status = self.ir_system.get_status()
        
        self.assertEqual(status['mode'], 'night')
        self.assertTrue(status['is_night'])
        self.assertEqual(status['led_brightness'], 70)
        self.assertEqual(status['ir_cut_active'], False)  # LOW in night mode
        self.assertEqual(status['ambient_threshold'], 30)
        self.assertIn('timestamp', status)
    
    def test_mode_change_callback(self):
        """Test callback execution on mode change."""
        callback_called = False
        callback_mode = None
        
        def test_callback(mode):
            nonlocal callback_called, callback_mode
            callback_called = True
            callback_mode = mode
        
        # Register callback
        self.ir_system.on_mode_change = test_callback
        
        # Change mode
        self.ir_system.set_night_mode()
        
        # Verify callback was called
        self.assertTrue(callback_called)
        self.assertEqual(callback_mode, DayNightMode.NIGHT)
    
    def test_error_handling_in_capture(self):
        """Test error handling during capture."""
        # Mock camera error
        self.camera_mock.capture_array.side_effect = Exception("Camera error")
        
        # Should handle error gracefully
        image = self.ir_system.capture_image()
        self.assertIsNone(image)
    
    def test_multiple_mode_switches(self):
        """Test rapid mode switching doesn't cause issues."""
        # Rapidly switch modes
        for _ in range(10):
            self.ir_system.set_day_mode()
            self.ir_system.set_night_mode()
            self.ir_system.set_auto_mode()
        
        # System should still be functional
        status = self.ir_system.get_status()
        self.assertEqual(status['mode'], 'auto')


if __name__ == '__main__':
    unittest.main()