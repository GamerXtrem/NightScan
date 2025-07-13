#!/usr/bin/env python3
"""
Tests for Camera Validator module.

Tests camera validation, quality assessment, and benchmarking.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import sys
import os
import json
import tempfile
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock hardware and camera modules
sys.modules['picamera2'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['RPi'] = MagicMock()
sys.modules['RPi.GPIO'] = MagicMock()

from camera_validator import CameraValidator, ValidationResult


class TestCameraValidator(unittest.TestCase):
    """Test cases for Camera Validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock subprocess for system commands
        self.subprocess_patcher = patch('camera_validator.subprocess')
        self.mock_subprocess = self.subprocess_patcher.start()
        
        # Mock camera
        self.camera_mock = MagicMock()
        sys.modules['picamera2'].Picamera2.return_value = self.camera_mock
        
        # Mock cv2
        self.cv2_mock = sys.modules['cv2']
        
        # Create validator instance
        self.validator = CameraValidator()
    
    def tearDown(self):
        """Clean up after tests."""
        self.subprocess_patcher.stop()
    
    def test_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator.camera)
        self.assertEqual(len(self.validator.test_results), 0)
        self.assertIsNone(self.validator.report)
    
    def test_camera_detection(self):
        """Test camera hardware detection."""
        # Mock successful detection
        self.mock_subprocess.run.return_value.returncode = 0
        self.mock_subprocess.run.return_value.stdout = "supported=1 detected=1"
        
        result = self.validator.test_camera_detection()
        
        self.assertEqual(result.test_name, "Camera Detection")
        self.assertTrue(result.passed)
        self.assertIn("Camera detected", result.message)
        
        # Verify command was called
        self.mock_subprocess.run.assert_called_with(
            ['vcgencmd', 'get_camera'],
            capture_output=True,
            text=True
        )
    
    def test_camera_not_detected(self):
        """Test when camera is not detected."""
        # Mock failed detection
        self.mock_subprocess.run.return_value.returncode = 0
        self.mock_subprocess.run.return_value.stdout = "supported=1 detected=0"
        
        result = self.validator.test_camera_detection()
        
        self.assertFalse(result.passed)
        self.assertIn("not detected", result.message)
    
    def test_basic_capture(self):
        """Test basic image capture."""
        # Mock successful capture
        mock_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.camera_mock.capture_array.return_value = mock_image
        
        result = self.validator.test_basic_capture()
        
        self.assertTrue(result.passed)
        self.assertIn("1920x1080", result.message)
        self.camera_mock.capture_array.assert_called_once()
    
    def test_capture_failure(self):
        """Test capture failure handling."""
        # Mock capture error
        self.camera_mock.capture_array.side_effect = Exception("Camera error")
        
        result = self.validator.test_basic_capture()
        
        self.assertFalse(result.passed)
        self.assertIn("Failed", result.message)
        self.assertIn("Camera error", result.details['error'])
    
    def test_resolution_tests(self):
        """Test different resolution captures."""
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        
        for width, height in resolutions:
            # Mock capture at specific resolution
            mock_image = np.zeros((height, width, 3), dtype=np.uint8)
            self.camera_mock.capture_array.return_value = mock_image
            
            result = self.validator.test_resolution(width, height)
            
            self.assertTrue(result.passed)
            self.assertIn(f"{width}x{height}", result.message)
    
    def test_image_quality_assessment(self):
        """Test image quality metrics."""
        # Create test image with known properties
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.camera_mock.capture_array.return_value = test_image
        
        # Mock OpenCV calculations
        self.cv2_mock.Laplacian.return_value = np.random.rand(480, 640)
        self.cv2_mock.mean.return_value = (128, 128, 128, 0)
        self.cv2_mock.meanStdDev.return_value = (None, np.array([[30], [30], [30]]))
        
        with patch('numpy.var', return_value=100):
            result = self.validator.test_image_quality()
        
        self.assertTrue(result.passed)
        self.assertIn('sharpness', result.details)
        self.assertIn('brightness', result.details)
        self.assertIn('contrast', result.details)
    
    def test_low_light_performance(self):
        """Test low light performance assessment."""
        # Create dark image
        dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 20  # Very dark
        self.camera_mock.capture_array.return_value = dark_image
        
        # Mock mean brightness
        self.cv2_mock.mean.return_value = (20, 20, 20, 0)
        
        result = self.validator.test_low_light_performance()
        
        self.assertTrue(result.passed)
        self.assertIn("Low light", result.test_name)
        self.assertLess(result.details['mean_brightness'], 50)
        self.assertIn('noise_estimate', result.details)
    
    @patch('time.time')
    def test_capture_speed(self, mock_time):
        """Test capture speed benchmarking."""
        # Mock time progression
        mock_time.side_effect = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 100ms per capture
        
        # Mock captures
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera_mock.capture_array.return_value = mock_image
        
        result = self.validator.test_capture_speed(num_captures=5)
        
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.details['avg_time_ms'], 100, places=0)
        self.assertAlmostEqual(result.details['fps'], 10, places=0)
    
    def test_exposure_modes(self):
        """Test different exposure modes."""
        modes = ['auto', 'night', 'sports']
        
        for mode in modes:
            # Mock successful mode change
            mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.camera_mock.capture_array.return_value = mock_image
            
            result = self.validator.test_exposure_mode(mode)
            
            self.assertTrue(result.passed)
            self.assertIn(mode, result.message)
    
    def test_white_balance_modes(self):
        """Test white balance modes."""
        modes = ['auto', 'incandescent', 'fluorescent', 'daylight']
        
        for mode in modes:
            mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.camera_mock.capture_array.return_value = mock_image
            
            result = self.validator.test_white_balance(mode)
            
            self.assertTrue(result.passed)
            self.assertIn(mode, result.message)
    
    def test_field_of_view(self):
        """Test field of view calculation."""
        # Mock calibration pattern detection
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera_mock.capture_array.return_value = mock_image
        
        # Mock corner detection
        mock_corners = np.array([[100, 100], [540, 100], [540, 380], [100, 380]])
        self.cv2_mock.findChessboardCorners.return_value = (True, mock_corners)
        
        result = self.validator.test_field_of_view()
        
        self.assertTrue(result.passed)
        self.assertIn('fov_horizontal', result.details)
        self.assertIn('fov_vertical', result.details)
    
    def test_ir_capability(self):
        """Test IR capability detection."""
        # Mock IR images
        image_no_ir = np.ones((480, 640, 3), dtype=np.uint8) * 50
        image_with_ir = np.ones((480, 640, 3), dtype=np.uint8) * 150
        
        self.camera_mock.capture_array.side_effect = [image_no_ir, image_with_ir]
        self.cv2_mock.mean.side_effect = [(50, 50, 50, 0), (150, 150, 150, 0)]
        
        # Mock GPIO for IR LED control
        with patch('RPi.GPIO.output'):
            result = self.validator.test_ir_capability()
        
        self.assertTrue(result.passed)
        self.assertIn("IR capable", result.message)
        self.assertGreater(result.details['brightness_increase'], 50)
    
    def test_generate_report(self):
        """Test report generation."""
        # Add some test results
        self.validator.test_results.append(
            ValidationResult("Test 1", True, "Passed", {"detail": "value"})
        )
        self.validator.test_results.append(
            ValidationResult("Test 2", False, "Failed", {"error": "reason"})
        )
        
        report = self.validator.generate_report()
        
        self.assertIn('summary', report)
        self.assertIn('tests', report)
        self.assertIn('timestamp', report)
        self.assertIn('system_info', report)
        
        self.assertEqual(report['summary']['total_tests'], 2)
        self.assertEqual(report['summary']['passed'], 1)
        self.assertEqual(report['summary']['failed'], 1)
        self.assertEqual(report['summary']['success_rate'], 50.0)
    
    def test_save_report(self):
        """Test saving report to file."""
        # Create test report
        test_report = {
            'summary': {'total_tests': 1, 'passed': 1},
            'tests': [],
            'timestamp': datetime.now().isoformat()
        }
        self.validator.report = test_report
        
        # Test JSON save
        with patch('builtins.open', mock_open()) as mock_file:
            self.validator.save_report('report.json', format='json')
            
            mock_file.assert_called_with('report.json', 'w')
            handle = mock_file()
            written_content = ''.join(call[0][0] for call in handle.write.call_args_list)
            
            # Verify JSON content
            written_json = json.loads(written_content)
            self.assertEqual(written_json['summary']['total_tests'], 1)
    
    def test_html_report_generation(self):
        """Test HTML report generation."""
        # Create test report
        self.validator.test_results.append(
            ValidationResult("Camera Test", True, "Success", {})
        )
        self.validator.generate_report()
        
        # Test HTML save
        with patch('builtins.open', mock_open()) as mock_file:
            self.validator.save_report('report.html', format='html')
            
            mock_file.assert_called_with('report.html', 'w')
            handle = mock_file()
            written_content = ''.join(call[0][0] for call in handle.write.call_args_list)
            
            # Verify HTML content
            self.assertIn('<html>', written_content)
            self.assertIn('Camera Validation Report', written_content)
            self.assertIn('Camera Test', written_content)
    
    def test_run_all_tests(self):
        """Test running complete validation suite."""
        # Mock all test methods
        test_methods = [
            'test_camera_detection',
            'test_basic_capture',
            'test_image_quality',
            'test_capture_speed'
        ]
        
        for method in test_methods:
            mock_result = ValidationResult(method, True, "Passed", {})
            setattr(self.validator, method, MagicMock(return_value=mock_result))
        
        # Run all tests
        self.validator.run_all_tests()
        
        # Verify all tests were called
        for method in test_methods:
            getattr(self.validator, method).assert_called_once()
        
        # Verify report was generated
        self.assertIsNotNone(self.validator.report)
        self.assertEqual(len(self.validator.test_results), len(test_methods))
    
    def test_stress_test(self):
        """Test camera stress testing."""
        # Mock continuous captures
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera_mock.capture_array.return_value = mock_image
        
        with patch('time.time') as mock_time:
            # Simulate 5 second test
            mock_time.side_effect = list(range(0, 6)) * 100
            
            result = self.validator.stress_test_camera(duration_seconds=5)
            
            self.assertTrue(result.passed)
            self.assertIn('captures_completed', result.details)
            self.assertIn('errors', result.details)
            self.assertIn('avg_fps', result.details)
    
    def test_thermal_monitoring(self):
        """Test thermal monitoring during tests."""
        # Mock temperature reading
        self.mock_subprocess.run.return_value.returncode = 0
        self.mock_subprocess.run.return_value.stdout = "temp=45.5'C"
        
        temp = self.validator.get_camera_temperature()
        
        self.assertEqual(temp, 45.5)
        self.mock_subprocess.run.assert_called_with(
            ['vcgencmd', 'measure_temp'],
            capture_output=True,
            text=True
        )
    
    def test_error_recovery(self):
        """Test error recovery during validation."""
        # Mock intermittent failures
        self.camera_mock.capture_array.side_effect = [
            Exception("Temporary error"),
            np.zeros((480, 640, 3), dtype=np.uint8)  # Success on retry
        ]
        
        result = self.validator.test_basic_capture(retry_count=2)
        
        self.assertTrue(result.passed)
        self.assertIn('retry', result.details)
        self.assertEqual(self.camera_mock.capture_array.call_count, 2)


if __name__ == '__main__':
    unittest.main()