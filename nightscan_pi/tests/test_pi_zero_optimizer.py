#!/usr/bin/env python3
"""
Tests for Pi Zero 2W Memory Optimizer module.

Tests memory optimization functionality for constrained devices.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
import gc
import threading
from datetime import datetime
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock RPi.GPIO before import
sys.modules['RPi'] = MagicMock()
sys.modules['RPi.GPIO'] = MagicMock()

from pi_zero_optimizer import PiZeroOptimizer


class TestPiZeroOptimizer(unittest.TestCase):
    """Test cases for Pi Zero 2W optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.patcher_psutil = patch('pi_zero_optimizer.psutil')
        self.mock_psutil = self.patcher_psutil.start()
        
        # Mock memory info
        self.mock_memory = MagicMock()
        self.mock_memory.total = 512 * 1024 * 1024  # 512MB
        self.mock_memory.available = 256 * 1024 * 1024  # 256MB available
        self.mock_memory.percent = 50.0
        self.mock_psutil.virtual_memory.return_value = self.mock_memory
        
        # Mock CPU info
        self.mock_psutil.cpu_count.return_value = 4
        self.mock_psutil.cpu_percent.return_value = 25.0
        
        # Mock disk usage
        self.mock_disk = MagicMock()
        self.mock_disk.free = 1024 * 1024 * 1024  # 1GB free
        self.mock_psutil.disk_usage.return_value = self.mock_disk
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher_psutil.stop()
    
    @patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0'))
    def test_is_pi_zero_detection(self):
        """Test Pi Zero 2W detection."""
        optimizer = PiZeroOptimizer()
        self.assertTrue(optimizer.is_pi_zero)
    
    @patch('builtins.open', mock_open(read_data='Raspberry Pi 4 Model B Rev 1.2'))
    def test_non_pi_zero_detection(self):
        """Test non-Pi Zero detection."""
        optimizer = PiZeroOptimizer()
        self.assertFalse(optimizer.is_pi_zero)
    
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_detection_file_not_found(self):
        """Test detection when model file doesn't exist."""
        optimizer = PiZeroOptimizer()
        self.assertFalse(optimizer.is_pi_zero)
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            usage = optimizer.get_memory_usage()
            
            self.assertEqual(usage['total_mb'], 512.0)
            self.assertEqual(usage['available_mb'], 256.0)
            self.assertEqual(usage['percent'], 50.0)
            self.assertEqual(usage['status'], 'normal')
    
    def test_memory_status_levels(self):
        """Test different memory status levels."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            
            # Test normal status
            self.mock_memory.percent = 60.0
            usage = optimizer.get_memory_usage()
            self.assertEqual(usage['status'], 'normal')
            
            # Test warning status
            self.mock_memory.percent = 75.0
            usage = optimizer.get_memory_usage()
            self.assertEqual(usage['status'], 'warning')
            
            # Test critical status
            self.mock_memory.percent = 88.0
            usage = optimizer.get_memory_usage()
            self.assertEqual(usage['status'], 'critical')
    
    def test_optimize_for_capture(self):
        """Test optimization for image capture."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            
            # Test normal memory - should return original resolution
            self.mock_memory.percent = 60.0
            resolution = optimizer.optimize_for_capture((1920, 1080))
            self.assertEqual(resolution, (1920, 1080))
            
            # Test warning memory - should reduce resolution
            self.mock_memory.percent = 75.0
            resolution = optimizer.optimize_for_capture((1920, 1080))
            self.assertEqual(resolution, (1536, 864))  # 80% of original
            
            # Test critical memory - should reduce further
            self.mock_memory.percent = 88.0
            resolution = optimizer.optimize_for_capture((1920, 1080))
            self.assertEqual(resolution, (1152, 648))  # 60% of original
    
    @patch('gc.collect')
    def test_cleanup_memory(self, mock_gc):
        """Test memory cleanup functionality."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            
            # Initial memory state
            initial_percent = 80.0
            self.mock_memory.percent = initial_percent
            
            # After cleanup, memory should improve
            def update_memory(*args):
                self.mock_memory.percent = 65.0
            
            mock_gc.side_effect = update_memory
            
            freed = optimizer.cleanup_memory()
            
            mock_gc.assert_called_once()
            self.assertGreater(freed, 0)
    
    @patch('threading.active_count')
    @patch('os.environ')
    def test_optimize_threading(self, mock_environ, mock_thread_count):
        """Test thread optimization."""
        mock_thread_count.return_value = 10
        
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            optimizer.optimize_threading()
            
            # Check that thread limits are set
            self.assertIn('OMP_NUM_THREADS', mock_environ.__setitem__.call_args_list[0][0])
            self.assertIn('OPENBLAS_NUM_THREADS', mock_environ.__setitem__.call_args_list[1][0])
            self.assertIn('MKL_NUM_THREADS', mock_environ.__setitem__.call_args_list[2][0])
    
    def test_check_disk_space(self):
        """Test disk space checking."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            
            # Sufficient disk space
            self.mock_disk.free = 2 * 1024 * 1024 * 1024  # 2GB
            has_space, free_mb = optimizer.check_disk_space(required_mb=500)
            self.assertTrue(has_space)
            self.assertEqual(free_mb, 2048.0)
            
            # Insufficient disk space
            self.mock_disk.free = 100 * 1024 * 1024  # 100MB
            has_space, free_mb = optimizer.check_disk_space(required_mb=500)
            self.assertFalse(has_space)
            self.assertEqual(free_mb, 100.0)
    
    def test_get_system_stats(self):
        """Test system statistics gathering."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            stats = optimizer.get_system_stats()
            
            self.assertIn('memory', stats)
            self.assertIn('cpu_percent', stats)
            self.assertIn('cpu_count', stats)
            self.assertIn('disk_free_mb', stats)
            self.assertIn('thread_count', stats)
            self.assertIn('timestamp', stats)
            self.assertIn('is_pi_zero', stats)
            
            self.assertEqual(stats['cpu_percent'], 25.0)
            self.assertEqual(stats['cpu_count'], 4)
            self.assertTrue(stats['is_pi_zero'])
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            # Mock gc.collect to track calls
            with patch('gc.collect') as mock_gc:
                with PiZeroOptimizer() as optimizer:
                    self.assertIsNotNone(optimizer)
                    self.assertTrue(optimizer.is_pi_zero)
                
                # Verify cleanup was called
                mock_gc.assert_called()
    
    def test_memory_pressure_callback(self):
        """Test memory pressure callback execution."""
        callback_executed = False
        
        def test_callback():
            nonlocal callback_executed
            callback_executed = True
        
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            
            # Set critical memory
            self.mock_memory.percent = 90.0
            
            # Optimize with callback
            resolution = optimizer.optimize_for_capture(
                (1920, 1080),
                memory_pressure_callback=test_callback
            )
            
            self.assertTrue(callback_executed)
            self.assertEqual(resolution, (1152, 648))  # Critical reduction
    
    def test_optimize_for_model_inference(self):
        """Test optimization for model inference."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            
            # Normal memory
            self.mock_memory.percent = 60.0
            batch_size = optimizer.optimize_for_inference(batch_size=32)
            self.assertEqual(batch_size, 32)
            
            # Warning memory
            self.mock_memory.percent = 75.0
            batch_size = optimizer.optimize_for_inference(batch_size=32)
            self.assertEqual(batch_size, 16)
            
            # Critical memory
            self.mock_memory.percent = 88.0
            batch_size = optimizer.optimize_for_inference(batch_size=32)
            self.assertEqual(batch_size, 8)
    
    def test_adaptive_quality_settings(self):
        """Test adaptive quality settings based on memory."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            
            # Normal memory - high quality
            self.mock_memory.percent = 50.0
            settings = optimizer.get_adaptive_settings()
            self.assertEqual(settings['jpeg_quality'], 85)
            self.assertEqual(settings['video_bitrate'], '2M')
            self.assertTrue(settings['enable_preview'])
            
            # Warning memory - reduced quality
            self.mock_memory.percent = 75.0
            settings = optimizer.get_adaptive_settings()
            self.assertEqual(settings['jpeg_quality'], 70)
            self.assertEqual(settings['video_bitrate'], '1.5M')
            self.assertTrue(settings['enable_preview'])
            
            # Critical memory - minimum quality
            self.mock_memory.percent = 88.0
            settings = optimizer.get_adaptive_settings()
            self.assertEqual(settings['jpeg_quality'], 60)
            self.assertEqual(settings['video_bitrate'], '1M')
            self.assertFalse(settings['enable_preview'])
    
    @patch('time.sleep')
    def test_wait_for_memory(self, mock_sleep):
        """Test waiting for memory availability."""
        with patch('builtins.open', mock_open(read_data='Raspberry Pi Zero 2 W Rev 1.0')):
            optimizer = PiZeroOptimizer()
            
            # Simulate memory becoming available
            memory_levels = [85.0, 80.0, 75.0, 68.0]  # Gradually improving
            self.mock_memory.percent = memory_levels[0]
            
            call_count = 0
            def update_memory(*args):
                nonlocal call_count
                call_count += 1
                if call_count < len(memory_levels):
                    self.mock_memory.percent = memory_levels[call_count]
            
            self.mock_psutil.virtual_memory.side_effect = update_memory
            
            # Should wait until memory is below threshold
            result = optimizer.wait_for_memory(threshold=70, timeout=5)
            self.assertTrue(result)
            self.assertEqual(mock_sleep.call_count, 3)  # Called 3 times before success


if __name__ == '__main__':
    unittest.main()