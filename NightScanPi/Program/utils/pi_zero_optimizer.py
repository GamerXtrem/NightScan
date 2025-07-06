"""
Pi Zero 2W Memory and Performance Optimizer for NightScanPi

This module provides comprehensive optimizations for Raspberry Pi Zero 2W,
focusing on the 512MB RAM limitation and limited processing power.

Key Optimizations:
- Memory usage monitoring and management
- Dynamic resolution scaling based on available memory
- Process memory limits and cleanup
- Dependency management for minimal footprint
- Camera buffer optimization
- Audio processing optimization
"""

import os
import gc
import sys
import psutil
import logging
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryInfo:
    """Memory information for optimization decisions."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    swap_mb: float
    is_pi_zero: bool


@dataclass 
class PiZeroOptimization:
    """Pi Zero specific optimization settings."""
    max_camera_resolution: Tuple[int, int] = (1280, 720)  # 720p max for Pi Zero
    gpu_memory_mb: int = 64  # Minimal GPU memory
    max_audio_buffer_size: int = 512  # Reduced audio buffers
    max_memory_usage_percent: float = 85.0  # Trigger cleanup at 85%
    force_garbage_collection: bool = True
    enable_memory_monitoring: bool = True
    opencv_threads: int = 1  # Single thread for OpenCV
    numpy_threads: int = 1  # Single thread for NumPy


class PiZeroMemoryOptimizer:
    """Memory optimizer specifically designed for Pi Zero 2W."""
    
    def __init__(self):
        self.is_pi_zero = self._detect_pi_zero()
        self.config = PiZeroOptimization() if self.is_pi_zero else None
        self.monitoring_active = False
        self._monitor_thread = None
        
        if self.is_pi_zero:
            logger.info("Pi Zero 2W detected - enabling memory optimizations")
            self._apply_global_optimizations()
        else:
            logger.info("Not Pi Zero - standard memory management")
    
    def _detect_pi_zero(self) -> bool:
        """Detect if running on Raspberry Pi Zero."""
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().strip()
                return "Pi Zero" in model
        except:
            # Fallback to environment variable or memory detection
            if os.getenv("NIGHTSCAN_FORCE_PI_ZERO", "false").lower() == "true":
                return True
            
            # If total memory < 1GB, likely Pi Zero
            try:
                total_memory = psutil.virtual_memory().total / (1024**3)  # GB
                return total_memory < 1.0
            except:
                return False
    
    def _apply_global_optimizations(self):
        """Apply global Python and system optimizations for Pi Zero."""
        if not self.is_pi_zero:
            return
        
        try:
            # Set OpenCV thread count
            try:
                import cv2
                cv2.setNumThreads(self.config.opencv_threads)
                logger.info(f"OpenCV threads limited to {self.config.opencv_threads}")
            except ImportError:
                pass
            
            # Set NumPy thread count
            try:
                import numpy as np
                if hasattr(np, 'seterr'):
                    # Disable numpy warnings to save memory
                    np.seterr(all='ignore')
                
                # Set thread count via environment
                os.environ['OPENBLAS_NUM_THREADS'] = str(self.config.numpy_threads)
                os.environ['MKL_NUM_THREADS'] = str(self.config.numpy_threads)
                os.environ['NUMEXPR_NUM_THREADS'] = str(self.config.numpy_threads)
                logger.info(f"NumPy/BLAS threads limited to {self.config.numpy_threads}")
            except ImportError:
                pass
            
            # Aggressive garbage collection
            if self.config.force_garbage_collection:
                gc.set_threshold(100, 10, 10)  # More aggressive than default (700, 10, 10)
                logger.info("Aggressive garbage collection enabled")
            
            # Memory monitoring
            if self.config.enable_memory_monitoring:
                self._start_memory_monitoring()
        
        except Exception as e:
            logger.warning(f"Failed to apply some Pi Zero optimizations: {e}")
    
    def get_memory_info(self) -> MemoryInfo:
        """Get current memory usage information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return MemoryInfo(
                total_mb=memory.total / (1024**2),
                available_mb=memory.available / (1024**2),
                used_mb=memory.used / (1024**2),
                percent_used=memory.percent,
                swap_mb=swap.used / (1024**2),
                is_pi_zero=self.is_pi_zero
            )
        
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return MemoryInfo(512, 256, 256, 50.0, 0, self.is_pi_zero)
    
    def should_optimize_resolution(self) -> bool:
        """Check if camera resolution should be reduced due to memory pressure."""
        if not self.is_pi_zero:
            return False
        
        memory_info = self.get_memory_info()
        return memory_info.percent_used > 75.0
    
    def get_optimal_camera_resolution(self, requested_resolution: Tuple[int, int]) -> Tuple[int, int]:
        """Get optimal camera resolution for current memory conditions."""
        if not self.is_pi_zero:
            return requested_resolution
        
        max_res = self.config.max_camera_resolution
        memory_info = self.get_memory_info()
        
        # If memory pressure is high, further reduce resolution
        if memory_info.percent_used > 80.0:
            # Very high memory pressure - use 480p
            optimized = (640, 480)
        elif memory_info.percent_used > 70.0:
            # High memory pressure - use 720p
            optimized = max_res
        else:
            # Normal operation - respect Pi Zero limits
            optimized = (
                min(requested_resolution[0], max_res[0]),
                min(requested_resolution[1], max_res[1])
            )
        
        if optimized != requested_resolution:
            logger.info(f"Resolution optimized for Pi Zero: {requested_resolution} → {optimized}")
        
        return optimized
    
    def get_optimal_audio_buffer_size(self, requested_size: int) -> int:
        """Get optimal audio buffer size for Pi Zero."""
        if not self.is_pi_zero:
            return requested_size
        
        max_size = self.config.max_audio_buffer_size
        optimized = min(requested_size, max_size)
        
        if optimized != requested_size:
            logger.info(f"Audio buffer optimized for Pi Zero: {requested_size} → {optimized}")
        
        return optimized
    
    def cleanup_memory(self, force: bool = False):
        """Perform memory cleanup and garbage collection."""
        if not self.is_pi_zero and not force:
            return
        
        initial_memory = self.get_memory_info()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Additional cleanup for Pi Zero
        if self.is_pi_zero:
            # Clear import caches
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            
            # Force garbage collection again
            gc.collect()
        
        final_memory = self.get_memory_info()
        freed_mb = initial_memory.used_mb - final_memory.used_mb
        
        logger.info(f"Memory cleanup: freed {freed_mb:.1f}MB, collected {collected} objects")
    
    def _start_memory_monitoring(self):
        """Start background memory monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def _memory_monitor_loop(self):
        """Background memory monitoring loop."""
        import time
        
        while self.monitoring_active:
            try:
                memory_info = self.get_memory_info()
                
                # Trigger cleanup if memory usage is too high
                if memory_info.percent_used > self.config.max_memory_usage_percent:
                    logger.warning(f"High memory usage: {memory_info.percent_used:.1f}% - triggering cleanup")
                    self.cleanup_memory()
                
                # Log memory status every 5 minutes
                if hasattr(self, '_last_memory_log'):
                    if time.time() - self._last_memory_log > 300:  # 5 minutes
                        logger.debug(f"Memory status: {memory_info.used_mb:.0f}/{memory_info.total_mb:.0f}MB ({memory_info.percent_used:.1f}%)")
                        self._last_memory_log = time.time()
                else:
                    self._last_memory_log = time.time()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    @contextmanager
    def memory_optimized_context(self):
        """Context manager for memory-intensive operations."""
        initial_memory = self.get_memory_info()
        
        try:
            # Pre-cleanup if memory is tight
            if initial_memory.percent_used > 70.0:
                self.cleanup_memory()
            
            yield self
            
        finally:
            # Post-cleanup for Pi Zero
            if self.is_pi_zero:
                self.cleanup_memory()
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and memory info."""
        memory_info = self.get_memory_info()
        
        return {
            'is_pi_zero': self.is_pi_zero,
            'optimizations_active': self.is_pi_zero,
            'memory': {
                'total_mb': memory_info.total_mb,
                'available_mb': memory_info.available_mb,
                'used_mb': memory_info.used_mb,
                'percent_used': memory_info.percent_used,
                'swap_mb': memory_info.swap_mb
            },
            'settings': {
                'max_camera_resolution': self.config.max_camera_resolution if self.config else None,
                'gpu_memory_mb': self.config.gpu_memory_mb if self.config else None,
                'max_audio_buffer': self.config.max_audio_buffer_size if self.config else None,
                'monitoring_active': self.monitoring_active
            }
        }
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")


# Global optimizer instance
_optimizer: Optional[PiZeroMemoryOptimizer] = None


def get_optimizer() -> PiZeroMemoryOptimizer:
    """Get global Pi Zero optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PiZeroMemoryOptimizer()
    return _optimizer


def is_pi_zero() -> bool:
    """Quick check if running on Pi Zero."""
    optimizer = get_optimizer()
    return optimizer.is_pi_zero


def optimize_camera_resolution(resolution: Tuple[int, int]) -> Tuple[int, int]:
    """Optimize camera resolution for Pi Zero if needed."""
    optimizer = get_optimizer()
    return optimizer.get_optimal_camera_resolution(resolution)


def optimize_audio_buffer_size(buffer_size: int) -> int:
    """Optimize audio buffer size for Pi Zero if needed."""
    optimizer = get_optimizer()
    return optimizer.get_optimal_audio_buffer_size(buffer_size)


def cleanup_memory(force: bool = False):
    """Perform memory cleanup."""
    optimizer = get_optimizer()
    optimizer.cleanup_memory(force)


def get_memory_status() -> Dict[str, Any]:
    """Get current memory status and optimization info."""
    optimizer = get_optimizer()
    return optimizer.get_optimization_status()


@contextmanager
def memory_optimized_operation():
    """Context manager for memory-intensive operations."""
    optimizer = get_optimizer()
    with optimizer.memory_optimized_context():
        yield


if __name__ == "__main__":
    # Test Pi Zero optimizer
    print("🔧 Testing Pi Zero 2W Memory Optimizer")
    print("=" * 40)
    
    optimizer = get_optimizer()
    status = optimizer.get_optimization_status()
    
    print(f"Pi Zero Detected: {'✅' if status['is_pi_zero'] else '❌'}")
    print(f"Optimizations Active: {'✅' if status['optimizations_active'] else '❌'}")
    print()
    
    memory = status['memory']
    print(f"Memory Status:")
    print(f"  • Total: {memory['total_mb']:.0f}MB")
    print(f"  • Used: {memory['used_mb']:.0f}MB ({memory['percent_used']:.1f}%)")
    print(f"  • Available: {memory['available_mb']:.0f}MB")
    print(f"  • Swap: {memory['swap_mb']:.0f}MB")
    print()
    
    if status['optimizations_active']:
        settings = status['settings']
        print(f"Pi Zero Settings:")
        print(f"  • Max Camera Resolution: {settings['max_camera_resolution']}")
        print(f"  • GPU Memory: {settings['gpu_memory_mb']}MB")
        print(f"  • Max Audio Buffer: {settings['max_audio_buffer']}")
        print(f"  • Memory Monitoring: {'✅' if settings['monitoring_active'] else '❌'}")
        print()
        
        # Test resolution optimization
        test_resolutions = [(1920, 1080), (1280, 720), (640, 480)]
        print("Resolution Optimization Tests:")
        for res in test_resolutions:
            optimized = optimize_camera_resolution(res)
            print(f"  • {res} → {optimized}")
        print()
        
        # Test memory cleanup
        print("Testing memory cleanup...")
        cleanup_memory(force=True)
    
    print("✅ Pi Zero optimizer test completed")