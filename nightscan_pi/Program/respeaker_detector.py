"""ReSpeaker Lite detection and configuration for NightScanPi."""
from __future__ import annotations

import subprocess
import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReSpeakerInfo:
    """Information about detected ReSpeaker Lite device."""
    device_id: int
    device_name: str
    channels: int
    sample_rates: List[int]
    usb_device_id: str
    firmware_version: Optional[str] = None
    is_usb_mode: bool = True


class ReSpeakerDetector:
    """Detector for ReSpeaker Lite USB audio devices."""
    
    # ReSpeaker Lite USB identifiers
    VENDOR_ID = "2886"
    PRODUCT_ID_USB = "0019"  # USB mode
    PRODUCT_ID_I2S = "001a"  # I2S mode (less common)
    
    DEVICE_NAMES = [
        "ReSpeaker Lite",
        "ReSpeaker 2-Mics",
        "SEEED ReSpeaker Lite"
    ]
    
    def __init__(self):
        self.detected_devices: List[ReSpeakerInfo] = []
    
    def detect_usb_device(self) -> bool:
        """Detect ReSpeaker Lite via USB using lsusb."""
        try:
            result = subprocess.run(
                ["lsusb"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Look for ReSpeaker USB device
            usb_pattern = rf"{self.VENDOR_ID}:{self.PRODUCT_ID_USB}"
            usb_i2s_pattern = rf"{self.VENDOR_ID}:{self.PRODUCT_ID_I2S}"
            
            for line in result.stdout.split('\n'):
                if re.search(usb_pattern, line) or re.search(usb_i2s_pattern, line):
                    logger.info(f"ReSpeaker Lite USB device detected: {line.strip()}")
                    return True
            
            return False
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to run lsusb: {e}")
            return False
    
    def detect_audio_device(self) -> Optional[ReSpeakerInfo]:
        """Detect ReSpeaker Lite as audio device via ALSA."""
        try:
            import pyaudio
            
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            
            for i in range(device_count):
                try:
                    device_info = p.get_device_info_by_index(i)
                    device_name = device_info.get('name', '').lower()
                    
                    # Check if this is a ReSpeaker device
                    for known_name in self.DEVICE_NAMES:
                        if known_name.lower() in device_name:
                            logger.info(f"ReSpeaker audio device found: {device_info['name']}")
                            
                            # Get supported sample rates
                            supported_rates = self._get_supported_sample_rates(i, p)
                            
                            respeaker_info = ReSpeakerInfo(
                                device_id=i,
                                device_name=device_info['name'],
                                channels=min(device_info.get('maxInputChannels', 2), 2),
                                sample_rates=supported_rates,
                                usb_device_id=f"{self.VENDOR_ID}:{self.PRODUCT_ID_USB}"
                            )
                            
                            p.terminate()
                            return respeaker_info
                            
                except Exception as e:
                    logger.debug(f"Error checking device {i}: {e}")
                    continue
            
            p.terminate()
            return None
            
        except ImportError:
            logger.error("PyAudio not available for device detection")
            return None
        except Exception as e:
            logger.error(f"Error detecting audio device: {e}")
            return None
    
    def _get_supported_sample_rates(self, device_id: int, p) -> List[int]:
        """Get supported sample rates for ReSpeaker Lite."""
        # ReSpeaker Lite officially supports these rates
        test_rates = [8000, 16000, 22050, 44100, 48000]
        supported = []
        
        for rate in test_rates:
            try:
                # ReSpeaker Lite max is 16kHz, but we test others for compatibility
                if rate <= 16000:  # Respect hardware limitation
                    if p.is_format_supported(
                        rate=rate,
                        input_device=device_id,
                        input_channels=2,
                        input_format=pyaudio.paInt16
                    ):
                        supported.append(rate)
            except Exception:
                continue
        
        # If nothing detected, assume 16kHz (ReSpeaker Lite max)
        if not supported:
            supported = [16000]
        
        return sorted(supported)
    
    def get_firmware_version(self) -> Optional[str]:
        """Get ReSpeaker Lite firmware version via dfu-util."""
        try:
            result = subprocess.run(
                ["dfu-util", "-l"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Look for ReSpeaker device info
            for line in result.stdout.split('\n'):
                if "2886:0019" in line and "ver=" in line:
                    version_match = re.search(r'ver=(\d+)', line)
                    if version_match:
                        version = version_match.group(1)
                        logger.info(f"ReSpeaker Lite firmware version: {version}")
                        return version
            
            return None
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("dfu-util not available or no ReSpeaker in DFU mode")
            return None
    
    def detect_all(self) -> bool:
        """Run complete ReSpeaker Lite detection."""
        logger.info("🎤 Detecting ReSpeaker Lite...")
        
        # Check USB detection
        usb_detected = self.detect_usb_device()
        
        # Check audio device detection
        audio_device = self.detect_audio_device()
        
        if audio_device:
            # Get firmware version
            audio_device.firmware_version = self.get_firmware_version()
            self.detected_devices.append(audio_device)
            
            logger.info(f"✅ ReSpeaker Lite detected successfully:")
            logger.info(f"   Device ID: {audio_device.device_id}")
            logger.info(f"   Name: {audio_device.device_name}")
            logger.info(f"   Channels: {audio_device.channels}")
            logger.info(f"   Supported rates: {audio_device.sample_rates}")
            if audio_device.firmware_version:
                logger.info(f"   Firmware: {audio_device.firmware_version}")
            
            return True
        elif usb_detected:
            logger.warning("⚠️ ReSpeaker Lite USB device detected but not available as audio device")
            logger.warning("   Check USB audio drivers and firmware version")
            return False
        else:
            logger.error("❌ ReSpeaker Lite not detected")
            return False
    
    def get_optimal_config(self) -> Optional[Dict]:
        """Get optimal audio configuration for detected ReSpeaker Lite."""
        if not self.detected_devices:
            return None
        
        device = self.detected_devices[0]
        
        # Find best sample rate (prefer 16kHz for ReSpeaker Lite)
        preferred_rates = [16000, 8000, 22050, 44100]
        best_rate = None
        
        for rate in preferred_rates:
            if rate in device.sample_rates:
                best_rate = rate
                break
        
        if not best_rate:
            best_rate = max(device.sample_rates) if device.sample_rates else 16000
        
        return {
            "device_id": device.device_id,
            "device_name": device.device_name,
            "sample_rate": best_rate,
            "channels": device.channels,
            "format": "paInt16",
            "chunk_size": 1024,
            "is_respeaker": True,
            "usb_device_id": device.usb_device_id
        }


def detect_respeaker() -> Optional[ReSpeakerInfo]:
    """Convenience function to detect ReSpeaker Lite."""
    detector = ReSpeakerDetector()
    if detector.detect_all() and detector.detected_devices:
        return detector.detected_devices[0]
    return None


def get_respeaker_config() -> Optional[Dict]:
    """Get optimal configuration for ReSpeaker Lite."""
    detector = ReSpeakerDetector()
    detector.detect_all()
    return detector.get_optimal_config()


def is_respeaker_available() -> bool:
    """Quick check if ReSpeaker Lite is available."""
    detector = ReSpeakerDetector()
    return detector.detect_all()


if __name__ == "__main__":
    # Test ReSpeaker detection
    logging.basicConfig(level=logging.INFO)
    
    detector = ReSpeakerDetector()
    if detector.detect_all():
        config = detector.get_optimal_config()
        if config:
            print(f"Optimal ReSpeaker config: {config}")
    else:
        print("ReSpeaker Lite not detected")