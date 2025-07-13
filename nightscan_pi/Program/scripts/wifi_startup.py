#!/usr/bin/env python3
"""
NightScan WiFi Startup Service
Automatically manages WiFi connections at system startup
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Add the program directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wifi_manager import get_wifi_manager, NetworkMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/nightscan/logs/wifi_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def wait_for_network_interface(interface: str = "wlan0", timeout: int = 30):
    """Wait for network interface to be available."""
    import subprocess
    
    logger.info(f"Waiting for network interface {interface}...")
    
    for i in range(timeout):
        try:
            result = subprocess.run(
                ["ip", "link", "show", interface],
                capture_output=True, check=True
            )
            logger.info(f"Network interface {interface} is available")
            return True
        except subprocess.CalledProcessError:
            if i < timeout - 1:
                time.sleep(1)
            else:
                logger.error(f"Network interface {interface} not available after {timeout}s")
                return False
    
    return False

def startup_wifi_management():
    """Main startup routine for WiFi management."""
    logger.info("🚀 Starting NightScan WiFi management...")
    
    # Wait for network interface
    if not wait_for_network_interface():
        logger.error("❌ Network interface not available, exiting")
        return False
    
    # Initialize WiFi manager
    try:
        wifi_manager = get_wifi_manager()
        logger.info("✅ WiFi manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize WiFi manager: {e}")
        return False
    
    # Check current status
    try:
        status = wifi_manager.get_status()
        logger.info(f"Current WiFi status: {status['mode']}")
        
        # If already connected, monitor the connection
        if status['mode'] == NetworkMode.CLIENT.value:
            logger.info(f"Already connected to: {status.get('connected_network', 'unknown')}")
            return True
        
        # Check if any networks are configured
        networks = wifi_manager.get_networks()
        
        if not networks:
            # First installation - no networks configured
            logger.info("🆕 First installation detected - no networks configured")
            logger.info("Starting hotspot mode for initial configuration...")
            
            hotspot_success = wifi_manager.start_hotspot()
            if hotspot_success:
                logger.info("✅ Hotspot mode started successfully")
                logger.info("📱 Connect to 'NightScan-Setup' WiFi")
                logger.info("🌐 Then go to http://192.168.4.1:5000 to configure")
                return True
            else:
                logger.error("❌ Failed to start hotspot mode")
                return False
        
        # Normal operation - try to connect to configured networks
        logger.info("Attempting to connect to best available network...")
        success = wifi_manager.connect_to_best_network()
        
        if success:
            status = wifi_manager.get_status()
            logger.info(f"✅ Connected to: {status.get('connected_network', 'unknown')}")
            return True
        else:
            logger.warning("⚠️ No suitable networks found")
            logger.info("Starting hotspot mode for configuration...")
            
            # Start hotspot if no networks available
            hotspot_success = wifi_manager.start_hotspot()
            if hotspot_success:
                logger.info("✅ Hotspot mode started for reconfiguration")
                return True
            else:
                logger.error("❌ Failed to start hotspot mode")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        return False

def daemon_mode():
    """Run in daemon mode with periodic checks."""
    logger.info("🔄 Starting daemon mode...")
    
    wifi_manager = get_wifi_manager()
    
    while True:
        try:
            status = wifi_manager.get_status()
            
            # Check every 60 seconds
            if status['mode'] == NetworkMode.DISABLED.value:
                logger.info("WiFi disabled, attempting to connect...")
                success = wifi_manager.connect_to_best_network()
                
                if not success:
                    logger.info("No networks available, checking if hotspot should be started...")
                    networks = wifi_manager.get_networks()
                    
                    if not networks:
                        logger.info("Starting hotspot mode...")
                        wifi_manager.start_hotspot()
            
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("Daemon mode stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in daemon mode: {e}")
            time.sleep(30)  # Wait before retrying

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NightScan WiFi Startup Service")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--oneshot", action="store_true", help="Run once and exit")
    parser.add_argument("--interface", default="wlan0", help="Network interface to use")
    parser.add_argument("--timeout", type=int, default=30, help="Interface wait timeout")
    
    args = parser.parse_args()
    
    # Create log directory
    Path("/opt/nightscan/logs").mkdir(parents=True, exist_ok=True)
    
    if args.daemon:
        # Run startup first
        startup_success = startup_wifi_management()
        
        if startup_success:
            # Then run daemon
            daemon_mode()
        else:
            logger.error("Startup failed, not entering daemon mode")
            sys.exit(1)
    
    elif args.oneshot:
        # Run once and exit
        success = startup_wifi_management()
        sys.exit(0 if success else 1)
    
    else:
        # Default: run startup and exit
        success = startup_wifi_management()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()