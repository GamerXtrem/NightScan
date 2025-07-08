#!/usr/bin/env python3
"""
NightScan WiFi Manager
Advanced WiFi management system supporting multiple networks, hotspot mode,
and automatic fallback between networks.
"""

import json
import logging
import subprocess
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import tempfile
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkMode(Enum):
    """WiFi network modes."""
    CLIENT = "client"
    HOTSPOT = "hotspot"
    DISABLED = "disabled"

class NetworkSecurity(Enum):
    """WiFi security types."""
    OPEN = "open"
    WPA_PSK = "wpa_psk"
    WPA2_PSK = "wpa2_psk"
    WPA_WPA2_PSK = "wpa_wpa2_psk"

@dataclass
class WiFiNetwork:
    """WiFi network configuration."""
    ssid: str
    password: str = ""
    security: NetworkSecurity = NetworkSecurity.WPA2_PSK
    priority: int = 0  # Higher number = higher priority
    auto_connect: bool = True
    hidden: bool = False
    country: str = "FR"
    frequency: Optional[int] = None  # 2.4GHz or 5GHz preference
    last_connected: Optional[float] = None
    connection_attempts: int = 0
    max_attempts: int = 3
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WiFiNetwork':
        """Create from dictionary."""
        # Handle enum conversion
        if 'security' in data and isinstance(data['security'], str):
            data['security'] = NetworkSecurity(data['security'])
        return cls(**data)

@dataclass
class HotspotConfig:
    """Hotspot configuration."""
    ssid: str = "NightScan-Setup"
    password: str = "nightscan2024"
    channel: int = 6
    hidden: bool = False
    max_clients: int = 10
    ip_range: str = "192.168.4.0/24"
    gateway: str = "192.168.4.1"
    dhcp_start: str = "192.168.4.100"
    dhcp_end: str = "192.168.4.200"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

@dataclass
class WiFiStatus:
    """Current WiFi status."""
    mode: NetworkMode
    connected_network: Optional[str] = None
    ip_address: Optional[str] = None
    signal_strength: Optional[int] = None
    connection_time: Optional[float] = None
    interface: str = "wlan0"
    last_scan: Optional[float] = None
    available_networks: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['mode'] = self.mode.value
        return result

class WiFiManager:
    """Advanced WiFi management system."""
    
    def __init__(self, config_dir: Path = Path("/opt/nightscan/config")):
        """Initialize WiFi manager."""
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration files
        self.networks_file = self.config_dir / "wifi_networks.json"
        self.hotspot_config_file = self.config_dir / "hotspot_config.json"
        self.status_file = self.config_dir / "wifi_status.json"
        
        # System paths
        self.wpa_supplicant_conf = Path("/etc/wpa_supplicant/wpa_supplicant.conf")
        self.hostapd_conf = Path("/etc/hostapd/hostapd.conf")
        self.dnsmasq_conf = Path("/etc/dnsmasq.conf")
        
        # Interface
        self.interface = "wlan0"
        
        # State
        self.networks: Dict[str, WiFiNetwork] = {}
        self.hotspot_config = HotspotConfig()
        self.current_status = WiFiStatus(mode=NetworkMode.DISABLED)
        self.connection_lock = threading.Lock()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Load configuration
        self._load_configuration()
        
        # Start monitoring
        self.start_monitoring()
    
    def _load_configuration(self):
        """Load configuration from files."""
        try:
            # Load network configurations
            if self.networks_file.exists():
                with open(self.networks_file) as f:
                    data = json.load(f)
                    self.networks = {
                        ssid: WiFiNetwork.from_dict(net_data)
                        for ssid, net_data in data.items()
                    }
            
            # Load hotspot configuration
            if self.hotspot_config_file.exists():
                with open(self.hotspot_config_file) as f:
                    data = json.load(f)
                    self.hotspot_config = HotspotConfig(**data)
            
            # Load status
            if self.status_file.exists():
                with open(self.status_file) as f:
                    data = json.load(f)
                    self.current_status = WiFiStatus(**data)
                    if 'mode' in data:
                        self.current_status.mode = NetworkMode(data['mode'])
            
            logger.info(f"Loaded {len(self.networks)} WiFi networks")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _save_configuration(self):
        """Save configuration to files."""
        try:
            # Save networks
            networks_data = {
                ssid: network.to_dict() 
                for ssid, network in self.networks.items()
            }
            with open(self.networks_file, 'w') as f:
                json.dump(networks_data, f, indent=2)
            
            # Save hotspot config
            with open(self.hotspot_config_file, 'w') as f:
                json.dump(self.hotspot_config.to_dict(), f, indent=2)
            
            # Save status
            with open(self.status_file, 'w') as f:
                json.dump(self.current_status.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def add_network(self, ssid: str, password: str = "", 
                   security: NetworkSecurity = NetworkSecurity.WPA2_PSK,
                   priority: int = 0, auto_connect: bool = True,
                   hidden: bool = False, notes: str = "") -> bool:
        """Add a new WiFi network."""
        try:
            network = WiFiNetwork(
                ssid=ssid,
                password=password,
                security=security,
                priority=priority,
                auto_connect=auto_connect,
                hidden=hidden,
                notes=notes
            )
            
            self.networks[ssid] = network
            self._save_configuration()
            
            logger.info(f"Added network: {ssid} (priority: {priority})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding network {ssid}: {e}")
            return False
    
    def remove_network(self, ssid: str) -> bool:
        """Remove a WiFi network."""
        try:
            if ssid in self.networks:
                del self.networks[ssid]
                self._save_configuration()
                logger.info(f"Removed network: {ssid}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing network {ssid}: {e}")
            return False
    
    def update_network(self, ssid: str, **kwargs) -> bool:
        """Update network configuration."""
        try:
            if ssid not in self.networks:
                return False
            
            network = self.networks[ssid]
            for key, value in kwargs.items():
                if hasattr(network, key):
                    setattr(network, key, value)
            
            self._save_configuration()
            logger.info(f"Updated network: {ssid}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating network {ssid}: {e}")
            return False
    
    def get_networks(self) -> List[Dict[str, Any]]:
        """Get all configured networks."""
        return [
            {
                'ssid': ssid,
                'priority': network.priority,
                'auto_connect': network.auto_connect,
                'hidden': network.hidden,
                'last_connected': network.last_connected,
                'connection_attempts': network.connection_attempts,
                'notes': network.notes
            }
            for ssid, network in sorted(
                self.networks.items(),
                key=lambda x: x[1].priority,
                reverse=True
            )
        ]
    
    def scan_networks(self, timeout: int = 10) -> List[Dict[str, Any]]:
        """Scan for available WiFi networks."""
        try:
            # Force interface up
            subprocess.run(["sudo", "ip", "link", "set", self.interface, "up"], 
                         check=False)
            
            # Scan for networks
            result = subprocess.run(
                ["sudo", "iwlist", self.interface, "scan"],
                capture_output=True, text=True, timeout=timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Network scan failed: {result.stderr}")
                return []
            
            # Parse scan results
            networks = []
            current_network = {}
            
            for line in result.stdout.split('\n'):
                line = line.strip()
                
                if line.startswith('Cell'):
                    if current_network:
                        networks.append(current_network)
                    current_network = {}
                    
                elif 'ESSID:' in line:
                    essid = line.split('ESSID:')[1].strip('"')
                    if essid:
                        current_network['ssid'] = essid
                        
                elif 'Signal level' in line:
                    # Extract signal strength
                    signal_match = re.search(r'Signal level[=:](-?\d+)', line)
                    if signal_match:
                        current_network['signal_strength'] = int(signal_match.group(1))
                        
                elif 'Encryption key:' in line:
                    encrypted = 'on' in line
                    current_network['encrypted'] = encrypted
                    
                elif 'IE: WPA' in line or 'WPA2' in line:
                    current_network['security'] = 'WPA/WPA2'
            
            if current_network:
                networks.append(current_network)
            
            # Filter out empty SSIDs and sort by signal strength
            networks = [n for n in networks if n.get('ssid')]
            networks.sort(key=lambda x: x.get('signal_strength', -100), reverse=True)
            
            self.current_status.last_scan = time.time()
            self.current_status.available_networks = [n['ssid'] for n in networks]
            
            logger.info(f"Found {len(networks)} networks")
            return networks
            
        except Exception as e:
            logger.error(f"Error scanning networks: {e}")
            return []
    
    def connect_to_network(self, ssid: str, force: bool = False) -> bool:
        """Connect to a specific network."""
        with self.connection_lock:
            try:
                if ssid not in self.networks:
                    logger.error(f"Network {ssid} not configured")
                    return False
                
                network = self.networks[ssid]
                
                # Check connection attempts
                if not force and network.connection_attempts >= network.max_attempts:
                    logger.warning(f"Max connection attempts reached for {ssid}")
                    return False
                
                # Disconnect current connection
                if self.current_status.mode == NetworkMode.CLIENT:
                    self._disconnect_client()
                elif self.current_status.mode == NetworkMode.HOTSPOT:
                    self._stop_hotspot()
                
                # Generate wpa_supplicant configuration
                if not self._generate_wpa_supplicant_config(network):
                    return False
                
                # Connect to network
                success = self._connect_client_mode(network)
                
                # Update network stats
                network.connection_attempts += 1
                if success:
                    network.last_connected = time.time()
                    network.connection_attempts = 0
                    
                    # Update status
                    self.current_status.mode = NetworkMode.CLIENT
                    self.current_status.connected_network = ssid
                    self.current_status.connection_time = time.time()
                    
                    logger.info(f"Connected to network: {ssid}")
                else:
                    logger.error(f"Failed to connect to network: {ssid}")
                
                self._save_configuration()
                return success
                
            except Exception as e:
                logger.error(f"Error connecting to network {ssid}: {e}")
                return False
    
    def connect_to_best_network(self) -> bool:
        """Connect to the best available network based on priority and signal strength."""
        try:
            # Get available networks
            available_networks = self.scan_networks()
            if not available_networks:
                logger.warning("No networks found during scan")
                return False
            
            # Find best network to connect to
            best_network = None
            best_score = -1
            
            for network_info in available_networks:
                ssid = network_info['ssid']
                
                if ssid not in self.networks:
                    continue
                
                network = self.networks[ssid]
                
                # Skip if auto_connect is disabled
                if not network.auto_connect:
                    continue
                
                # Skip if max attempts reached
                if network.connection_attempts >= network.max_attempts:
                    continue
                
                # Calculate score (priority + signal strength bonus)
                signal_strength = network_info.get('signal_strength', -100)
                score = network.priority * 100 + max(0, signal_strength + 100)
                
                if score > best_score:
                    best_score = score
                    best_network = ssid
            
            if best_network:
                logger.info(f"Connecting to best network: {best_network} (score: {best_score})")
                return self.connect_to_network(best_network)
            else:
                logger.warning("No suitable network found")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to best network: {e}")
            return False
    
    def start_hotspot(self) -> bool:
        """Start WiFi hotspot mode."""
        with self.connection_lock:
            try:
                # Disconnect client mode first
                if self.current_status.mode == NetworkMode.CLIENT:
                    self._disconnect_client()
                
                # Configure hotspot
                if not self._configure_hotspot():
                    return False
                
                # Start hotspot services
                if not self._start_hotspot_services():
                    return False
                
                # Update status
                self.current_status.mode = NetworkMode.HOTSPOT
                self.current_status.connected_network = self.hotspot_config.ssid
                self.current_status.ip_address = self.hotspot_config.gateway
                self.current_status.connection_time = time.time()
                
                self._save_configuration()
                logger.info("Hotspot started successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error starting hotspot: {e}")
                return False
    
    def stop_hotspot(self) -> bool:
        """Stop WiFi hotspot mode."""
        with self.connection_lock:
            try:
                if self.current_status.mode != NetworkMode.HOTSPOT:
                    return True
                
                success = self._stop_hotspot()
                
                if success:
                    self.current_status.mode = NetworkMode.DISABLED
                    self.current_status.connected_network = None
                    self.current_status.ip_address = None
                    self.current_status.connection_time = None
                    
                    self._save_configuration()
                    logger.info("Hotspot stopped successfully")
                
                return success
                
            except Exception as e:
                logger.error(f"Error stopping hotspot: {e}")
                return False
    
    def disconnect(self) -> bool:
        """Disconnect from current network."""
        with self.connection_lock:
            try:
                if self.current_status.mode == NetworkMode.CLIENT:
                    success = self._disconnect_client()
                elif self.current_status.mode == NetworkMode.HOTSPOT:
                    success = self._stop_hotspot()
                else:
                    return True
                
                if success:
                    self.current_status.mode = NetworkMode.DISABLED
                    self.current_status.connected_network = None
                    self.current_status.ip_address = None
                    self.current_status.connection_time = None
                    
                    self._save_configuration()
                    logger.info("Disconnected from network")
                
                return success
                
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
                return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current WiFi status."""
        # Update IP address if connected
        if self.current_status.mode == NetworkMode.CLIENT:
            self.current_status.ip_address = self._get_ip_address()
        
        return self.current_status.to_dict()
    
    def start_monitoring(self):
        """Start connection monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("WiFi monitoring started")
    
    def stop_monitoring(self):
        """Stop connection monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("WiFi monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check connection status every 30 seconds
                if self.current_status.mode == NetworkMode.CLIENT:
                    if not self._is_connected():
                        logger.warning("Connection lost, attempting to reconnect...")
                        
                        # Try to reconnect to the same network first
                        if self.current_status.connected_network:
                            success = self.connect_to_network(
                                self.current_status.connected_network, force=True
                            )
                            if success:
                                time.sleep(30)
                                continue
                        
                        # Try to connect to best available network
                        success = self.connect_to_best_network()
                        if not success:
                            # Fall back to hotspot mode
                            logger.info("No networks available, starting hotspot...")
                            self.start_hotspot()
                
                elif self.current_status.mode == NetworkMode.DISABLED:
                    # Try to connect to a network
                    if self.networks:
                        logger.info("Attempting to connect to available network...")
                        success = self.connect_to_best_network()
                        if not success:
                            logger.info("No networks available, starting hotspot...")
                            self.start_hotspot()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def _generate_wpa_supplicant_config(self, network: WiFiNetwork) -> bool:
        """Generate wpa_supplicant configuration for a network."""
        try:
            config_lines = [
                f"country={network.country}",
                "ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev",
                "update_config=1",
                ""
            ]
            
            # Network block
            config_lines.append("network={")
            config_lines.append(f'    ssid="{network.ssid}"')
            
            if network.security == NetworkSecurity.OPEN:
                config_lines.append("    key_mgmt=NONE")
            else:
                config_lines.append(f'    psk="{network.password}"')
            
            if network.hidden:
                config_lines.append("    scan_ssid=1")
            
            config_lines.append(f"    priority={network.priority}")
            config_lines.append("}")
            
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write('\n'.join(config_lines))
                tmp_path = tmp.name
            
            # Copy to system location
            subprocess.run(
                ["sudo", "cp", tmp_path, str(self.wpa_supplicant_conf)],
                check=True
            )
            
            # Set permissions
            subprocess.run(
                ["sudo", "chmod", "600", str(self.wpa_supplicant_conf)],
                check=True
            )
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating wpa_supplicant config: {e}")
            return False
    
    def _connect_client_mode(self, network: WiFiNetwork) -> bool:
        """Connect in client mode."""
        try:
            # Bring interface up
            subprocess.run(["sudo", "ip", "link", "set", self.interface, "up"], 
                         check=True)
            
            # Kill any existing wpa_supplicant processes
            subprocess.run(["sudo", "pkill", "-f", "wpa_supplicant"], 
                         check=False)
            
            # Start wpa_supplicant
            subprocess.run([
                "sudo", "wpa_supplicant", "-B", "-i", self.interface,
                "-c", str(self.wpa_supplicant_conf)
            ], check=True)
            
            # Wait for connection
            time.sleep(5)
            
            # Request DHCP lease
            subprocess.run(["sudo", "dhclient", self.interface], 
                         check=False)
            
            # Wait for IP address
            for _ in range(10):
                if self._get_ip_address():
                    return True
                time.sleep(1)
            
            return False
            
        except Exception as e:
            logger.error(f"Error connecting in client mode: {e}")
            return False
    
    def _disconnect_client(self) -> bool:
        """Disconnect from client mode."""
        try:
            # Release DHCP lease
            subprocess.run(["sudo", "dhclient", "-r", self.interface], 
                         check=False)
            
            # Kill wpa_supplicant
            subprocess.run(["sudo", "pkill", "-f", "wpa_supplicant"], 
                         check=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting client: {e}")
            return False
    
    def _configure_hotspot(self) -> bool:
        """Configure hotspot mode."""
        try:
            # Configure hostapd
            hostapd_config = [
                f"interface={self.interface}",
                f"ssid={self.hotspot_config.ssid}",
                f"channel={self.hotspot_config.channel}",
                "hw_mode=g",
                "ieee80211n=1",
                "wmm_enabled=1",
                "macaddr_acl=0",
                "auth_algs=1",
                "ignore_broadcast_ssid=0",
                "wpa=2",
                "wpa_key_mgmt=WPA-PSK",
                "wpa_pairwise=CCMP",
                f"wpa_passphrase={self.hotspot_config.password}",
                f"max_num_sta={self.hotspot_config.max_clients}"
            ]
            
            if self.hotspot_config.hidden:
                hostapd_config.append("ignore_broadcast_ssid=1")
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write('\n'.join(hostapd_config))
                tmp_path = tmp.name
            
            subprocess.run(
                ["sudo", "cp", tmp_path, str(self.hostapd_conf)],
                check=True
            )
            os.unlink(tmp_path)
            
            # Configure dnsmasq
            dnsmasq_config = [
                f"interface={self.interface}",
                f"dhcp-range={self.hotspot_config.dhcp_start},{self.hotspot_config.dhcp_end},255.255.255.0,24h",
                f"dhcp-option=3,{self.hotspot_config.gateway}",
                f"dhcp-option=6,{self.hotspot_config.gateway}",
                "server=8.8.8.8",
                "log-queries",
                "log-dhcp",
                "listen-address=127.0.0.1",
                f"listen-address={self.hotspot_config.gateway}"
            ]
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write('\n'.join(dnsmasq_config))
                tmp_path = tmp.name
            
            subprocess.run(
                ["sudo", "cp", tmp_path, str(self.dnsmasq_conf)],
                check=True
            )
            os.unlink(tmp_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error configuring hotspot: {e}")
            return False
    
    def _start_hotspot_services(self) -> bool:
        """Start hotspot services."""
        try:
            # Configure interface
            subprocess.run([
                "sudo", "ip", "addr", "add", 
                f"{self.hotspot_config.gateway}/24", 
                "dev", self.interface
            ], check=True)
            
            subprocess.run([
                "sudo", "ip", "link", "set", self.interface, "up"
            ], check=True)
            
            # Start hostapd
            subprocess.run([
                "sudo", "systemctl", "start", "hostapd"
            ], check=True)
            
            # Start dnsmasq
            subprocess.run([
                "sudo", "systemctl", "start", "dnsmasq"
            ], check=True)
            
            # Enable IP forwarding
            subprocess.run([
                "sudo", "sysctl", "net.ipv4.ip_forward=1"
            ], check=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting hotspot services: {e}")
            return False
    
    def _stop_hotspot(self) -> bool:
        """Stop hotspot services."""
        try:
            # Stop services
            subprocess.run([
                "sudo", "systemctl", "stop", "hostapd"
            ], check=False)
            
            subprocess.run([
                "sudo", "systemctl", "stop", "dnsmasq"
            ], check=False)
            
            # Remove IP address
            subprocess.run([
                "sudo", "ip", "addr", "del", 
                f"{self.hotspot_config.gateway}/24", 
                "dev", self.interface
            ], check=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping hotspot: {e}")
            return False
    
    def _get_ip_address(self) -> Optional[str]:
        """Get current IP address of the interface."""
        try:
            result = subprocess.run([
                "ip", "addr", "show", self.interface
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse IP address from output
                for line in result.stdout.split('\n'):
                    if 'inet ' in line and 'scope global' in line:
                        ip_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
                        if ip_match:
                            return ip_match.group(1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting IP address: {e}")
            return None
    
    def _is_connected(self) -> bool:
        """Check if we're currently connected to a network."""
        try:
            # Check if we have an IP address
            if not self._get_ip_address():
                return False
            
            # Check if we can reach the gateway
            result = subprocess.run([
                "ping", "-c", "1", "-W", "3", "8.8.8.8"
            ], capture_output=True)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error checking connection: {e}")
            return False


# Global instance
_wifi_manager: Optional[WiFiManager] = None

def get_wifi_manager() -> WiFiManager:
    """Get or create global WiFi manager instance."""
    global _wifi_manager
    if _wifi_manager is None:
        _wifi_manager = WiFiManager()
    return _wifi_manager

def reset_wifi_manager():
    """Reset global WiFi manager instance."""
    global _wifi_manager
    if _wifi_manager:
        _wifi_manager.stop_monitoring()
    _wifi_manager = None


if __name__ == "__main__":
    # Test the WiFi manager
    import argparse
    
    parser = argparse.ArgumentParser(description="NightScan WiFi Manager")
    parser.add_argument("--scan", action="store_true", help="Scan for networks")
    parser.add_argument("--connect", help="Connect to network")
    parser.add_argument("--hotspot", action="store_true", help="Start hotspot")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--add-network", nargs=2, metavar=("SSID", "PASSWORD"), 
                       help="Add network")
    
    args = parser.parse_args()
    
    manager = get_wifi_manager()
    
    if args.scan:
        networks = manager.scan_networks()
        print(f"Found {len(networks)} networks:")
        for network in networks:
            print(f"  {network['ssid']}: {network.get('signal_strength', 'N/A')} dBm")
    
    elif args.connect:
        success = manager.connect_to_network(args.connect)
        print(f"Connection {'successful' if success else 'failed'}")
    
    elif args.hotspot:
        success = manager.start_hotspot()
        print(f"Hotspot {'started' if success else 'failed'}")
    
    elif args.add_network:
        ssid, password = args.add_network
        success = manager.add_network(ssid, password)
        print(f"Network {'added' if success else 'failed'}")
    
    elif args.status:
        status = manager.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
    
    else:
        parser.print_help()