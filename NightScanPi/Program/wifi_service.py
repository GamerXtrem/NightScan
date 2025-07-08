from __future__ import annotations

from pathlib import Path
import logging
import json
import time
import threading
from typing import Optional

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from . import wifi_config
from . import camera_trigger
from . import audio_threshold
from .utils.smart_scheduler import get_process_manager
from .wifi_manager import get_wifi_manager as get_advanced_wifi_manager, NetworkSecurity


class CameraPreviewService:
    """Service for streaming camera preview frames."""
    
    def __init__(self):
        self.camera_manager = camera_trigger.get_camera_manager()
        self.streaming = False
        self.preview_thread: Optional[threading.Thread] = None
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        
    def start_preview(self) -> bool:
        """Start camera preview streaming."""
        if not self.camera_manager.is_available():
            return False
            
        if self.streaming:
            return True
            
        self.streaming = True
        self.preview_thread = threading.Thread(target=self._preview_loop)
        self.preview_thread.daemon = True
        self.preview_thread.start()
        return True
    
    def stop_preview(self):
        """Stop camera preview streaming."""
        self.streaming = False
        if self.preview_thread:
            self.preview_thread.join(timeout=5)
            
    def _preview_loop(self):
        """Preview loop for Pi Zero optimized streaming."""
        from .utils.pi_zero_optimizer import is_pi_zero
        
        try:
            if self.camera_manager.api_type == "picamera2":
                self._preview_loop_modern()
            elif self.camera_manager.api_type == "picamera":
                self._preview_loop_legacy()
        except Exception as e:
            logging.error(f"Camera preview error: {e}")
        finally:
            self.streaming = False
            
    def _preview_loop_modern(self):
        """Modern picamera2 preview loop."""
        from picamera2 import Picamera2
        import io
        
        # Pi Zero optimized settings
        resolution = (320, 240)  # Low resolution for Pi Zero
        fps = 5  # Low FPS for Pi Zero
        
        with Picamera2() as camera:
            config = camera.create_video_configuration(
                main={"size": resolution, "format": "RGB888"},
                lores={"size": (160, 120), "format": "YUV420"},
                display="lores"
            )
            camera.configure(config)
            camera.start()
            
            while self.streaming:
                try:
                    # Capture frame as JPEG
                    stream = io.BytesIO()
                    camera.capture_file(stream, format='jpeg')
                    
                    with self.frame_lock:
                        self.frame_buffer = stream.getvalue()
                    
                    time.sleep(1.0 / fps)
                    
                except Exception as e:
                    logging.error(f"Frame capture error: {e}")
                    time.sleep(0.5)
                    
    def _preview_loop_legacy(self):
        """Legacy picamera preview loop."""
        from picamera import PiCamera
        import io
        
        resolution = (320, 240)
        fps = 5
        
        with PiCamera() as camera:
            camera.resolution = resolution
            camera.framerate = fps
            time.sleep(2)  # Camera warm-up
            
            stream = io.BytesIO()
            
            for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                if not self.streaming:
                    break
                    
                with self.frame_lock:
                    self.frame_buffer = stream.getvalue()
                    
                stream.seek(0)
                stream.truncate()
                time.sleep(1.0 / fps)
                
    def get_frame(self) -> Optional[bytes]:
        """Get the latest frame."""
        with self.frame_lock:
            return self.frame_buffer


def create_app(config_path: Path | None = None) -> Flask:
    """Return a Flask app that writes Wi-Fi credentials and provides camera preview."""
    if config_path is None:
        config_path = wifi_config.CONFIG_PATH
    app = Flask(__name__)
    
    # Enable CORS for mobile app
    CORS(app, origins=["*"])
    
    # Global camera preview service
    preview_service = CameraPreviewService()

    @app.post("/wifi")
    def set_wifi():
        """Legacy endpoint - adds network and connects to it."""
        data = request.get_json() or {}
        ssid = data.get("ssid")
        password = data.get("password")
        if not ssid or not password:
            return jsonify({"error": "ssid and password required"}), 400
        
        # Use new WiFi manager
        wifi_manager = get_advanced_wifi_manager()
        
        # Add network with high priority (legacy behavior)
        success = wifi_manager.add_network(
            ssid=ssid,
            password=password,
            priority=100,  # High priority for manually configured networks
            auto_connect=True
        )
        
        if success:
            # Try to connect immediately
            connect_success = wifi_manager.connect_to_network(ssid)
            return jsonify({
                "status": "ok",
                "added": True,
                "connected": connect_success
            })
        else:
            return jsonify({"error": "Failed to add network"}), 500

    @app.get("/camera/status")
    def camera_status():
        """Get camera status and capabilities."""
        camera_info = camera_trigger.get_camera_info()
        return jsonify({
            "camera_available": camera_info["camera_working"],
            "api_type": camera_info["active_api"],
            "sensor_type": camera_info["sensor_type"],
            "streaming": preview_service.streaming,
            "preview_available": camera_info["camera_working"]
        })

    @app.post("/camera/preview/start")
    def start_camera_preview():
        """Start camera preview streaming."""
        if not preview_service.camera_manager.is_available():
            return jsonify({"error": "Camera not available"}), 400
            
        success = preview_service.start_preview()
        if success:
            return jsonify({"status": "preview_started"})
        else:
            return jsonify({"error": "Failed to start preview"}), 500

    @app.post("/camera/preview/stop")
    def stop_camera_preview():
        """Stop camera preview streaming."""
        preview_service.stop_preview()
        return jsonify({"status": "preview_stopped"})

    @app.get("/camera/preview/stream")
    def video_stream():
        """MJPEG video stream endpoint."""
        def generate():
            while preview_service.streaming:
                frame = preview_service.get_frame()
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)  # Small delay to prevent overwhelming
        
        return Response(generate(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.post("/camera/capture")
    def capture_image():
        """Capture a single image and return info."""
        try:
            from tempfile import NamedTemporaryFile
            import os
            
            # Create temp file for capture
            with NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            # Capture image
            success = preview_service.camera_manager.capture_image(tmp_path)
            
            if success:
                # Get file info
                file_size = tmp_path.stat().st_size
                return jsonify({
                    "status": "captured",
                    "file_size": file_size,
                    "timestamp": time.time()
                })
            else:
                return jsonify({"error": "Capture failed"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up temp file
            if tmp_path.exists():
                os.unlink(tmp_path)

    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "nightscan-pi",
            "camera_available": preview_service.camera_manager.is_available(),
            "timestamp": time.time()
        })

    # Audio threshold configuration endpoints
    @app.get("/audio/threshold/status")
    def get_audio_threshold_status():
        """Get current audio threshold configuration and status."""
        try:
            detector = audio_threshold.get_threshold_detector()
            status = detector.get_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/audio/threshold/config")
    def update_audio_threshold_config():
        """Update audio threshold configuration."""
        try:
            data = request.get_json() or {}
            detector = audio_threshold.get_threshold_detector()
            
            # Validate input parameters
            valid_params = {
                'threshold_db', 'min_duration_ms', 'max_silence_ms',
                'adaptive_enabled', 'adaptive_factor', 'environment_preset'
            }
            
            update_params = {k: v for k, v in data.items() if k in valid_params}
            
            if not update_params:
                return jsonify({"error": "No valid parameters provided"}), 400
            
            success = detector.update_config(**update_params)
            
            if success:
                return jsonify({
                    "status": "updated",
                    "updated_params": update_params,
                    "current_config": detector.get_status()["config"]
                })
            else:
                return jsonify({"error": "Failed to update configuration"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/audio/threshold/preset/<preset_name>")
    def apply_audio_threshold_preset(preset_name):
        """Apply a predefined threshold preset."""
        try:
            detector = audio_threshold.get_threshold_detector()
            success = detector.apply_preset(preset_name)
            
            if success:
                return jsonify({
                    "status": "preset_applied",
                    "preset": preset_name,
                    "current_config": detector.get_status()["config"]
                })
            else:
                return jsonify({"error": f"Unknown preset: {preset_name}"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/audio/threshold/calibrate")
    def calibrate_audio_threshold():
        """Calibrate noise floor for threshold detection."""
        try:
            data = request.get_json() or {}
            duration = data.get('duration_seconds', 5)
            
            # Validate duration
            if not isinstance(duration, (int, float)) or duration < 1 or duration > 30:
                return jsonify({"error": "Duration must be between 1 and 30 seconds"}), 400
            
            detector = audio_threshold.get_threshold_detector()
            success = detector.calibrate_noise_floor(int(duration))
            
            if success:
                status = detector.get_status()
                return jsonify({
                    "status": "calibrated",
                    "noise_floor_db": status["adaptive_noise_floor"],
                    "current_config": status["config"]
                })
            else:
                return jsonify({"error": "Calibration failed"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/audio/threshold/test")
    def test_audio_threshold():
        """Test current threshold settings."""
        try:
            data = request.get_json() or {}
            duration = data.get('duration_seconds', 3)
            
            # Validate duration
            if not isinstance(duration, (int, float)) or duration < 1 or duration > 10:
                return jsonify({"error": "Duration must be between 1 and 10 seconds"}), 400
            
            detector = audio_threshold.get_threshold_detector()
            test_result = detector.test_threshold(int(duration))
            
            return jsonify({
                "status": "test_completed",
                "test_result": test_result
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/audio/threshold/presets")
    def get_audio_threshold_presets():
        """Get available threshold presets."""
        try:
            detector = audio_threshold.get_threshold_detector()
            presets = detector.PRESETS
            return jsonify({
                "presets": presets,
                "current_preset": detector.config.environment_preset
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/audio/threshold/live")
    def get_live_audio_levels():
        """Get real-time audio levels for threshold adjustment."""
        try:
            # This endpoint would stream real-time audio levels
            # For now, return current status
            detector = audio_threshold.get_threshold_detector()
            
            # Quick audio sample to get current level
            from .audio_capture import get_audio_config
            audio_config = get_audio_config()
            
            import pyaudio
            p = pyaudio.PyAudio()
            
            stream_kwargs = {
                'format': pyaudio.paInt16,
                'channels': audio_config['channels'],
                'rate': audio_config['sample_rate'],
                'input': True,
                'frames_per_buffer': 512
            }
            
            if audio_config['device_id'] is not None:
                stream_kwargs['input_device_index'] = audio_config['device_id']
            
            stream = p.open(**stream_kwargs)
            
            # Get single sample
            data = stream.read(512, exception_on_overflow=False)
            current_db = detector.calculate_db_level(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            return jsonify({
                "current_db": current_db,
                "threshold_db": detector.config.threshold_db,
                "noise_floor_db": detector.adaptive_noise_floor,
                "above_threshold": current_db > detector.config.threshold_db,
                "timestamp": time.time()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Energy management endpoints
    @app.get("/energy/status")
    def get_energy_status():
        """Get current energy management status."""
        try:
            process_manager = get_process_manager()
            status = process_manager.get_system_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/energy/wifi/activate")
    def activate_wifi():
        """Activate WiFi for specified duration."""
        try:
            data = request.get_json() or {}
            duration_minutes = data.get('duration_minutes', 10)
            
            if not isinstance(duration_minutes, (int, float)) or duration_minutes < 1 or duration_minutes > 120:
                return jsonify({"error": "Duration must be between 1 and 120 minutes"}), 400
            
            wifi_manager = get_wifi_manager()
            success = wifi_manager.activate_wifi(int(duration_minutes))
            
            if success:
                return jsonify({
                    "status": "activated",
                    "duration_minutes": duration_minutes,
                    "message": f"WiFi activated for {duration_minutes} minutes"
                })
            else:
                return jsonify({"error": "Failed to activate WiFi"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/energy/wifi/deactivate")
    def deactivate_wifi():
        """Deactivate WiFi immediately."""
        try:
            wifi_manager = get_wifi_manager()
            success = wifi_manager.deactivate_wifi()
            
            if success:
                return jsonify({
                    "status": "deactivated",
                    "message": "WiFi deactivated successfully"
                })
            else:
                return jsonify({"error": "Failed to deactivate WiFi"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/energy/wifi/extend")
    def extend_wifi():
        """Extend current WiFi session."""
        try:
            data = request.get_json() or {}
            additional_minutes = data.get('additional_minutes', 10)
            
            if not isinstance(additional_minutes, (int, float)) or additional_minutes < 1 or additional_minutes > 60:
                return jsonify({"error": "Additional time must be between 1 and 60 minutes"}), 400
            
            wifi_manager = get_wifi_manager()
            
            if not wifi_manager.is_wifi_active():
                return jsonify({"error": "WiFi is not currently active"}), 400
            
            success = wifi_manager.extend_wifi_session(int(additional_minutes))
            
            if success:
                return jsonify({
                    "status": "extended",
                    "additional_minutes": additional_minutes,
                    "message": f"WiFi session extended by {additional_minutes} minutes"
                })
            else:
                return jsonify({"error": "Failed to extend WiFi session"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/energy/wifi/status")
    def get_wifi_status():
        """Get WiFi activation status."""
        try:
            wifi_manager = get_wifi_manager()
            is_active = wifi_manager.is_wifi_active()
            
            return jsonify({
                "wifi_active": is_active,
                "timestamp": time.time()
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Advanced WiFi Management Endpoints
    @app.get("/wifi/networks")
    def get_wifi_networks():
        """Get all configured WiFi networks."""
        try:
            wifi_manager = get_advanced_wifi_manager()
            networks = wifi_manager.get_networks()
            return jsonify({"networks": networks})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/wifi/networks")
    def add_wifi_network():
        """Add a new WiFi network."""
        try:
            data = request.get_json() or {}
            ssid = data.get("ssid")
            password = data.get("password", "")
            priority = data.get("priority", 50)
            auto_connect = data.get("auto_connect", True)
            hidden = data.get("hidden", False)
            notes = data.get("notes", "")
            
            if not ssid:
                return jsonify({"error": "SSID required"}), 400
            
            # Determine security type
            security = NetworkSecurity.OPEN if not password else NetworkSecurity.WPA2_PSK
            
            wifi_manager = get_advanced_wifi_manager()
            success = wifi_manager.add_network(
                ssid=ssid,
                password=password,
                security=security,
                priority=priority,
                auto_connect=auto_connect,
                hidden=hidden,
                notes=notes
            )
            
            if success:
                return jsonify({"status": "added", "ssid": ssid})
            else:
                return jsonify({"error": "Failed to add network"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.delete("/wifi/networks/<ssid>")
    def remove_wifi_network(ssid):
        """Remove a WiFi network."""
        try:
            wifi_manager = get_advanced_wifi_manager()
            success = wifi_manager.remove_network(ssid)
            
            if success:
                return jsonify({"status": "removed", "ssid": ssid})
            else:
                return jsonify({"error": "Network not found"}), 404
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.put("/wifi/networks/<ssid>")
    def update_wifi_network(ssid):
        """Update WiFi network configuration."""
        try:
            data = request.get_json() or {}
            wifi_manager = get_advanced_wifi_manager()
            
            # Filter valid update fields
            valid_fields = ['password', 'priority', 'auto_connect', 'hidden', 'notes']
            updates = {k: v for k, v in data.items() if k in valid_fields}
            
            if not updates:
                return jsonify({"error": "No valid fields to update"}), 400
            
            success = wifi_manager.update_network(ssid, **updates)
            
            if success:
                return jsonify({"status": "updated", "ssid": ssid})
            else:
                return jsonify({"error": "Network not found"}), 404
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/wifi/scan")
    def scan_wifi_networks():
        """Scan for available WiFi networks."""
        try:
            wifi_manager = get_advanced_wifi_manager()
            networks = wifi_manager.scan_networks()
            return jsonify({"networks": networks})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/wifi/connect/<ssid>")
    def connect_to_wifi_network(ssid):
        """Connect to a specific WiFi network."""
        try:
            data = request.get_json() or {}
            force = data.get("force", False)
            
            wifi_manager = get_advanced_wifi_manager()
            success = wifi_manager.connect_to_network(ssid, force=force)
            
            if success:
                return jsonify({"status": "connected", "ssid": ssid})
            else:
                return jsonify({"error": "Connection failed"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/wifi/connect/best")
    def connect_to_best_network():
        """Connect to the best available network."""
        try:
            wifi_manager = get_advanced_wifi_manager()
            success = wifi_manager.connect_to_best_network()
            
            if success:
                status = wifi_manager.get_status()
                return jsonify({
                    "status": "connected",
                    "network": status.get("connected_network")
                })
            else:
                return jsonify({"error": "No suitable network found"}), 404
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/wifi/disconnect")
    def disconnect_wifi():
        """Disconnect from current network."""
        try:
            wifi_manager = get_advanced_wifi_manager()
            success = wifi_manager.disconnect()
            
            if success:
                return jsonify({"status": "disconnected"})
            else:
                return jsonify({"error": "Disconnection failed"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/wifi/status")
    def get_wifi_detailed_status():
        """Get detailed WiFi status."""
        try:
            wifi_manager = get_advanced_wifi_manager()
            status = wifi_manager.get_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    

    @app.post("/wifi/hotspot/start")
    def start_wifi_hotspot():
        """Start WiFi hotspot mode."""
        try:
            wifi_manager = get_advanced_wifi_manager()
            success = wifi_manager.start_hotspot()
            
            if success:
                return jsonify({"status": "hotspot_started"})
            else:
                return jsonify({"error": "Failed to start hotspot"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/wifi/hotspot/stop")
    def stop_wifi_hotspot():
        """Stop WiFi hotspot mode."""
        try:
            wifi_manager = get_advanced_wifi_manager()
            success = wifi_manager.stop_hotspot()
            
            if success:
                return jsonify({"status": "hotspot_stopped"})
            else:
                return jsonify({"error": "Failed to stop hotspot"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/wifi/hotspot/config")
    def get_hotspot_config():
        """Get current hotspot configuration."""
        try:
            wifi_manager = get_advanced_wifi_manager()
            config = wifi_manager.hotspot_config.to_dict()
            return jsonify(config)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.put("/wifi/hotspot/config")
    def update_hotspot_config():
        """Update hotspot configuration."""
        try:
            data = request.get_json() or {}
            wifi_manager = get_advanced_wifi_manager()
            
            # Update hotspot config
            valid_fields = ['ssid', 'password', 'channel', 'hidden', 'max_clients']
            for field in valid_fields:
                if field in data:
                    setattr(wifi_manager.hotspot_config, field, data[field])
            
            # Save configuration
            wifi_manager._save_configuration()
            
            return jsonify({"status": "updated"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000)
