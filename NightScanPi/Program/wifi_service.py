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
        data = request.get_json() or {}
        ssid = data.get("ssid")
        password = data.get("password")
        if not ssid or not password:
            return jsonify({"error": "ssid and password required"}), 400
        wifi_config.write_wifi_config(ssid, password, config_path)
        return jsonify({"status": "ok"})

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

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000)
