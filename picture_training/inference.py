#!/usr/bin/env python3
"""
Picture inference module for NightScan.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import from the enhanced version
try:
    from picture_training_enhanced.scripts.predict_enhanced import ModelInference, predict_image
except ImportError:
    # Fallback implementation
    class ModelInference:
        """Basic model inference class."""
        def __init__(self, model_path=None):
            self.model_path = model_path
            print(f"Initialized ModelInference with path: {model_path}")
        
        def predict(self, image_path):
            """Placeholder prediction method."""
            return {"species": "unknown", "confidence": 0.0}
    
    def predict_image(image_path, model=None):
        """Placeholder prediction function."""
        return [{"species": "unknown", "confidence": 0.0}]

__all__ = ['ModelInference', 'predict_image']