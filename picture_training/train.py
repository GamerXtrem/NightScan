#!/usr/bin/env python3
"""
Picture training module for NightScan.
This is a placeholder that imports from the enhanced version.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import from the enhanced version
try:
    from picture_training_enhanced.scripts.train_enhanced import main
except ImportError:
    print("Error: Could not import picture_training_enhanced module")
    print("Please ensure picture_training_enhanced is properly installed")
    sys.exit(1)

if __name__ == "__main__":
    main()