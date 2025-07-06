"""
Shared modules for NightScan to eliminate code duplication.
"""

from .training_framework import BaseTrainer, AudioTrainer, ImageTrainer

__all__ = ['BaseTrainer', 'AudioTrainer', 'ImageTrainer']
