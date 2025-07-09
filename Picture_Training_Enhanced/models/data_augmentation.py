"""
Data Augmentation Pipeline for Wildlife Image Classification

Advanced data augmentation techniques optimized for nocturnal wildlife imagery,
including geometric transformations, photometric adjustments, and advanced
techniques like CutMix and MixUp.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageEnhance, ImageFilter
import cv2


class WildlifeImageAugmentation:
    """Advanced augmentation pipeline for wildlife images."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize augmentation pipeline.
        
        Args:
            config_dict: Configuration dictionary with augmentation parameters
        """
        self.config = config_dict
        self.input_size = config_dict.get("input_size", (224, 224))
        
        # Basic augmentation parameters
        self.rotation_degrees = config_dict.get("rotation_degrees", 15)
        self.color_jitter_strength = config_dict.get("color_jitter_strength", 0.3)
        self.horizontal_flip_prob = config_dict.get("horizontal_flip_prob", 0.5)
        self.vertical_flip_prob = config_dict.get("vertical_flip_prob", 0.2)
        
        # Advanced augmentation parameters
        self.use_cutmix = config_dict.get("use_cutmix", True)
        self.cutmix_prob = config_dict.get("cutmix_prob", 0.5)
        self.cutmix_alpha = config_dict.get("cutmix_alpha", 1.0)
        
        self.use_mixup = config_dict.get("use_mixup", True)
        self.mixup_prob = config_dict.get("mixup_prob", 0.3)
        self.mixup_alpha = config_dict.get("mixup_alpha", 0.2)
        
        # Nocturnal-specific augmentations
        self.use_night_vision = config_dict.get("use_night_vision", True)
        self.night_vision_prob = config_dict.get("night_vision_prob", 0.3)
        self.infrared_prob = config_dict.get("infrared_prob", 0.2)
        
        # Build augmentation pipeline
        self.train_transforms = self._build_train_transforms()
        self.val_transforms = self._build_val_transforms()
    
    def _build_train_transforms(self) -> transforms.Compose:
        """Build training augmentation pipeline."""
        transform_list = []
        
        # Resize and basic transforms
        transform_list.extend([
            transforms.Resize((self.input_size[0] + 32, self.input_size[1] + 32)),
            transforms.RandomCrop(self.input_size),
        ])
        
        # Geometric augmentations
        if self.horizontal_flip_prob > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob))
        
        if self.vertical_flip_prob > 0:
            transform_list.append(transforms.RandomVerticalFlip(p=self.vertical_flip_prob))
        
        if self.rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(degrees=self.rotation_degrees))
        
        # Photometric augmentations
        if self.color_jitter_strength > 0:
            transform_list.append(transforms.ColorJitter(
                brightness=self.color_jitter_strength,
                contrast=self.color_jitter_strength,
                saturation=self.color_jitter_strength,
                hue=self.color_jitter_strength / 2
            ))
        
        # Advanced augmentations
        transform_list.extend([
            RandomGaussianBlur(p=0.2),
            RandomNoise(p=0.15),
            RandomGammaCorrection(p=0.25),
            RandomShadow(p=0.2),
        ])
        
        # Nocturnal-specific augmentations
        if self.use_night_vision:
            transform_list.append(NightVisionSimulation(p=self.night_vision_prob))
        
        transform_list.append(InfraredSimulation(p=self.infrared_prob))
        
        # Final normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def _build_val_transforms(self) -> transforms.Compose:
        """Build validation (no augmentation) pipeline."""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image: Image.Image, is_training: bool = True) -> torch.Tensor:
        """Apply augmentation pipeline to image."""
        if is_training:
            return self.train_transforms(image)
        else:
            return self.val_transforms(image)


class RandomGaussianBlur:
    """Random Gaussian blur augmentation."""
    
    def __init__(self, p: float = 0.2, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image


class RandomNoise:
    """Random noise addition."""
    
    def __init__(self, p: float = 0.15, intensity: float = 0.1):
        self.p = p
        self.intensity = intensity
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img_array = np.array(image)
            noise = np.random.normal(0, self.intensity * 255, img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_img)
        return image


class RandomGammaCorrection:
    """Random gamma correction for lighting variation."""
    
    def __init__(self, p: float = 0.25, gamma_range: Tuple[float, float] = (0.5, 2.0)):
        self.p = p
        self.gamma_range = gamma_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            img_array = np.array(image)
            corrected = np.clip(255 * (img_array / 255) ** gamma, 0, 255).astype(np.uint8)
            return Image.fromarray(corrected)
        return image


class RandomShadow:
    """Random shadow simulation."""
    
    def __init__(self, p: float = 0.2):
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            # Create shadow mask
            shadow_intensity = random.uniform(0.3, 0.7)
            shadow_size = random.uniform(0.2, 0.8)
            
            # Random shadow position
            x1 = random.randint(0, int(w * (1 - shadow_size)))
            y1 = random.randint(0, int(h * (1 - shadow_size)))
            x2 = int(x1 + w * shadow_size)
            y2 = int(y1 + h * shadow_size)
            
            # Apply shadow
            shadow_img = img_array.copy()
            shadow_img[y1:y2, x1:x2] = (shadow_img[y1:y2, x1:x2] * shadow_intensity).astype(np.uint8)
            
            return Image.fromarray(shadow_img)
        return image


class NightVisionSimulation:
    """Simulate night vision/thermal imaging effects."""
    
    def __init__(self, p: float = 0.3):
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # Convert to grayscale and apply green tint
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Enhance contrast
            enhanced = cv2.equalizeHist(img_array)
            
            # Add green tint
            green_tinted = np.zeros((enhanced.shape[0], enhanced.shape[1], 3), dtype=np.uint8)
            green_tinted[:, :, 1] = enhanced  # Green channel
            green_tinted[:, :, 0] = enhanced * 0.3  # Slight red
            
            return Image.fromarray(green_tinted)
        return image


class InfraredSimulation:
    """Simulate infrared imaging effects."""
    
    def __init__(self, p: float = 0.2):
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img_array = np.array(image)
            
            # Convert to thermal-like color map
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            thermal_rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
            
            return Image.fromarray(thermal_rgb)
        return image


class CutMix:
    """CutMix augmentation implementation."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix augmentation to a batch.
        
        Args:
            batch: Input batch of images [N, C, H, W]
            labels: Corresponding labels [N]
            
        Returns:
            Augmented batch and mixed labels
        """
        indices = torch.randperm(batch.size(0))
        shuffled_batch = batch[indices]
        shuffled_labels = labels[indices]
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get bounding box coordinates
        batch_size, _, height, width = batch.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_rat)
        cut_h = int(height * cut_rat)
        
        # Random position
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        
        # Apply CutMix
        mixed_batch = batch.clone()
        mixed_batch[:, :, bby1:bby2, bbx1:bbx2] = shuffled_batch[:, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))
        
        return mixed_batch, labels, shuffled_labels, lam


class MixUp:
    """MixUp augmentation implementation."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, batch: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp augmentation to a batch.
        
        Args:
            batch: Input batch of images [N, C, H, W]
            labels: Corresponding labels [N]
            
        Returns:
            Augmented batch and mixed labels
        """
        indices = torch.randperm(batch.size(0))
        shuffled_batch = batch[indices]
        shuffled_labels = labels[indices]
        
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_batch = lam * batch + (1 - lam) * shuffled_batch
        
        return mixed_batch, labels, shuffled_labels, lam


def cutmix_criterion(pred: torch.Tensor, labels_a: torch.Tensor, labels_b: torch.Tensor, 
                    lam: float, criterion: nn.Module) -> torch.Tensor:
    """Compute loss for CutMix augmentation."""
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


def mixup_criterion(pred: torch.Tensor, labels_a: torch.Tensor, labels_b: torch.Tensor, 
                   lam: float, criterion: nn.Module) -> torch.Tensor:
    """Compute loss for MixUp augmentation."""
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


class AugmentationManager:
    """Manager for all augmentation techniques."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_augmentation = WildlifeImageAugmentation(config)
        
        # Initialize advanced augmentations
        self.cutmix = CutMix(alpha=config.get("cutmix_alpha", 1.0))
        self.mixup = MixUp(alpha=config.get("mixup_alpha", 0.2))
        
        self.use_cutmix = config.get("use_cutmix", True)
        self.use_mixup = config.get("use_mixup", True)
        self.cutmix_prob = config.get("cutmix_prob", 0.5)
        self.mixup_prob = config.get("mixup_prob", 0.3)
    
    def apply_batch_augmentation(self, batch: torch.Tensor, labels: torch.Tensor, 
                               is_training: bool = True) -> Tuple[torch.Tensor, Any]:
        """
        Apply batch-level augmentations (CutMix, MixUp).
        
        Args:
            batch: Input batch [N, C, H, W]
            labels: Labels [N]
            is_training: Whether in training mode
            
        Returns:
            Augmented batch and label information
        """
        if not is_training:
            return batch, labels
        
        # Decide which augmentation to apply
        augmentation_choice = random.random()
        
        if self.use_cutmix and augmentation_choice < self.cutmix_prob:
            mixed_batch, labels_a, labels_b, lam = self.cutmix(batch, labels)
            return mixed_batch, {
                "type": "cutmix",
                "labels_a": labels_a,
                "labels_b": labels_b,
                "lam": lam
            }
        elif self.use_mixup and augmentation_choice < self.cutmix_prob + self.mixup_prob:
            mixed_batch, labels_a, labels_b, lam = self.mixup(batch, labels)
            return mixed_batch, {
                "type": "mixup",
                "labels_a": labels_a,
                "labels_b": labels_b,
                "lam": lam
            }
        else:
            return batch, labels
    
    def compute_loss(self, pred: torch.Tensor, label_info: Any, criterion: nn.Module) -> torch.Tensor:
        """
        Compute loss considering augmentation type.
        
        Args:
            pred: Model predictions
            label_info: Label information (can be tensor or dict)
            criterion: Loss function
            
        Returns:
            Computed loss
        """
        if isinstance(label_info, dict):
            if label_info["type"] == "cutmix":
                return cutmix_criterion(pred, label_info["labels_a"], label_info["labels_b"], 
                                      label_info["lam"], criterion)
            elif label_info["type"] == "mixup":
                return mixup_criterion(pred, label_info["labels_a"], label_info["labels_b"], 
                                     label_info["lam"], criterion)
        else:
            return criterion(pred, label_info)


# Test the augmentation pipeline
if __name__ == "__main__":
    # Test configuration
    config = {
        "input_size": (224, 224),
        "rotation_degrees": 15,
        "color_jitter_strength": 0.3,
        "horizontal_flip_prob": 0.5,
        "vertical_flip_prob": 0.2,
        "use_cutmix": True,
        "cutmix_prob": 0.5,
        "cutmix_alpha": 1.0,
        "use_mixup": True,
        "mixup_prob": 0.3,
        "mixup_alpha": 0.2,
        "use_night_vision": True,
        "night_vision_prob": 0.3,
        "infrared_prob": 0.2
    }
    
    # Test augmentation manager
    aug_manager = AugmentationManager(config)
    print("Augmentation Manager initialized successfully!")
    
    # Test with dummy batch
    dummy_batch = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.randint(0, 8, (4,))
    
    augmented_batch, label_info = aug_manager.apply_batch_augmentation(dummy_batch, dummy_labels)
    print(f"Augmented batch shape: {augmented_batch.shape}")
    print(f"Label info type: {type(label_info)}")
    
    # Test loss computation
    dummy_pred = torch.randn(4, 8)
    criterion = nn.CrossEntropyLoss()
    loss = aug_manager.compute_loss(dummy_pred, label_info, criterion)
    print(f"Loss computed: {loss.item():.4f}")