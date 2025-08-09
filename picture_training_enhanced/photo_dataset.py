#!/usr/bin/env python3
"""
Dataset personnalisé pour l'entraînement avec images réelles.
Utilise ImageFolder de PyTorch avec des augmentations adaptées.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple, Dict, Any
import json
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

class PhotoDataset:
    """
    Gestionnaire de dataset pour les images de wildlife.
    Utilise ImageFolder avec augmentations personnalisées.
    """
    
    def __init__(self, data_dir: str, metadata_path: Optional[str] = None):
        """
        Initialise le dataset.
        
        Args:
            data_dir: Dossier racine contenant train/val/test
            metadata_path: Chemin vers le fichier de métadonnées (optionnel)
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = metadata_path
        
        # Charger les métadonnées si disponibles
        self.metadata = self._load_metadata()
        
        # Statistiques par défaut (ImageNet) ou personnalisées
        if self.metadata and 'image_stats' in self.metadata:
            self.mean = self.metadata['image_stats'].get('mean', [0.485, 0.456, 0.406])
            self.std = self.metadata['image_stats'].get('std', [0.229, 0.224, 0.225])
        else:
            self.mean = [0.485, 0.456, 0.406]  # ImageNet
            self.std = [0.229, 0.224, 0.225]   # ImageNet
        
        # Déterminer le nombre de classes et récupérer les noms
        train_dir = self.data_dir / 'train'
        if train_dir.exists():
            class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
            self.num_classes = len(class_dirs)
            self.classes = [d.name for d in class_dirs]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.num_classes = 0
            self.classes = []
            self.class_to_idx = {}
        
        # Si les métadonnées contiennent les classes, les utiliser pour cohérence
        if self.metadata and 'classes' in self.metadata:
            self.classes = self.metadata['classes']
            self.class_to_idx = self.metadata.get('class_to_idx', {cls: idx for idx, cls in enumerate(self.classes)})
        
        logger.info(f"Dataset initialisé: {self.num_classes} classes détectées")
        if self.classes:
            logger.info(f"Classes: {', '.join(self.classes[:5])}{'...' if len(self.classes) > 5 else ''}")
        logger.info(f"Normalisation - Mean: {self.mean}, Std: {self.std}")
    
    def _load_metadata(self) -> Optional[Dict]:
        """Charge les métadonnées du dataset."""
        if self.metadata_path:
            metadata_file = Path(self.metadata_path)
        else:
            metadata_file = self.data_dir / 'dataset_metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_train_transforms(self, image_size: int = 224, 
                           augmentation_level: str = 'moderate') -> transforms.Compose:
        """
        Crée les transformations pour l'entraînement.
        
        Args:
            image_size: Taille de l'image finale
            augmentation_level: 'light', 'moderate', 'heavy'
            
        Returns:
            Composition de transformations
        """
        transform_list = []
        
        # Redimensionnement et crop aléatoire
        if augmentation_level in ['moderate', 'heavy']:
            transform_list.append(transforms.RandomResizedCrop(
                image_size, 
                scale=(0.8, 1.0) if augmentation_level == 'moderate' else (0.7, 1.0),
                ratio=(0.75, 1.33)
            ))
        else:
            transform_list.extend([
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size)
            ])
        
        # Augmentations géométriques
        if augmentation_level != 'light':
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            if augmentation_level == 'heavy':
                transform_list.append(transforms.RandomVerticalFlip(p=0.2))
                transform_list.append(transforms.RandomRotation(degrees=20))
            else:
                transform_list.append(transforms.RandomRotation(degrees=10))
        
        # Augmentations de couleur
        if augmentation_level == 'heavy':
            transform_list.append(transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ))
            transform_list.append(transforms.RandomGrayscale(p=0.1))
        elif augmentation_level == 'moderate':
            transform_list.append(transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ))
        
        # Ajout possible d'autres augmentations
        if augmentation_level == 'heavy':
            transform_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.3))
            transform_list.append(transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ))
        
        # Conversion en tensor et normalisation
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Ajout optionnel de RandomErasing (Cutout)
        if augmentation_level == 'heavy':
            transform_list.append(transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3)
            ))
        
        return transforms.Compose(transform_list)
    
    def get_val_transforms(self, image_size: int = 224) -> transforms.Compose:
        """
        Crée les transformations pour la validation/test.
        
        Args:
            image_size: Taille de l'image finale
            
        Returns:
            Composition de transformations
        """
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def create_dataloader(self, split: str = 'train', 
                         batch_size: int = 32,
                         num_workers: int = 4,
                         image_size: int = 224,
                         augmentation_level: str = 'moderate',
                         pin_memory: bool = True,
                         persistent_workers: bool = True,
                         prefetch_factor: int = 2) -> DataLoader:
        """
        Crée un DataLoader optimisé pour le GPU.
        
        Args:
            split: 'train', 'val', ou 'test'
            batch_size: Taille du batch
            num_workers: Nombre de workers pour le chargement
            image_size: Taille des images
            augmentation_level: Niveau d'augmentation pour train
            pin_memory: Optimisation GPU
            persistent_workers: Garder les workers entre les epochs
            prefetch_factor: Nombre de batches à précharger
            
        Returns:
            DataLoader configuré
        """
        split_dir = self.data_dir / split
        
        if not split_dir.exists():
            raise ValueError(f"Le dossier {split_dir} n'existe pas")
        
        # Sélectionner les transformations
        if split == 'train':
            transform = self.get_train_transforms(image_size, augmentation_level)
            shuffle = True
            drop_last = True  # Pour éviter les problèmes avec BatchNorm
        else:
            transform = self.get_val_transforms(image_size)
            shuffle = False
            drop_last = False
        
        # Créer le dataset
        dataset = datasets.ImageFolder(
            root=split_dir,
            transform=transform
        )
        
        # Vérifier que le dataset n'est pas vide
        if len(dataset) == 0:
            raise ValueError(f"Aucune image trouvée dans {split_dir}")
        
        # Vérifier la cohérence des classes
        if split == 'train' and not self.classes:
            # Si on n'a pas encore les classes, les récupérer depuis ImageFolder
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
            self.num_classes = len(self.classes)
        
        logger.info(f"Dataset {split}: {len(dataset)} images, {len(dataset.classes)} classes")
        
        # Créer le DataLoader avec optimisations
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        return dataloader
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calcule les poids des classes pour gérer le déséquilibre.
        
        Returns:
            Tensor avec les poids par classe
        """
        if not self.metadata or 'per_class' not in self.metadata:
            # Poids uniformes si pas de métadonnées
            return torch.ones(self.num_classes)
        
        # Calculer les poids inversement proportionnels au nombre d'échantillons
        class_counts = []
        for class_name in sorted(self.metadata['per_class'].keys()):
            count = self.metadata['per_class'][class_name].get('train', 0)
            class_counts.append(count if count > 0 else 1)
        
        class_counts = np.array(class_counts)
        
        # Méthode 1: Inverse de la fréquence
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(weights)
        
        # Méthode 2: Effective number of samples (alternative)
        # beta = 0.999
        # effective_num = 1.0 - np.power(beta, class_counts)
        # weights = (1.0 - beta) / effective_num
        # weights = weights / weights.sum() * len(weights)
        
        return torch.FloatTensor(weights)
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Retourne des informations sur le dataset.
        
        Returns:
            Dictionnaire avec les informations
        """
        info = {
            'num_classes': self.num_classes,
            'normalization': {
                'mean': self.mean,
                'std': self.std
            }
        }
        
        # Toujours inclure les classes récupérées
        info['classes'] = self.classes
        info['class_to_idx'] = self.class_to_idx
        
        if self.metadata:
            info['splits'] = self.metadata.get('splits', {})
        
        return info
    
    def create_all_dataloaders(self, batch_size: int = 32,
                               num_workers: int = 4,
                               image_size: int = 224,
                               augmentation_level: str = 'moderate') -> Dict[str, DataLoader]:
        """
        Crée les DataLoaders pour tous les splits.
        
        Args:
            batch_size: Taille du batch
            num_workers: Nombre de workers
            image_size: Taille des images
            augmentation_level: Niveau d'augmentation
            
        Returns:
            Dictionnaire avec les DataLoaders
        """
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                dataloaders[split] = self.create_dataloader(
                    split=split,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    image_size=image_size,
                    augmentation_level=augmentation_level if split == 'train' else 'light'
                )
        
        return dataloaders


def test_dataset():
    """Fonction de test pour vérifier le dataset."""
    import matplotlib.pyplot as plt
    
    # Créer un dataset de test
    dataset = PhotoDataset('./data/processed')
    
    # Créer les dataloaders
    dataloaders = dataset.create_all_dataloaders(
        batch_size=16,
        num_workers=2,
        augmentation_level='moderate'
    )
    
    # Afficher les infos
    info = dataset.get_data_info()
    print(f"Dataset info: {info}")
    
    # Obtenir les poids des classes
    weights = dataset.get_class_weights()
    print(f"Class weights: {weights}")
    
    # Tester le chargement d'un batch
    if 'train' in dataloaders:
        dataloader = dataloaders['train']
        images, labels = next(iter(dataloader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        
        # Optionnel: visualiser quelques images
        # dénormaliser pour l'affichage
        mean = torch.tensor(dataset.mean).reshape(3, 1, 1)
        std = torch.tensor(dataset.std).reshape(3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        
        # Afficher une grille d'images
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                img = images[i].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(f"Label: {labels[i].item()}")
                ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_dataset()