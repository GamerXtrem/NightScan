#!/usr/bin/env python3
"""
Script de préparation des données pour l'entraînement avec images réelles.
Détecte automatiquement les classes et organise les données en train/val/test.
"""

import os
import shutil
import random
import gc
from pathlib import Path
from typing import Dict, List, Tuple
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparation:
    """Classe pour préparer et organiser les données d'entraînement."""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42):
        """
        Initialise la préparation des données.
        
        Args:
            input_dir: Dossier contenant les sous-dossiers de classes
            output_dir: Dossier de sortie pour les données organisées
            train_ratio: Proportion pour l'entraînement
            val_ratio: Proportion pour la validation
            test_ratio: Proportion pour le test
            seed: Seed pour la reproductibilité
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Vérifier que les ratios sont valides
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Les ratios doivent sommer à 1.0"
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.classes = []
        self.class_to_idx = {}
        self.statistics = {}
        self.image_stats = []  # Pour stocker les statistiques de taille
        
    def discover_classes(self) -> Dict[str, int]:
        """
        Découvre automatiquement les classes depuis les sous-dossiers.
        
        Returns:
            Dictionnaire mapping classe -> index
        """
        logger.info(f"Découverte des classes dans {self.input_dir}")
        
        # Lister tous les sous-dossiers
        subdirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        subdirs.sort()  # Tri alphabétique pour reproductibilité
        
        self.classes = [d.name for d in subdirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        logger.info(f"✅ {len(self.classes)} classes découvertes: {self.classes}")
        
        return self.class_to_idx
    
    def validate_images(self, class_dir: Path) -> List[Path]:
        """
        Valide et filtre les images dans un dossier de classe.
        
        Args:
            class_dir: Dossier de la classe
            
        Returns:
            Liste des chemins d'images valides
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        valid_images = []
        corrupted_count = 0
        
        # Lister toutes les images pour avoir le total
        img_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in valid_extensions]
        total_images = len(img_paths)
        
        for idx, img_path in enumerate(img_paths):
            try:
                # Ouvrir l'image une seule fois pour vérification et dimensions
                img = Image.open(img_path)
                # Vérifier que l'image est valide
                img.verify()
                
                # Réouvrir pour obtenir les dimensions (verify() ferme le fichier)
                img = Image.open(img_path)
                width, height = img.size
                
                # Stocker les stats
                self.image_stats.append({
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height,
                    'size_px': width * height,
                    'class': class_dir.name
                })
                
                # Fermer explicitement l'image
                img.close()
                del img
                
                valid_images.append(img_path)
                
                # Garbage collection périodique tous les 50 images
                if (idx + 1) % 50 == 0:
                    gc.collect()
                    
                # Message de progression tous les 100 images
                if (idx + 1) % 100 == 0:
                    logger.info(f"    Traité {idx + 1}/{total_images} images dans {class_dir.name}...")
                    
            except Exception as e:
                logger.warning(f"Image corrompue ignorée: {img_path} - {e}")
                corrupted_count += 1
        
        # Garbage collection final pour cette classe
        gc.collect()
        
        if corrupted_count > 0:
            logger.info(f"  ⚠️ {corrupted_count} images corrompues ignorées dans {class_dir.name}")
        
        return valid_images
    
    def split_data(self, images: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Divise les images en ensembles train/val/test.
        
        Args:
            images: Liste des chemins d'images
            
        Returns:
            Tuple (train_images, val_images, test_images)
        """
        # Mélanger les images
        shuffled = images.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_images = shuffled[:n_train]
        val_images = shuffled[n_train:n_train + n_val]
        test_images = shuffled[n_train + n_val:]
        
        return train_images, val_images, test_images
    
    def copy_images(self, images: List[Path], split: str, class_name: str):
        """
        Copie les images dans le dossier de destination approprié.
        
        Args:
            images: Liste des chemins d'images
            split: 'train', 'val', ou 'test'
            class_name: Nom de la classe
        """
        dest_dir = self.output_dir / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(images, desc=f"Copie {split}/{class_name}", leave=False):
            dest_path = dest_dir / img_path.name
            shutil.copy2(img_path, dest_path)
    
    def compute_statistics(self) -> Dict:
        """
        Calcule les statistiques sur les données.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        stats = {
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'splits': {},
            'per_class': {}
        }
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            if not split_dir.exists():
                continue
                
            total_images = 0
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    n_images = len(list(class_dir.glob('*')))
                    total_images += n_images
                    
                    if class_dir.name not in stats['per_class']:
                        stats['per_class'][class_dir.name] = {}
                    stats['per_class'][class_dir.name][split] = n_images
            
            stats['splits'][split] = total_images
        
        # Calculer les moyennes et écarts-types des pixels si demandé
        stats['image_stats'] = self.compute_pixel_statistics()
        
        # Ajouter les statistiques de dimensions
        if self.image_stats:
            widths = [s['width'] for s in self.image_stats]
            heights = [s['height'] for s in self.image_stats]
            aspect_ratios = [s['aspect_ratio'] for s in self.image_stats]
            
            stats['dimension_stats'] = {
                'width': {
                    'min': min(widths),
                    'max': max(widths),
                    'mean': np.mean(widths),
                    'median': np.median(widths),
                    'std': np.std(widths)
                },
                'height': {
                    'min': min(heights),
                    'max': max(heights),
                    'mean': np.mean(heights),
                    'median': np.median(heights),
                    'std': np.std(heights)
                },
                'aspect_ratio': {
                    'min': min(aspect_ratios),
                    'max': max(aspect_ratios),
                    'mean': np.mean(aspect_ratios),
                    'median': np.median(aspect_ratios)
                },
                'recommendations': self._get_size_recommendations(widths, heights)
            }
        
        return stats
    
    def compute_pixel_statistics(self, sample_size: int = 100) -> Dict:
        """
        Calcule les statistiques des pixels pour la normalisation.
        
        Args:
            sample_size: Nombre d'images à échantillonner
            
        Returns:
            Dictionnaire avec mean et std par canal
        """
        logger.info("Calcul des statistiques de pixels pour la normalisation...")
        
        train_dir = self.output_dir / 'train'
        if not train_dir.exists():
            return {}
        
        # Échantillonner des images
        all_images = list(train_dir.glob('*/*'))
        sample = random.sample(all_images, min(sample_size, len(all_images)))
        
        pixel_values = []
        for idx, img_path in enumerate(tqdm(sample, desc="Analyse des pixels", leave=False)):
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                img_array = np.array(img) / 255.0
                pixel_values.append(img_array.reshape(-1, 3))
                
                # Fermer explicitement l'image
                img.close()
                del img
                del img_array
                
                # Garbage collection périodique
                if (idx + 1) % 20 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.warning(f"Erreur lors de l'analyse de {img_path}: {e}")
        
        if pixel_values:
            all_pixels = np.vstack(pixel_values)
            mean = all_pixels.mean(axis=0).tolist()
            std = all_pixels.std(axis=0).tolist()
            
            return {
                'mean': mean,
                'std': std,
                'sample_size': len(pixel_values)
            }
        
        return {}
    
    def prepare(self):
        """
        Exécute le pipeline complet de préparation des données.
        """
        logger.info("🚀 Début de la préparation des données")
        
        # 1. Découvrir les classes
        self.discover_classes()
        
        # 2. Créer la structure de dossiers
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        
        # 3. Traiter chaque classe
        for class_name in tqdm(self.classes, desc="Traitement des classes"):
            class_dir = self.input_dir / class_name
            
            # Valider les images
            valid_images = self.validate_images(class_dir)
            logger.info(f"Classe '{class_name}': {len(valid_images)} images valides")
            
            if len(valid_images) == 0:
                logger.warning(f"⚠️ Aucune image valide pour la classe '{class_name}'")
                continue
            
            # Diviser les données
            train_imgs, val_imgs, test_imgs = self.split_data(valid_images)
            
            # Copier les images
            self.copy_images(train_imgs, 'train', class_name)
            self.copy_images(val_imgs, 'val', class_name)
            self.copy_images(test_imgs, 'test', class_name)
            
            logger.info(f"  ✅ {class_name}: train={len(train_imgs)}, "
                       f"val={len(val_imgs)}, test={len(test_imgs)}")
            
            # Garbage collection après chaque classe pour libérer la mémoire
            gc.collect()
        
        # 4. Calculer et sauvegarder les statistiques
        self.statistics = self.compute_statistics()
        
        # Sauvegarder les métadonnées
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        logger.info(f"✅ Métadonnées sauvegardées dans {metadata_path}")
        
        # 5. Afficher le résumé
        self.print_summary()
    
    def print_summary(self):
        """Affiche un résumé des données préparées."""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE LA PRÉPARATION DES DONNÉES")
        print("="*60)
        print(f"Nombre de classes: {self.statistics['num_classes']}")
        print(f"Classes: {', '.join(self.statistics['classes'][:10])}")
        if len(self.statistics['classes']) > 10:
            print(f"         ... et {len(self.statistics['classes']) - 10} autres")
        
        print("\n📈 Distribution des données:")
        for split, count in self.statistics['splits'].items():
            percentage = (count / sum(self.statistics['splits'].values())) * 100
            print(f"  {split:5}: {count:6} images ({percentage:.1f}%)")
        
        if 'image_stats' in self.statistics and self.statistics['image_stats']:
            print("\n🎨 Statistiques des pixels (pour normalisation):")
            print(f"  Mean (RGB): {[f'{x:.3f}' for x in self.statistics['image_stats']['mean']]}")
            print(f"  Std  (RGB): {[f'{x:.3f}' for x in self.statistics['image_stats']['std']]}")
        
        if 'dimension_stats' in self.statistics:
            dims = self.statistics['dimension_stats']
            print("\n📏 Dimensions des images:")
            print(f"  Largeur: {dims['width']['min']:.0f}-{dims['width']['max']:.0f}px (moy: {dims['width']['mean']:.0f}px)")
            print(f"  Hauteur: {dims['height']['min']:.0f}-{dims['height']['max']:.0f}px (moy: {dims['height']['mean']:.0f}px)")
            print(f"  Ratio d'aspect: {dims['aspect_ratio']['min']:.2f}-{dims['aspect_ratio']['max']:.2f}")
            
            if dims['recommendations']:
                print("\n⚠️ Recommandations:")
                for rec in dims['recommendations']:
                    print(f"  - {rec}")
        
        print("\n✅ Données prêtes pour l'entraînement!")
        print(f"📁 Dossier de sortie: {self.output_dir}")
        print("="*60)
    
    def _get_size_recommendations(self, widths: list, heights: list) -> list:
        """Génère des recommandations basées sur les tailles d'images."""
        recommendations = []
        
        min_dim = min(min(widths), min(heights))
        max_dim = max(max(widths), max(heights))
        
        if min_dim < 224:
            recommendations.append(f"{sum(1 for w, h in zip(widths, heights) if min(w, h) < 224)} images < 224px (taille minimale pour EfficientNet)")
        
        if max_dim > 3000:
            recommendations.append(f"{sum(1 for w, h in zip(widths, heights) if max(w, h) > 3000)} images > 3000px (considérer le prétraitement pour économiser la mémoire)")
        
        # Vérifier les ratios extrêmes
        extreme_ratios = sum(1 for w, h in zip(widths, heights) if w/h > 2.5 or h/w > 2.5)
        if extreme_ratios > 0:
            recommendations.append(f"{extreme_ratios} images avec ratio d'aspect extrême (>2.5:1)")
        
        if not recommendations:
            recommendations.append("Toutes les images ont des dimensions optimales")
        
        return recommendations


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Prépare les données pour l'entraînement")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Dossier contenant les sous-dossiers de classes')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Dossier de sortie pour les données organisées')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proportion pour l\'entraînement (défaut: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Proportion pour la validation (défaut: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Proportion pour le test (défaut: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour la reproductibilité (défaut: 42)')
    
    args = parser.parse_args()
    
    # Vérifier que le dossier d'entrée existe
    if not Path(args.input_dir).exists():
        logger.error(f"Le dossier d'entrée n'existe pas: {args.input_dir}")
        return
    
    # Créer l'objet de préparation
    prep = DataPreparation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Exécuter la préparation
    prep.prepare()


if __name__ == "__main__":
    main()