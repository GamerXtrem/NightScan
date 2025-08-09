#!/usr/bin/env python3
"""
Script de prétraitement des images pour optimiser leur taille et qualité
avant l'entraînement. Gère les images de tailles non standardisées.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Préprocesseur d'images pour optimisation avant entraînement."""
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 max_size: int = 1500,
                 min_size: int = 200,
                 target_format: str = 'JPEG',
                 quality: int = 95,
                 check_blur: bool = True,
                 blur_threshold: float = 100.0):
        """
        Initialise le préprocesseur.
        
        Args:
            input_dir: Dossier source des images
            output_dir: Dossier de sortie pour les images traitées
            max_size: Taille maximale (largeur ou hauteur) en pixels
            min_size: Taille minimale acceptable en pixels
            target_format: Format de sortie (JPEG, PNG)
            quality: Qualité de compression JPEG (1-100)
            check_blur: Vérifier si les images sont floues
            blur_threshold: Seuil de détection du flou (variance du Laplacien)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_size = max_size
        self.min_size = min_size
        self.target_format = target_format.upper()
        self.quality = quality
        self.check_blur = check_blur
        self.blur_threshold = blur_threshold
        
        # Statistiques
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'skipped_small': 0,
            'skipped_blur': 0,
            'skipped_corrupt': 0,
            'resized': 0,
            'format_converted': 0,
            'size_distribution': {},
            'aspect_ratios': [],
            'blur_scores': [],
            'original_total_size_mb': 0,
            'processed_total_size_mb': 0
        }
        
        # Extensions supportées
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
    def calculate_blur_score(self, image_path: Path) -> float:
        """
        Calcule le score de netteté d'une image.
        Plus le score est élevé, plus l'image est nette.
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Score de netteté (variance du Laplacien)
        """
        try:
            # Charger l'image en niveaux de gris
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            
            # Calculer la variance du Laplacien
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            score = laplacian.var()
            
            return score
        except Exception as e:
            logger.warning(f"Erreur calcul netteté pour {image_path}: {e}")
            return 0.0
    
    def get_optimal_size(self, width: int, height: int) -> Tuple[int, int]:
        """
        Calcule la taille optimale en respectant le ratio d'aspect.
        
        Args:
            width: Largeur originale
            height: Hauteur originale
            
        Returns:
            Tuple (nouvelle_largeur, nouvelle_hauteur)
        """
        # Si l'image est déjà plus petite que max_size, ne pas l'agrandir
        if max(width, height) <= self.max_size:
            return width, height
        
        # Calculer le facteur de redimensionnement
        if width > height:
            scale = self.max_size / width
        else:
            scale = self.max_size / height
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return new_width, new_height
    
    def process_image(self, image_path: Path, output_path: Path) -> Dict:
        """
        Traite une image individuelle.
        
        Args:
            image_path: Chemin de l'image source
            output_path: Chemin de sortie
            
        Returns:
            Dictionnaire avec les informations de traitement
        """
        result = {
            'path': str(image_path),
            'status': 'pending',
            'original_size': None,
            'new_size': None,
            'blur_score': None,
            'aspect_ratio': None
        }
        
        try:
            # Obtenir la taille du fichier original
            original_file_size = image_path.stat().st_size / (1024 * 1024)  # MB
            self.stats['original_total_size_mb'] += original_file_size
            
            # Ouvrir l'image
            with Image.open(image_path) as img:
                # Convertir en RGB si nécessaire (pour JPEG)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                width, height = img.size
                result['original_size'] = (width, height)
                result['aspect_ratio'] = width / height
                self.stats['aspect_ratios'].append(result['aspect_ratio'])
                
                # Vérifier la taille minimale
                if min(width, height) < self.min_size:
                    self.stats['skipped_small'] += 1
                    result['status'] = 'skipped_small'
                    logger.warning(f"Image trop petite ignorée: {image_path} ({width}x{height})")
                    return result
                
                # Vérifier le flou si demandé
                if self.check_blur:
                    blur_score = self.calculate_blur_score(image_path)
                    result['blur_score'] = blur_score
                    self.stats['blur_scores'].append(blur_score)
                    
                    if blur_score < self.blur_threshold:
                        self.stats['skipped_blur'] += 1
                        result['status'] = 'skipped_blur'
                        logger.warning(f"Image floue ignorée: {image_path} (score: {blur_score:.2f})")
                        return result
                
                # Redimensionner si nécessaire
                new_width, new_height = self.get_optimal_size(width, height)
                if (new_width, new_height) != (width, height):
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    self.stats['resized'] += 1
                    result['new_size'] = (new_width, new_height)
                else:
                    result['new_size'] = result['original_size']
                
                # Créer le dossier de sortie si nécessaire
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Sauvegarder dans le format cible
                save_kwargs = {}
                if self.target_format == 'JPEG':
                    save_kwargs['quality'] = self.quality
                    save_kwargs['optimize'] = True
                    # Changer l'extension si nécessaire
                    if output_path.suffix.lower() not in ['.jpg', '.jpeg']:
                        output_path = output_path.with_suffix('.jpg')
                        self.stats['format_converted'] += 1
                
                img.save(output_path, format=self.target_format, **save_kwargs)
                
                # Obtenir la taille du fichier traité
                processed_file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                self.stats['processed_total_size_mb'] += processed_file_size
                
                self.stats['processed'] += 1
                result['status'] = 'processed'
                
                # Mettre à jour la distribution des tailles
                size_category = self._get_size_category(new_width, new_height)
                self.stats['size_distribution'][size_category] = \
                    self.stats['size_distribution'].get(size_category, 0) + 1
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {image_path}: {e}")
            self.stats['skipped_corrupt'] += 1
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _get_size_category(self, width: int, height: int) -> str:
        """Catégorise la taille de l'image."""
        max_dim = max(width, height)
        if max_dim < 300:
            return 'très_petite (<300px)'
        elif max_dim < 500:
            return 'petite (300-500px)'
        elif max_dim < 800:
            return 'moyenne (500-800px)'
        elif max_dim < 1200:
            return 'grande (800-1200px)'
        elif max_dim < 2000:
            return 'très_grande (1200-2000px)'
        else:
            return 'énorme (>2000px)'
    
    def process_directory(self, num_workers: int = 4):
        """
        Traite tous les sous-dossiers (classes) du répertoire.
        
        Args:
            num_workers: Nombre de threads parallèles
        """
        logger.info(f"Début du prétraitement de {self.input_dir}")
        
        # Collecter toutes les images à traiter
        tasks = []
        for class_dir in self.input_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            output_class_dir = self.output_dir / class_name
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.valid_extensions:
                    output_path = output_class_dir / img_path.name
                    tasks.append((img_path, output_path))
                    self.stats['total_images'] += 1
        
        logger.info(f"📊 {self.stats['total_images']} images trouvées")
        
        # Traiter en parallèle avec barre de progression
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.process_image, img_path, out_path): (img_path, out_path)
                for img_path, out_path in tasks
            }
            
            with tqdm(total=len(tasks), desc="Traitement des images") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    pbar.update(1)
                    
                    # Afficher les infos importantes
                    if result['status'] == 'processed' and result.get('new_size') != result.get('original_size'):
                        pbar.set_postfix_str(f"Redimensionné: {result['original_size']} → {result['new_size']}")
    
    def generate_report(self) -> str:
        """
        Génère un rapport détaillé du prétraitement.
        
        Returns:
            Chemin du rapport
        """
        report_path = self.output_dir / 'preprocessing_report.json'
        
        # Calculer les statistiques finales
        if self.stats['aspect_ratios']:
            aspect_stats = {
                'min': min(self.stats['aspect_ratios']),
                'max': max(self.stats['aspect_ratios']),
                'mean': np.mean(self.stats['aspect_ratios']),
                'median': np.median(self.stats['aspect_ratios'])
            }
        else:
            aspect_stats = {}
        
        if self.stats['blur_scores']:
            blur_stats = {
                'min': min(self.stats['blur_scores']),
                'max': max(self.stats['blur_scores']),
                'mean': np.mean(self.stats['blur_scores']),
                'median': np.median(self.stats['blur_scores'])
            }
        else:
            blur_stats = {}
        
        # Créer le rapport complet
        report = {
            'summary': {
                'total_images': self.stats['total_images'],
                'processed': self.stats['processed'],
                'skipped_small': self.stats['skipped_small'],
                'skipped_blur': self.stats['skipped_blur'],
                'skipped_corrupt': self.stats['skipped_corrupt'],
                'resized': self.stats['resized'],
                'format_converted': self.stats['format_converted']
            },
            'size_distribution': self.stats['size_distribution'],
            'aspect_ratio_stats': aspect_stats,
            'blur_stats': blur_stats,
            'compression': {
                'original_size_mb': round(self.stats['original_total_size_mb'], 2),
                'processed_size_mb': round(self.stats['processed_total_size_mb'], 2),
                'reduction_percent': round(
                    (1 - self.stats['processed_total_size_mb'] / max(self.stats['original_total_size_mb'], 0.001)) * 100, 2
                ) if self.stats['original_total_size_mb'] > 0 else 0
            },
            'settings': {
                'max_size': self.max_size,
                'min_size': self.min_size,
                'target_format': self.target_format,
                'quality': self.quality,
                'blur_threshold': self.blur_threshold if self.check_blur else None
            }
        }
        
        # Sauvegarder le rapport
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Afficher le résumé
        self.print_summary(report)
        
        return str(report_path)
    
    def print_summary(self, report: Dict):
        """Affiche un résumé du traitement."""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DU PRÉTRAITEMENT DES IMAGES")
        print("="*60)
        
        summary = report['summary']
        print(f"Total d'images trouvées: {summary['total_images']}")
        print(f"Images traitées avec succès: {summary['processed']}")
        
        if summary['skipped_small'] > 0:
            print(f"⚠️ Images trop petites ignorées: {summary['skipped_small']}")
        if summary['skipped_blur'] > 0:
            print(f"⚠️ Images floues ignorées: {summary['skipped_blur']}")
        if summary['skipped_corrupt'] > 0:
            print(f"❌ Images corrompues ignorées: {summary['skipped_corrupt']}")
        
        print(f"\n📐 Images redimensionnées: {summary['resized']}")
        print(f"🔄 Formats convertis: {summary['format_converted']}")
        
        if report['size_distribution']:
            print("\n📊 Distribution des tailles:")
            for category, count in sorted(report['size_distribution'].items()):
                print(f"  {category}: {count} images")
        
        if report['aspect_ratio_stats']:
            stats = report['aspect_ratio_stats']
            print(f"\n📐 Ratios d'aspect:")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            print(f"  Moyenne: {stats['mean']:.2f}")
            print(f"  Médiane: {stats['median']:.2f}")
        
        compression = report['compression']
        print(f"\n💾 Compression:")
        print(f"  Taille originale: {compression['original_size_mb']:.1f} MB")
        print(f"  Taille après traitement: {compression['processed_size_mb']:.1f} MB")
        print(f"  Réduction: {compression['reduction_percent']:.1f}%")
        
        print("\n✅ Prétraitement terminé!")
        print(f"📁 Images traitées dans: {self.output_dir}")
        print("="*60)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Prétraite les images pour optimiser leur taille et qualité"
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Dossier source contenant les sous-dossiers de classes')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Dossier de sortie pour les images traitées')
    parser.add_argument('--max_size', type=int, default=1500,
                       help='Taille maximale en pixels (défaut: 1500)')
    parser.add_argument('--min_size', type=int, default=200,
                       help='Taille minimale acceptable (défaut: 200)')
    parser.add_argument('--format', type=str, default='JPEG',
                       choices=['JPEG', 'PNG'],
                       help='Format de sortie (défaut: JPEG)')
    parser.add_argument('--quality', type=int, default=95,
                       help='Qualité JPEG 1-100 (défaut: 95)')
    parser.add_argument('--check_blur', action='store_true',
                       help='Vérifier et filtrer les images floues')
    parser.add_argument('--blur_threshold', type=float, default=100.0,
                       help='Seuil de détection du flou (défaut: 100.0)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Nombre de threads parallèles (défaut: 4)')
    
    args = parser.parse_args()
    
    # Vérifier que le dossier d'entrée existe
    if not Path(args.input_dir).exists():
        logger.error(f"Le dossier d'entrée n'existe pas: {args.input_dir}")
        return 1
    
    # Créer le préprocesseur
    preprocessor = ImagePreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_size=args.max_size,
        min_size=args.min_size,
        target_format=args.format,
        quality=args.quality,
        check_blur=args.check_blur,
        blur_threshold=args.blur_threshold
    )
    
    # Traiter les images
    preprocessor.process_directory(num_workers=args.workers)
    
    # Générer le rapport
    report_path = preprocessor.generate_report()
    logger.info(f"Rapport sauvegardé: {report_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())