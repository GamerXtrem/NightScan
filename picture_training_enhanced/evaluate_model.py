#!/usr/bin/env python3
"""
Script d'évaluation finale du modèle entraîné.
Teste le modèle sur l'ensemble de test et génère un rapport complet.
"""

import os
import sys
import argparse
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# Importer les modules locaux
from photo_dataset import PhotoDataset
from photo_model_dynamic import create_dynamic_model
from metrics import MetricsTracker, compute_confusion_matrix, analyze_errors
from visualize_results import ResultsVisualizer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Classe pour l'évaluation complète du modèle."""
    
    def __init__(self, checkpoint_path: str, test_dir: str, 
                 device: Optional[torch.device] = None):
        """
        Initialise l'évaluateur.
        
        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle
            test_dir: Dossier contenant les données de test
            device: Device à utiliser (auto-détection si None)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.test_dir = Path(test_dir)
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logger.info(f"Device utilisé: {self.device}")
        
        # Charger le checkpoint
        self.checkpoint = self._load_checkpoint()
        
        # Créer le modèle
        self.model = self._create_model()
        
        # Créer le dataset
        self.dataset = self._create_dataset()
        
        # Tracker de métriques
        self.metrics_tracker = MetricsTracker(
            num_classes=len(self.dataset.classes),
            class_names=self.dataset.classes
        )
        
        # Résultats
        self.results = {}
        
    def _load_checkpoint(self) -> Dict:
        """Charge le checkpoint du modèle."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint non trouvé: {self.checkpoint_path}")
        
        logger.info(f"Chargement du checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Afficher les infos du checkpoint
        if 'epoch' in checkpoint:
            logger.info(f"Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            val_acc = checkpoint['metrics'].get('accuracy', 'N/A')
            logger.info(f"Validation accuracy: {val_acc}")
        
        return checkpoint
    
    def _create_model(self) -> nn.Module:
        """Crée et charge le modèle."""
        # Récupérer la configuration du modèle
        if 'model_config' in self.checkpoint:
            config = self.checkpoint['model_config']
        elif 'config' in self.checkpoint:
            config = self.checkpoint['config'].get('model', {})
        else:
            # Configuration par défaut
            config = {'num_classes': 8, 'model_name': 'efficientnet-b1'}
        
        # Créer le modèle
        model = create_dynamic_model(
            num_classes=config.get('num_classes', 8),
            model_name=config.get('model_name'),
            pretrained=False,  # Pas besoin des poids pré-entraînés
            dropout_rate=0  # Pas de dropout en évaluation
        )
        
        # Charger les poids
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Modèle chargé: {total_params:,} paramètres")
        
        return model
    
    def _create_dataset(self) -> datasets.ImageFolder:
        """Crée le dataset de test."""
        # Récupérer les statistiques de normalisation
        if 'dataset_info' in self.checkpoint:
            normalization = self.checkpoint['dataset_info'].get('normalization', {})
            mean = normalization.get('mean', [0.485, 0.456, 0.406])
            std = normalization.get('std', [0.229, 0.224, 0.225])
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        
        # Créer les transformations (sans augmentation)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Créer le dataset
        dataset = datasets.ImageFolder(root=self.test_dir, transform=transform)
        
        logger.info(f"Dataset de test: {len(dataset)} images, {len(dataset.classes)} classes")
        logger.info(f"Classes: {dataset.classes}")
        
        return dataset
    
    def evaluate(self, batch_size: int = 32, num_workers: int = 4) -> Dict[str, Any]:
        """
        Évalue le modèle sur l'ensemble de test.
        
        Args:
            batch_size: Taille du batch
            num_workers: Nombre de workers pour le DataLoader
            
        Returns:
            Dictionnaire avec tous les résultats
        """
        logger.info("Début de l'évaluation...")
        
        # Créer le DataLoader
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Variables pour collecter les résultats
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_image_paths = []
        
        # Temps d'inférence
        inference_times = []
        
        # Évaluation
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Évaluation")):
                images = images.to(self.device)
                
                # Mesurer le temps d'inférence
                start_time = time.time()
                outputs = self.model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / len(images))
                
                # Probabilités et prédictions
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # Collecter les résultats
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Obtenir les chemins des images
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(self.dataset))
                for i in range(batch_start, batch_end):
                    img_path, _ = self.dataset.samples[i]
                    all_image_paths.append(img_path)
        
        # Convertir en arrays numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculer les métriques
        logger.info("Calcul des métriques...")
        metrics = self.metrics_tracker.compute_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        # Ajouter les statistiques de temps
        metrics['inference'] = {
            'mean_time_ms': np.mean(inference_times) * 1000,
            'std_time_ms': np.std(inference_times) * 1000,
            'min_time_ms': np.min(inference_times) * 1000,
            'max_time_ms': np.max(inference_times) * 1000,
            'total_images': len(all_labels),
            'images_per_second': 1.0 / np.mean(inference_times)
        }
        
        # Analyser les erreurs
        errors_df = analyze_errors(
            all_labels, all_predictions, all_probabilities,
            class_names=self.dataset.classes
        )
        
        # Ajouter les chemins des images aux erreurs
        if not errors_df.empty:
            error_indices = errors_df['index'].values
            errors_df['image_path'] = [all_image_paths[i] for i in error_indices]
        
        # Stocker les résultats
        self.results = {
            'metrics': metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'image_paths': all_image_paths,
            'errors': errors_df,
            'confusion_matrix': compute_confusion_matrix(all_labels, all_predictions)
        }
        
        return self.results
    
    def test_robustness(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Teste la robustesse du modèle avec des perturbations.
        
        Args:
            num_samples: Nombre d'échantillons à tester
            
        Returns:
            Dictionnaire avec les résultats de robustesse
        """
        logger.info("Test de robustesse...")
        
        # Créer différentes transformations perturbées
        perturbations = {
            'normal': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]),
            'blur': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.GaussianBlur(kernel_size=5),
                transforms.ToTensor()
            ]),
            'noise': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)
            ]),
            'rotation': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomRotation(degrees=30),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]),
            'brightness': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ColorJitter(brightness=0.5),
                transforms.ToTensor()
            ])
        }
        
        robustness_results = {}
        
        # Échantillonner des images
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for perturbation_name, transform in perturbations.items():
            correct = 0
            
            for idx in indices:
                # Charger l'image originale
                img_path, true_label = self.dataset.samples[idx]
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                
                # Appliquer la perturbation
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                
                # Normaliser
                mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(self.device)
                img_tensor = (img_tensor - mean) / std
                
                # Prédiction
                with torch.no_grad():
                    output = self.model(img_tensor)
                    _, pred = torch.max(output, 1)
                    
                    if pred.item() == true_label:
                        correct += 1
            
            accuracy = (correct / len(indices)) * 100
            robustness_results[perturbation_name] = accuracy
        
        return robustness_results
    
    def generate_report(self, output_dir: str, include_visualizations: bool = True) -> str:
        """
        Génère un rapport complet d'évaluation.
        
        Args:
            output_dir: Dossier de sortie pour le rapport
            include_visualizations: Inclure les graphiques
            
        Returns:
            Chemin du rapport généré
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer le rapport texte
        report_path = output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("RAPPORT D'ÉVALUATION DU MODÈLE\n")
            f.write("="*60 + "\n\n")
            
            # Informations sur le modèle
            f.write("MODÈLE\n")
            f.write("-"*30 + "\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Nombre de classes: {len(self.dataset.classes)}\n")
            f.write(f"Classes: {', '.join(self.dataset.classes)}\n\n")
            
            # Métriques globales
            if 'metrics' in self.results:
                metrics = self.results['metrics']
                
                f.write("MÉTRIQUES GLOBALES\n")
                f.write("-"*30 + "\n")
                f.write(f"Accuracy: {metrics.get('accuracy', 0):.2f}%\n")
                f.write(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.2f}%\n")
                f.write(f"Cohen's Kappa: {metrics.get('cohen_kappa', 0):.3f}\n")
                f.write(f"Matthews Correlation: {metrics.get('matthews_corrcoef', 0):.3f}\n\n")
                
                # Métriques macro
                if 'macro' in metrics:
                    f.write("MÉTRIQUES MACRO (moyenne simple)\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Precision: {metrics['macro']['precision']:.2f}%\n")
                    f.write(f"Recall: {metrics['macro']['recall']:.2f}%\n")
                    f.write(f"F1-Score: {metrics['macro']['f1_score']:.2f}%\n\n")
                
                # Métriques weighted
                if 'weighted' in metrics:
                    f.write("MÉTRIQUES WEIGHTED (pondérées)\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Precision: {metrics['weighted']['precision']:.2f}%\n")
                    f.write(f"Recall: {metrics['weighted']['recall']:.2f}%\n")
                    f.write(f"F1-Score: {metrics['weighted']['f1_score']:.2f}%\n\n")
                
                # Top-K accuracy
                if 'top_k_accuracy' in metrics:
                    f.write("TOP-K ACCURACY\n")
                    f.write("-"*30 + "\n")
                    for k, acc in metrics['top_k_accuracy'].items():
                        f.write(f"{k}: {acc:.2f}%\n")
                    f.write("\n")
                
                # Temps d'inférence
                if 'inference' in metrics:
                    inf = metrics['inference']
                    f.write("PERFORMANCE D'INFÉRENCE\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Temps moyen: {inf['mean_time_ms']:.2f} ms/image\n")
                    f.write(f"Écart-type: {inf['std_time_ms']:.2f} ms\n")
                    f.write(f"Min: {inf['min_time_ms']:.2f} ms\n")
                    f.write(f"Max: {inf['max_time_ms']:.2f} ms\n")
                    f.write(f"Images/seconde: {inf['images_per_second']:.1f}\n\n")
                
                # Métriques par classe
                if 'per_class' in metrics:
                    f.write("MÉTRIQUES PAR CLASSE\n")
                    f.write("-"*30 + "\n")
                    
                    # Créer un tableau
                    f.write(f"{'Classe':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
                    f.write("-"*70 + "\n")
                    
                    for class_name, class_metrics in metrics['per_class'].items():
                        f.write(f"{class_name:<20} "
                               f"{class_metrics['precision']:>11.1f}% "
                               f"{class_metrics['recall']:>11.1f}% "
                               f"{class_metrics['f1_score']:>11.1f}% "
                               f"{class_metrics['support']:>10}\n")
                    f.write("\n")
            
            # Analyse des erreurs
            if 'errors' in self.results and not self.results['errors'].empty:
                errors_df = self.results['errors']
                
                f.write("ANALYSE DES ERREURS\n")
                f.write("-"*30 + "\n")
                f.write(f"Nombre total d'erreurs: {len(errors_df)}\n")
                f.write(f"Taux d'erreur: {len(errors_df)/len(self.results['labels'])*100:.2f}%\n\n")
                
                # Top erreurs
                f.write("Top 10 types d'erreurs:\n")
                error_counts = errors_df['error_type'].value_counts().head(10)
                for error_type, count in error_counts.items():
                    f.write(f"  {error_type}: {count} fois\n")
                f.write("\n")
                
                # Erreurs par classe
                f.write("Erreurs par vraie classe:\n")
                true_class_errors = errors_df['true_class'].value_counts()
                for class_name, count in true_class_errors.items():
                    f.write(f"  {class_name}: {count} erreurs\n")
        
        logger.info(f"Rapport texte sauvegardé: {report_path}")
        
        # Sauvegarder les métriques en JSON
        json_path = output_dir / 'evaluation_metrics.json'
        with open(json_path, 'w') as f:
            # Convertir les arrays numpy en listes pour JSON
            metrics_json = self.results['metrics'].copy()
            if 'confusion_matrix' in metrics_json:
                metrics_json['confusion_matrix'] = self.results['confusion_matrix'].tolist()
            
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Métriques JSON sauvegardées: {json_path}")
        
        # Sauvegarder les erreurs en CSV
        if 'errors' in self.results and not self.results['errors'].empty:
            errors_path = output_dir / 'errors_analysis.csv'
            self.results['errors'].to_csv(errors_path, index=False)
            logger.info(f"Analyse des erreurs sauvegardée: {errors_path}")
        
        # Générer les visualisations
        if include_visualizations:
            visualizer = ResultsVisualizer(output_dir, 'evaluation')
            
            # Matrice de confusion
            if 'confusion_matrix' in self.results:
                visualizer.plot_confusion_matrix(
                    self.results['confusion_matrix'],
                    self.dataset.classes
                )
            
            # Métriques par classe
            if 'metrics' in self.results:
                visualizer.plot_metrics_per_class(self.results['metrics'])
                visualizer.plot_performance_summary(self.results['metrics'])
            
            # Analyse des erreurs
            if 'errors' in self.results and not self.results['errors'].empty:
                visualizer.plot_error_analysis(self.results['errors'])
            
            # Générer le rapport HTML
            html_path = visualizer.create_html_report(
                self.results.get('metrics', {}),
                title="Rapport d'Évaluation du Modèle"
            )
            
            logger.info(f"Rapport HTML généré: {html_path}")
            
            return html_path
        
        return str(report_path)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Évaluer un modèle entraîné")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Dossier contenant les données de test')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Dossier de sortie pour les résultats')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Taille du batch')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Nombre de workers pour le DataLoader')
    parser.add_argument('--test_robustness', action='store_true',
                       help='Tester la robustesse du modèle')
    parser.add_argument('--no_visualizations', action='store_true',
                       help='Ne pas générer les visualisations')
    
    args = parser.parse_args()
    
    # Créer l'évaluateur
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir
    )
    
    # Évaluer le modèle
    results = evaluator.evaluate(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Afficher les résultats principaux
    print("\n" + "="*60)
    print("RÉSULTATS D'ÉVALUATION")
    print("="*60)
    print(f"Accuracy: {results['metrics']['accuracy']:.2f}%")
    print(f"Balanced Accuracy: {results['metrics']['balanced_accuracy']:.2f}%")
    
    if 'macro' in results['metrics']:
        print(f"Macro F1-Score: {results['metrics']['macro']['f1_score']:.2f}%")
    
    if 'weighted' in results['metrics']:
        print(f"Weighted F1-Score: {results['metrics']['weighted']['f1_score']:.2f}%")
    
    if 'inference' in results['metrics']:
        print(f"Vitesse: {results['metrics']['inference']['images_per_second']:.1f} images/sec")
    
    # Test de robustesse optionnel
    if args.test_robustness:
        print("\n" + "="*60)
        print("TEST DE ROBUSTESSE")
        print("="*60)
        
        robustness = evaluator.test_robustness()
        for perturbation, accuracy in robustness.items():
            print(f"{perturbation:<15}: {accuracy:.2f}%")
    
    # Générer le rapport
    report_path = evaluator.generate_report(
        output_dir=args.output_dir,
        include_visualizations=not args.no_visualizations
    )
    
    print("\n" + "="*60)
    print(f"✅ Évaluation terminée!")
    print(f"Rapport sauvegardé: {report_path}")
    print("="*60)


if __name__ == "__main__":
    main()