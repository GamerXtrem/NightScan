#!/usr/bin/env python3
"""
Module de métriques détaillées pour l'entraînement et l'évaluation des modèles.
Fournit des métriques complètes par classe et agrégées.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef,
    balanced_accuracy_score, top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
from scipy import stats
import json
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracker complet pour les métriques d'entraînement et d'évaluation.
    Supporte le calcul incrémental et l'historique.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialise le tracker de métriques.
        
        Args:
            num_classes: Nombre de classes
            class_names: Noms des classes (optionnel)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Historique des métriques par epoch
        self.history = {
            'train': [],
            'val': [],
            'test': None
        }
        
        # Buffers pour calcul incrémental
        self.reset()
        
    def reset(self):
        """Réinitialise les buffers pour une nouvelle epoch."""
        self.y_true_buffer = []
        self.y_pred_buffer = []
        self.y_proba_buffer = []
        self.loss_buffer = []
        
    def update_batch(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    y_proba: Optional[np.ndarray] = None, loss: Optional[float] = None):
        """
        Met à jour les buffers avec un nouveau batch.
        
        Args:
            y_true: Labels vrais
            y_pred: Prédictions
            y_proba: Probabilités (optionnel)
            loss: Loss du batch (optionnel)
        """
        self.y_true_buffer.extend(y_true.tolist() if isinstance(y_true, np.ndarray) else y_true)
        self.y_pred_buffer.extend(y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred)
        
        if y_proba is not None:
            self.y_proba_buffer.extend(y_proba.tolist() if isinstance(y_proba, np.ndarray) else y_proba)
        
        if loss is not None:
            self.loss_buffer.append(loss)
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Calcule toutes les métriques.
        
        Args:
            y_true: Labels vrais
            y_pred: Prédictions
            y_proba: Probabilités pour ROC/AUC (optionnel)
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {}
        
        # Métriques de base
        metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred) * 100
        
        # Métriques par classe
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': precision[i] * 100,
                'recall': recall[i] * 100,
                'f1_score': f1[i] * 100,
                'support': int(support[i])
            }
        
        # Métriques agrégées
        metrics['macro'] = {
            'precision': np.mean(precision) * 100,
            'recall': np.mean(recall) * 100,
            'f1_score': np.mean(f1) * 100
        }
        
        # Weighted average
        total_support = np.sum(support)
        metrics['weighted'] = {
            'precision': np.sum(precision * support) / total_support * 100,
            'recall': np.sum(recall * support) / total_support * 100,
            'f1_score': np.sum(f1 * support) / total_support * 100
        }
        
        # Métriques avancées
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Matrice de confusion
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Top-K accuracy si probabilités disponibles
        if y_proba is not None:
            metrics['top_k_accuracy'] = self._compute_top_k_accuracy(y_true, y_proba)
            metrics['roc_auc'] = self._compute_roc_auc(y_true, y_proba)
        
        return metrics
    
    def _compute_top_k_accuracy(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[int, float]:
        """
        Calcule la Top-K accuracy.
        
        Args:
            y_true: Labels vrais
            y_proba: Probabilités prédites
            
        Returns:
            Dictionnaire avec Top-K accuracies
        """
        top_k = {}
        
        for k in [1, 3, 5]:
            if k <= self.num_classes:
                # Obtenir les K meilleures prédictions
                top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
                
                # Vérifier si le vrai label est dans le top-K
                correct = 0
                for i, true_label in enumerate(y_true):
                    if true_label in top_k_preds[i]:
                        correct += 1
                
                top_k[f'top_{k}'] = (correct / len(y_true)) * 100
        
        return top_k
    
    def _compute_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, any]:
        """
        Calcule ROC AUC pour multi-classes.
        
        Args:
            y_true: Labels vrais
            y_proba: Probabilités prédites
            
        Returns:
            Dictionnaire avec scores AUC
        """
        roc_auc = {}
        
        # Binariser les labels pour one-vs-rest
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        # AUC par classe
        for i, class_name in enumerate(self.class_names):
            if np.sum(y_true_bin[:, i]) > 0:  # Vérifier qu'il y a des échantillons
                try:
                    auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                    roc_auc[class_name] = auc_score
                except:
                    roc_auc[class_name] = None
        
        # Macro et Weighted AUC
        try:
            roc_auc['macro'] = roc_auc_score(y_true_bin, y_proba, average='macro')
            roc_auc['weighted'] = roc_auc_score(y_true_bin, y_proba, average='weighted')
        except:
            roc_auc['macro'] = None
            roc_auc['weighted'] = None
        
        return roc_auc
    
    def compute_epoch_metrics(self, split: str = 'train') -> Dict[str, any]:
        """
        Calcule les métriques pour l'epoch actuelle depuis les buffers.
        
        Args:
            split: Type de split ('train', 'val', 'test')
            
        Returns:
            Dictionnaire avec les métriques
        """
        if not self.y_true_buffer:
            return {}
        
        y_true = np.array(self.y_true_buffer)
        y_pred = np.array(self.y_pred_buffer)
        y_proba = np.array(self.y_proba_buffer) if self.y_proba_buffer else None
        
        metrics = self.compute_metrics(y_true, y_pred, y_proba)
        
        # Ajouter la loss moyenne si disponible
        if self.loss_buffer:
            metrics['loss'] = np.mean(self.loss_buffer)
        
        # Sauvegarder dans l'historique
        if split in ['train', 'val']:
            self.history[split].append(metrics)
        elif split == 'test':
            self.history['test'] = metrics
        
        return metrics
    
    def get_summary(self, split: Optional[str] = None) -> Dict[str, any]:
        """
        Obtient un résumé des métriques.
        
        Args:
            split: Split spécifique ou None pour tout
            
        Returns:
            Résumé des métriques
        """
        if split:
            if split == 'test':
                return self.history['test'] or {}
            else:
                return self.history[split][-1] if self.history[split] else {}
        
        # Résumé complet
        summary = {
            'num_epochs': len(self.history['train']),
            'best_val_accuracy': 0,
            'best_epoch': 0,
            'final_metrics': {}
        }
        
        if self.history['val']:
            val_accuracies = [m['accuracy'] for m in self.history['val']]
            best_idx = np.argmax(val_accuracies)
            summary['best_val_accuracy'] = val_accuracies[best_idx]
            summary['best_epoch'] = best_idx + 1
            summary['best_metrics'] = self.history['val'][best_idx]
        
        if self.history['train']:
            summary['final_metrics']['train'] = self.history['train'][-1]
        
        if self.history['val']:
            summary['final_metrics']['val'] = self.history['val'][-1]
        
        if self.history['test']:
            summary['final_metrics']['test'] = self.history['test']
        
        return summary
    
    def save_metrics(self, filepath: str):
        """
        Sauvegarde les métriques dans un fichier JSON.
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        output = {
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'history': self.history,
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Métriques sauvegardées dans {filepath}")
    
    def load_metrics(self, filepath: str):
        """
        Charge les métriques depuis un fichier JSON.
        
        Args:
            filepath: Chemin du fichier
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.class_names = data['class_names']
        self.num_classes = data['num_classes']
        self.history = data['history']
        
        logger.info(f"Métriques chargées depuis {filepath}")
    
    def to_dataframe(self, split: str = 'val') -> pd.DataFrame:
        """
        Convertit l'historique en DataFrame pandas.
        
        Args:
            split: Split à convertir
            
        Returns:
            DataFrame avec les métriques
        """
        if not self.history[split]:
            return pd.DataFrame()
        
        # Extraire les métriques principales
        data = []
        for epoch, metrics in enumerate(self.history[split]):
            row = {
                'epoch': epoch + 1,
                'accuracy': metrics.get('accuracy', 0),
                'loss': metrics.get('loss', 0),
                'macro_f1': metrics['macro']['f1_score'] if 'macro' in metrics else 0,
                'weighted_f1': metrics['weighted']['f1_score'] if 'weighted' in metrics else 0
            }
            
            # Ajouter les métriques par classe
            if 'per_class' in metrics:
                for class_name, class_metrics in metrics['per_class'].items():
                    row[f'{class_name}_f1'] = class_metrics['f1_score']
            
            data.append(row)
        
        return pd.DataFrame(data)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: Optional[str] = None) -> np.ndarray:
    """
    Calcule la matrice de confusion.
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        normalize: Type de normalisation ('true', 'pred', 'all', None)
        
    Returns:
        Matrice de confusion
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize == 'true':
        # Normaliser par ligne (vrais labels)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif normalize == 'pred':
        # Normaliser par colonne (prédictions)
        cm = cm.astype('float') / cm.sum(axis=0)
    elif normalize == 'all':
        # Normaliser par le total
        cm = cm.astype('float') / cm.sum()
    
    return cm


def compute_class_weights(y_train: np.ndarray, method: str = 'inverse') -> np.ndarray:
    """
    Calcule les poids des classes pour gérer le déséquilibre.
    
    Args:
        y_train: Labels d'entraînement
        method: Méthode de calcul ('inverse', 'effective', 'balanced')
        
    Returns:
        Array avec les poids par classe
    """
    unique, counts = np.unique(y_train, return_counts=True)
    n_classes = len(unique)
    n_samples = len(y_train)
    
    if method == 'inverse':
        # Inverse de la fréquence
        weights = n_samples / (n_classes * counts)
    elif method == 'effective':
        # Effective number of samples
        beta = 0.999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    elif method == 'balanced':
        # Sklearn balanced
        weights = n_samples / (counts * n_classes)
    else:
        weights = np.ones(n_classes)
    
    # Normaliser pour que la somme soit égale au nombre de classes
    weights = weights * n_classes / weights.sum()
    
    return weights


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray,
                                 metric_func: callable,
                                 n_bootstraps: int = 1000,
                                 confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calcule l'intervalle de confiance par bootstrap.
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        metric_func: Fonction de métrique
        n_bootstraps: Nombre d'échantillons bootstrap
        confidence_level: Niveau de confiance
        
    Returns:
        (métrique, borne_inf, borne_sup)
    """
    n_samples = len(y_true)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        # Échantillonnage avec remplacement
        indices = np.random.randint(0, n_samples, size=n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculer la métrique
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrapped_scores.append(score)
    
    # Calculer l'intervalle de confiance
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrapped_scores, lower_percentile)
    upper_bound = np.percentile(bootstrapped_scores, upper_percentile)
    mean_score = np.mean(bootstrapped_scores)
    
    return mean_score, lower_bound, upper_bound


def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: Optional[np.ndarray] = None,
                   class_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyse détaillée des erreurs de classification.
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        y_proba: Probabilités (optionnel)
        class_names: Noms des classes
        
    Returns:
        DataFrame avec l'analyse des erreurs
    """
    errors_data = []
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            error_info = {
                'index': i,
                'true_label': true,
                'pred_label': pred,
                'true_class': class_names[true] if class_names else f"Class_{true}",
                'pred_class': class_names[pred] if class_names else f"Class_{pred}"
            }
            
            if y_proba is not None:
                error_info['true_prob'] = y_proba[i, true]
                error_info['pred_prob'] = y_proba[i, pred]
                error_info['confidence_diff'] = y_proba[i, pred] - y_proba[i, true]
            
            errors_data.append(error_info)
    
    errors_df = pd.DataFrame(errors_data)
    
    # Ajouter des statistiques
    if not errors_df.empty:
        errors_df['error_type'] = errors_df.apply(
            lambda x: f"{x['true_class']} -> {x['pred_class']}", axis=1
        )
    
    return errors_df


if __name__ == "__main__":
    # Test du module
    print("Test du module de métriques")
    
    # Créer des données de test
    np.random.seed(42)
    num_classes = 5
    n_samples = 100
    
    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred = y_true.copy()
    # Ajouter quelques erreurs
    errors_idx = np.random.choice(n_samples, 20, replace=False)
    y_pred[errors_idx] = np.random.randint(0, num_classes, 20)
    
    # Créer des probabilités simulées
    y_proba = np.random.rand(n_samples, num_classes)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Tester le tracker
    tracker = MetricsTracker(num_classes=num_classes)
    
    # Simuler plusieurs batches
    batch_size = 10
    for i in range(0, n_samples, batch_size):
        batch_true = y_true[i:i+batch_size]
        batch_pred = y_pred[i:i+batch_size]
        batch_proba = y_proba[i:i+batch_size]
        
        tracker.update_batch(batch_true, batch_pred, batch_proba, loss=np.random.rand())
    
    # Calculer les métriques
    metrics = tracker.compute_epoch_metrics('train')
    
    print("\nMétriques calculées:")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2f}%")
    print(f"Cohen Kappa: {metrics['cohen_kappa']:.3f}")
    print(f"Matthews Correlation: {metrics['matthews_corrcoef']:.3f}")
    
    print("\nMétriques par classe:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}: P={class_metrics['precision']:.1f}%, "
              f"R={class_metrics['recall']:.1f}%, F1={class_metrics['f1_score']:.1f}%")
    
    # Tester l'analyse d'erreurs
    errors_df = analyze_errors(y_true, y_pred, y_proba)
    print(f"\nNombre d'erreurs: {len(errors_df)}")
    if not errors_df.empty:
        print("Top erreurs:")
        print(errors_df['error_type'].value_counts().head())
    
    print("\n✅ Tests terminés avec succès!")