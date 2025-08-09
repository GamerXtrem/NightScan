#!/usr/bin/env python3
"""
Module de visualisation des résultats d'entraînement et d'évaluation.
Génère des graphiques interactifs et des rapports HTML.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import logging

# Configuration matplotlib et seaborn
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """
    Classe pour visualiser les résultats d'entraînement et d'évaluation.
    Génère des graphiques statiques et interactifs.
    """
    
    def __init__(self, output_dir: str, experiment_name: Optional[str] = None):
        """
        Initialise le visualiseur.
        
        Args:
            output_dir: Dossier de sortie pour les graphiques
            experiment_name: Nom de l'expérience
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        self.figures = []  # Pour le rapport HTML
        
    def plot_training_curves(self, history: Dict, save: bool = True,
                           show_confidence: bool = True) -> go.Figure:
        """
        Trace les courbes d'entraînement (loss et accuracy).
        
        Args:
            history: Historique des métriques
            save: Sauvegarder le graphique
            show_confidence: Afficher les intervalles de confiance
            
        Returns:
            Figure Plotly
        """
        # Créer une figure avec subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Accuracy', 'Learning Rate', 'F1-Score'),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        epochs = list(range(1, len(history['train']) + 1))
        
        # 1. Loss curves
        train_loss = [m.get('loss', 0) for m in history['train']]
        val_loss = [m.get('loss', 0) for m in history['val']] if history.get('val') else []
        
        fig.add_trace(
            go.Scatter(x=epochs, y=train_loss, name='Train Loss',
                      mode='lines+markers', line=dict(width=2)),
            row=1, col=1
        )
        
        if val_loss:
            fig.add_trace(
                go.Scatter(x=epochs, y=val_loss, name='Val Loss',
                          mode='lines+markers', line=dict(width=2)),
                row=1, col=1
            )
            
            # Marquer le minimum
            min_idx = np.argmin(val_loss)
            fig.add_trace(
                go.Scatter(x=[epochs[min_idx]], y=[val_loss[min_idx]],
                          name='Best Val Loss', mode='markers',
                          marker=dict(size=12, color='red', symbol='star')),
                row=1, col=1
            )
        
        # 2. Accuracy curves
        train_acc = [m.get('accuracy', 0) for m in history['train']]
        val_acc = [m.get('accuracy', 0) for m in history['val']] if history.get('val') else []
        
        fig.add_trace(
            go.Scatter(x=epochs, y=train_acc, name='Train Acc',
                      mode='lines+markers', line=dict(width=2)),
            row=1, col=2
        )
        
        if val_acc:
            fig.add_trace(
                go.Scatter(x=epochs, y=val_acc, name='Val Acc',
                          mode='lines+markers', line=dict(width=2)),
                row=1, col=2
            )
            
            # Marquer le maximum
            max_idx = np.argmax(val_acc)
            fig.add_trace(
                go.Scatter(x=[epochs[max_idx]], y=[val_acc[max_idx]],
                          name='Best Val Acc', mode='markers',
                          marker=dict(size=12, color='green', symbol='star')),
                row=1, col=2
            )
        
        # 3. Learning Rate (si disponible)
        if 'learning_rate' in history.get('train', [{}])[0]:
            lr = [m.get('learning_rate', 0) for m in history['train']]
            fig.add_trace(
                go.Scatter(x=epochs, y=lr, name='Learning Rate',
                          mode='lines', line=dict(width=2)),
                row=2, col=1
            )
        
        # 4. F1-Score
        train_f1 = [m.get('weighted', {}).get('f1_score', 0) for m in history['train']]
        val_f1 = [m.get('weighted', {}).get('f1_score', 0) for m in history['val']] if history.get('val') else []
        
        if train_f1 and any(train_f1):
            fig.add_trace(
                go.Scatter(x=epochs, y=train_f1, name='Train F1',
                          mode='lines+markers', line=dict(width=2)),
                row=2, col=2
            )
        
        if val_f1 and any(val_f1):
            fig.add_trace(
                go.Scatter(x=epochs, y=val_f1, name='Val F1',
                          mode='lines+markers', line=dict(width=2)),
                row=2, col=2
            )
        
        # Mise en forme
        fig.update_layout(
            title=f"Training Curves - {self.experiment_name}",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
        fig.update_yaxes(title_text="F1-Score (%)", row=2, col=2)
        
        if save:
            fig.write_html(self.plots_dir / 'training_curves.html')
            fig.write_image(self.plots_dir / 'training_curves.png')
        
        self.figures.append(('Training Curves', fig))
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                            normalize: bool = True, save: bool = True) -> go.Figure:
        """
        Trace la matrice de confusion interactive.
        
        Args:
            cm: Matrice de confusion
            class_names: Noms des classes
            normalize: Normaliser la matrice
            save: Sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
        else:
            cm_normalized = cm
        
        # Créer le texte pour les annotations
        text = []
        for i in range(len(cm)):
            row_text = []
            for j in range(len(cm)):
                if normalize:
                    row_text.append(f'{cm_normalized[i,j]:.1%}<br>({cm[i,j]})')
                else:
                    row_text.append(str(cm[i,j]))
            text.append(row_text)
        
        # Créer la heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized if normalize else cm,
            x=class_names,
            y=class_names,
            text=text,
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Rate' if normalize else 'Count')
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {self.experiment_name}",
            xaxis_title="Predicted",
            yaxis_title="True",
            width=max(800, len(class_names) * 60),
            height=max(600, len(class_names) * 60),
            xaxis=dict(tickangle=-45)
        )
        
        if save:
            fig.write_html(self.plots_dir / 'confusion_matrix.html')
            fig.write_image(self.plots_dir / 'confusion_matrix.png')
        
        self.figures.append(('Confusion Matrix', fig))
        return fig
    
    def plot_metrics_per_class(self, metrics_dict: Dict, save: bool = True) -> go.Figure:
        """
        Trace les métriques par classe.
        
        Args:
            metrics_dict: Dictionnaire des métriques
            save: Sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        if 'per_class' not in metrics_dict:
            logger.warning("Pas de métriques par classe disponibles")
            return None
        
        # Préparer les données
        class_names = []
        precision_values = []
        recall_values = []
        f1_values = []
        support_values = []
        
        for class_name, class_metrics in metrics_dict['per_class'].items():
            class_names.append(class_name)
            precision_values.append(class_metrics['precision'])
            recall_values.append(class_metrics['recall'])
            f1_values.append(class_metrics['f1_score'])
            support_values.append(class_metrics['support'])
        
        # Créer le graphique à barres groupées
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Precision',
            x=class_names,
            y=precision_values,
            text=[f'{v:.1f}%' for v in precision_values],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Recall',
            x=class_names,
            y=recall_values,
            text=[f'{v:.1f}%' for v in recall_values],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=class_names,
            y=f1_values,
            text=[f'{v:.1f}%' for v in f1_values],
            textposition='auto'
        ))
        
        # Ajouter une ligne pour le support
        fig.add_trace(go.Scatter(
            name='Support',
            x=class_names,
            y=support_values,
            yaxis='y2',
            mode='lines+markers',
            line=dict(width=2, color='red'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Metrics per Class - {self.experiment_name}",
            xaxis_title="Class",
            yaxis_title="Metric (%)",
            yaxis2=dict(
                title="Support (count)",
                overlaying='y',
                side='right'
            ),
            barmode='group',
            height=600,
            hovermode='x unified',
            xaxis=dict(tickangle=-45)
        )
        
        if save:
            fig.write_html(self.plots_dir / 'metrics_per_class.html')
            fig.write_image(self.plots_dir / 'metrics_per_class.png')
        
        self.figures.append(('Metrics per Class', fig))
        return fig
    
    def plot_roc_curves(self, fpr_dict: Dict, tpr_dict: Dict, 
                       auc_dict: Dict, save: bool = True) -> go.Figure:
        """
        Trace les courbes ROC multi-classes.
        
        Args:
            fpr_dict: False positive rates par classe
            tpr_dict: True positive rates par classe
            auc_dict: AUC scores par classe
            save: Sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        fig = go.Figure()
        
        # Courbe ROC pour chaque classe
        for class_name in fpr_dict.keys():
            fig.add_trace(go.Scatter(
                x=fpr_dict[class_name],
                y=tpr_dict[class_name],
                mode='lines',
                name=f'{class_name} (AUC = {auc_dict[class_name]:.3f})',
                line=dict(width=2)
            ))
        
        # Ligne diagonale (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title=f"ROC Curves - {self.experiment_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=800,
            height=600,
            hovermode='closest',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        if save:
            fig.write_html(self.plots_dir / 'roc_curves.html')
            fig.write_image(self.plots_dir / 'roc_curves.png')
        
        self.figures.append(('ROC Curves', fig))
        return fig
    
    def plot_error_analysis(self, errors_df: pd.DataFrame, 
                          top_n: int = 20, save: bool = True) -> go.Figure:
        """
        Analyse et visualise les erreurs de classification.
        
        Args:
            errors_df: DataFrame avec les erreurs
            top_n: Nombre d'erreurs à afficher
            save: Sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        if errors_df.empty:
            logger.info("Pas d'erreurs à analyser")
            return None
        
        # Créer des subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Error Types', 'Confidence Distribution',
                          'Error Rate by True Class', 'Error Rate by Predicted Class'),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}],
                  [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 1. Top types d'erreurs
        error_counts = errors_df['error_type'].value_counts().head(top_n)
        fig.add_trace(
            go.Bar(x=error_counts.values, y=error_counts.index,
                  orientation='h', name='Error Count'),
            row=1, col=1
        )
        
        # 2. Distribution de confiance (si disponible)
        if 'confidence_diff' in errors_df.columns:
            fig.add_trace(
                go.Histogram(x=errors_df['confidence_diff'],
                           name='Confidence Diff', nbinsx=30),
                row=1, col=2
            )
        
        # 3. Taux d'erreur par vraie classe
        true_class_errors = errors_df['true_class'].value_counts()
        fig.add_trace(
            go.Bar(x=true_class_errors.index, y=true_class_errors.values,
                  name='By True Class'),
            row=2, col=1
        )
        
        # 4. Taux d'erreur par classe prédite
        pred_class_errors = errors_df['pred_class'].value_counts()
        fig.add_trace(
            go.Bar(x=pred_class_errors.index, y=pred_class_errors.values,
                  name='By Predicted Class'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Error Analysis - {self.experiment_name}",
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Confidence Difference", row=1, col=2)
        fig.update_xaxes(title_text="True Class", row=2, col=1, tickangle=-45)
        fig.update_xaxes(title_text="Predicted Class", row=2, col=2, tickangle=-45)
        
        if save:
            fig.write_html(self.plots_dir / 'error_analysis.html')
            fig.write_image(self.plots_dir / 'error_analysis.png')
        
        self.figures.append(('Error Analysis', fig))
        return fig
    
    def plot_performance_summary(self, metrics: Dict, save: bool = True) -> go.Figure:
        """
        Crée un résumé visuel des performances.
        
        Args:
            metrics: Dictionnaire des métriques
            save: Sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        # Créer un tableau de métriques
        summary_data = []
        
        # Métriques principales
        summary_data.append(['Accuracy', f"{metrics.get('accuracy', 0):.2f}%"])
        summary_data.append(['Balanced Accuracy', f"{metrics.get('balanced_accuracy', 0):.2f}%"])
        
        if 'macro' in metrics:
            summary_data.append(['Macro Precision', f"{metrics['macro']['precision']:.2f}%"])
            summary_data.append(['Macro Recall', f"{metrics['macro']['recall']:.2f}%"])
            summary_data.append(['Macro F1-Score', f"{metrics['macro']['f1_score']:.2f}%"])
        
        if 'weighted' in metrics:
            summary_data.append(['Weighted Precision', f"{metrics['weighted']['precision']:.2f}%"])
            summary_data.append(['Weighted Recall', f"{metrics['weighted']['recall']:.2f}%"])
            summary_data.append(['Weighted F1-Score', f"{metrics['weighted']['f1_score']:.2f}%"])
        
        if 'cohen_kappa' in metrics:
            summary_data.append(['Cohen Kappa', f"{metrics['cohen_kappa']:.3f}"])
        
        if 'matthews_corrcoef' in metrics:
            summary_data.append(['Matthews Correlation', f"{metrics['matthews_corrcoef']:.3f}"])
        
        # Créer le tableau
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=list(zip(*summary_data)),
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title=f"Performance Summary - {self.experiment_name}",
            height=400
        )
        
        if save:
            fig.write_html(self.plots_dir / 'performance_summary.html')
            fig.write_image(self.plots_dir / 'performance_summary.png')
        
        self.figures.append(('Performance Summary', fig))
        return fig
    
    def create_html_report(self, metrics: Dict, history: Optional[Dict] = None,
                          title: Optional[str] = None) -> str:
        """
        Crée un rapport HTML complet avec tous les graphiques.
        
        Args:
            metrics: Métriques finales
            history: Historique d'entraînement
            title: Titre du rapport
            
        Returns:
            Chemin du fichier HTML généré
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title or f'Training Report - {self.experiment_name}'}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    padding: 20px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h2 {{
                    color: #666;
                    margin-top: 30px;
                    padding: 10px;
                    background-color: white;
                    border-left: 4px solid #4CAF50;
                }}
                .metric-card {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .timestamp {{
                    text-align: center;
                    color: #999;
                    margin: 20px 0;
                }}
                .plot-container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 10px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <h1>{title or f'Training Report - {self.experiment_name}'}</h1>
            <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        """
        
        # Ajouter le résumé des performances
        if metrics:
            html_content += """
            <div class="metric-card">
                <h2>Performance Summary</h2>
                <table style="width:100%; border-collapse: collapse;">
            """
            
            # Ajouter les métriques principales
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    html_content += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;"><b>{key.replace('_', ' ').title()}</b></td>
                        <td style="padding: 8px; text-align: right;">{value:.2f}</td>
                    </tr>
                    """
            
            html_content += """
                </table>
            </div>
            """
        
        # Ajouter tous les graphiques
        for title, fig in self.figures:
            html_content += f"""
            <div class="plot-container">
                <h2>{title}</h2>
                <div id="{title.replace(' ', '_').lower()}">
                    {fig.to_html(include_plotlyjs=False, div_id=title.replace(' ', '_').lower())}
                </div>
            </div>
            """
        
        # Ajouter les données JSON pour référence
        if metrics:
            html_content += f"""
            <div class="metric-card">
                <h2>Raw Metrics (JSON)</h2>
                <details>
                    <summary>Click to expand</summary>
                    <pre style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto;">
{json.dumps(metrics, indent=2)}
                    </pre>
                </details>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Sauvegarder le rapport
        report_path = self.output_dir / f'report_{self.experiment_name}.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Rapport HTML généré: {report_path}")
        return str(report_path)
    
    def plot_all(self, metrics: Dict, history: Dict, 
                confusion_matrix: Optional[np.ndarray] = None,
                errors_df: Optional[pd.DataFrame] = None,
                class_names: Optional[List[str]] = None) -> str:
        """
        Génère tous les graphiques et le rapport complet.
        
        Args:
            metrics: Métriques finales
            history: Historique d'entraînement
            confusion_matrix: Matrice de confusion
            errors_df: DataFrame des erreurs
            class_names: Noms des classes
            
        Returns:
            Chemin du rapport HTML
        """
        # Graphiques d'entraînement
        if history:
            self.plot_training_curves(history)
        
        # Métriques par classe
        if 'per_class' in metrics:
            self.plot_metrics_per_class(metrics)
        
        # Matrice de confusion
        if confusion_matrix is not None and class_names:
            self.plot_confusion_matrix(confusion_matrix, class_names)
        
        # Analyse des erreurs
        if errors_df is not None and not errors_df.empty:
            self.plot_error_analysis(errors_df)
        
        # Résumé des performances
        self.plot_performance_summary(metrics)
        
        # Générer le rapport HTML
        report_path = self.create_html_report(metrics, history)
        
        return report_path


def create_comparison_plot(experiments: Dict[str, Dict], 
                          metric_name: str = 'accuracy') -> go.Figure:
    """
    Compare plusieurs expériences.
    
    Args:
        experiments: Dict avec nom_exp -> métriques
        metric_name: Métrique à comparer
        
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    
    for exp_name, metrics in experiments.items():
        if metric_name in metrics:
            epochs = list(range(1, len(metrics[metric_name]) + 1))
            fig.add_trace(go.Scatter(
                x=epochs,
                y=metrics[metric_name],
                mode='lines+markers',
                name=exp_name
            ))
    
    fig.update_layout(
        title=f"Comparison - {metric_name}",
        xaxis_title="Epoch",
        yaxis_title=metric_name.replace('_', ' ').title(),
        hovermode='x unified'
    )
    
    return fig


if __name__ == "__main__":
    # Test du module
    print("Test du module de visualisation")
    
    # Créer des données de test
    np.random.seed(42)
    
    # Historique simulé
    history = {
        'train': [],
        'val': []
    }
    
    for epoch in range(20):
        train_metrics = {
            'loss': 2.0 * np.exp(-epoch/10) + np.random.rand() * 0.1,
            'accuracy': min(95, 60 + epoch * 2 + np.random.rand() * 5),
            'weighted': {
                'f1_score': min(93, 58 + epoch * 2 + np.random.rand() * 5)
            }
        }
        
        val_metrics = {
            'loss': 2.2 * np.exp(-epoch/10) + np.random.rand() * 0.15,
            'accuracy': min(90, 55 + epoch * 1.8 + np.random.rand() * 7),
            'weighted': {
                'f1_score': min(88, 53 + epoch * 1.8 + np.random.rand() * 7)
            }
        }
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
    
    # Métriques finales simulées
    final_metrics = {
        'accuracy': 89.5,
        'balanced_accuracy': 87.3,
        'cohen_kappa': 0.862,
        'matthews_corrcoef': 0.871,
        'macro': {
            'precision': 88.2,
            'recall': 87.9,
            'f1_score': 88.0
        },
        'weighted': {
            'precision': 89.1,
            'recall': 89.5,
            'f1_score': 89.3
        },
        'per_class': {
            f'Class_{i}': {
                'precision': 85 + np.random.rand() * 10,
                'recall': 83 + np.random.rand() * 12,
                'f1_score': 84 + np.random.rand() * 11,
                'support': np.random.randint(80, 120)
            }
            for i in range(5)
        }
    }
    
    # Matrice de confusion simulée
    n_classes = 5
    cm = np.random.randint(0, 20, (n_classes, n_classes))
    np.fill_diagonal(cm, np.random.randint(80, 100, n_classes))
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Créer le visualiseur
    visualizer = ResultsVisualizer('./test_output', 'test_experiment')
    
    # Générer tous les graphiques et le rapport
    report_path = visualizer.plot_all(
        metrics=final_metrics,
        history=history,
        confusion_matrix=cm,
        class_names=class_names
    )
    
    print(f"\n✅ Rapport généré: {report_path}")
    print("Tests terminés avec succès!")