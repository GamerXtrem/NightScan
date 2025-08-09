#!/usr/bin/env python3
"""
Script d'export du modèle dans différents formats.
Supporte PyTorch, TorchScript, ONNX, CoreML et TensorFlow Lite.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Classe pour exporter le modèle dans différents formats."""
    
    def __init__(self, checkpoint_path: str, output_dir: str):
        """
        Initialise l'exporteur.
        
        Args:
            checkpoint_path: Chemin vers le checkpoint du modèle
            output_dir: Dossier de sortie pour les modèles exportés
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device utilisé: {self.device}")
        
        # Charger le checkpoint
        self.checkpoint = self._load_checkpoint()
        
        # Créer le modèle
        self.model = self._create_model()
        
        # Informations du modèle
        self.model_info = self._extract_model_info()
        
        # Résultats d'export
        self.export_results = {}
        
    def _load_checkpoint(self) -> Dict:
        """Charge le checkpoint du modèle."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint non trouvé: {self.checkpoint_path}")
        
        logger.info(f"Chargement du checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        return checkpoint
    
    def _create_model(self) -> nn.Module:
        """Crée et charge le modèle."""
        # Importer le module de création du modèle
        from photo_model_dynamic import create_dynamic_model
        
        # Récupérer la configuration
        if 'model_config' in self.checkpoint:
            config = self.checkpoint['model_config']
        elif 'config' in self.checkpoint:
            config = self.checkpoint['config'].get('model', {})
        else:
            config = {'num_classes': 8, 'model_name': 'efficientnet-b1'}
        
        # Créer le modèle
        model = create_dynamic_model(
            num_classes=config.get('num_classes', 8),
            model_name=config.get('model_name'),
            pretrained=False,
            dropout_rate=0  # Pas de dropout pour l'export
        )
        
        # Charger les poids
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def _extract_model_info(self) -> Dict[str, Any]:
        """Extrait les informations du modèle."""
        info = {
            'original_checkpoint': str(self.checkpoint_path),
            'export_date': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        # Configuration du modèle
        if 'model_config' in self.checkpoint:
            info['model_config'] = self.checkpoint['model_config']
        
        # Métriques de performance
        if 'metrics' in self.checkpoint:
            info['performance'] = {
                'accuracy': self.checkpoint['metrics'].get('accuracy'),
                'loss': self.checkpoint['metrics'].get('loss')
            }
        
        # Informations sur le dataset
        if 'dataset_info' in self.checkpoint:
            info['dataset'] = self.checkpoint['dataset_info']
        
        # Classes
        if 'class_names' in self.checkpoint:
            info['classes'] = self.checkpoint['class_names']
        elif 'dataset_info' in self.checkpoint:
            info['classes'] = self.checkpoint['dataset_info'].get('classes', [])
        
        # Taille du modèle
        total_params = sum(p.numel() for p in self.model.parameters())
        info['model_size'] = {
            'parameters': total_params,
            'size_mb': (total_params * 4) / (1024 * 1024)  # Float32
        }
        
        return info
    
    def export_pytorch(self) -> str:
        """
        Exporte le modèle au format PyTorch standard.
        
        Returns:
            Chemin du fichier exporté
        """
        logger.info("Export PyTorch...")
        
        output_path = self.output_dir / 'model_pytorch.pth'
        
        # Sauvegarder le modèle complet avec toutes les infos
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_info': self.model_info,
            'export_type': 'pytorch',
            'version': '1.0.0'
        }, output_path)
        
        # Taille du fichier
        file_size = output_path.stat().st_size / (1024 * 1024)
        
        self.export_results['pytorch'] = {
            'path': str(output_path),
            'size_mb': file_size,
            'status': 'success'
        }
        
        logger.info(f"✅ PyTorch exporté: {output_path} ({file_size:.1f} MB)")
        return str(output_path)
    
    def export_torchscript(self, optimize: bool = True) -> str:
        """
        Exporte le modèle au format TorchScript.
        
        Args:
            optimize: Optimiser le modèle pour le mobile
            
        Returns:
            Chemin du fichier exporté
        """
        logger.info("Export TorchScript...")
        
        # Créer un exemple d'entrée
        example_input = torch.randn(1, 3, 224, 224)
        
        try:
            # Tracer le modèle
            self.model.cpu()  # TorchScript sur CPU
            traced_model = torch.jit.trace(self.model, example_input)
            
            if optimize:
                # Optimisations pour mobile
                from torch.utils.mobile_optimizer import optimize_for_mobile
                traced_model = optimize_for_mobile(traced_model)
            
            # Sauvegarder
            output_path = self.output_dir / 'model_torchscript.pt'
            traced_model.save(str(output_path))
            
            # Taille du fichier
            file_size = output_path.stat().st_size / (1024 * 1024)
            
            self.export_results['torchscript'] = {
                'path': str(output_path),
                'size_mb': file_size,
                'optimized': optimize,
                'status': 'success'
            }
            
            logger.info(f"✅ TorchScript exporté: {output_path} ({file_size:.1f} MB)")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export TorchScript: {e}")
            self.export_results['torchscript'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def export_onnx(self, opset_version: int = 11, 
                   dynamic_batch: bool = True) -> str:
        """
        Exporte le modèle au format ONNX.
        
        Args:
            opset_version: Version ONNX opset
            dynamic_batch: Permettre une taille de batch dynamique
            
        Returns:
            Chemin du fichier exporté
        """
        logger.info("Export ONNX...")
        
        try:
            import onnx
            import onnxruntime
            
            # Préparer l'entrée
            dummy_input = torch.randn(1, 3, 224, 224)
            output_path = self.output_dir / 'model.onnx'
            
            # Noms d'entrée/sortie
            input_names = ['input']
            output_names = ['output']
            
            # Axes dynamiques si demandé
            dynamic_axes = None
            if dynamic_batch:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            # Export
            self.model.cpu()
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            
            # Vérifier le modèle ONNX
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Test avec ONNX Runtime
            ort_session = onnxruntime.InferenceSession(str(output_path))
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Taille du fichier
            file_size = output_path.stat().st_size / (1024 * 1024)
            
            self.export_results['onnx'] = {
                'path': str(output_path),
                'size_mb': file_size,
                'opset_version': opset_version,
                'dynamic_batch': dynamic_batch,
                'status': 'success'
            }
            
            logger.info(f"✅ ONNX exporté: {output_path} ({file_size:.1f} MB)")
            return str(output_path)
            
        except ImportError:
            logger.error("onnx ou onnxruntime non installé. Installer avec: pip install onnx onnxruntime")
            self.export_results['onnx'] = {
                'status': 'failed',
                'error': 'Dependencies not installed'
            }
            return None
        except Exception as e:
            logger.error(f"Erreur lors de l'export ONNX: {e}")
            self.export_results['onnx'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def export_coreml(self) -> str:
        """
        Exporte le modèle au format CoreML pour iOS.
        
        Returns:
            Chemin du fichier exporté
        """
        logger.info("Export CoreML...")
        
        try:
            import coremltools as ct
            
            # D'abord exporter en TorchScript
            example_input = torch.randn(1, 3, 224, 224)
            self.model.cpu()
            traced_model = torch.jit.trace(self.model, example_input)
            
            # Convertir en CoreML
            model_coreml = ct.convert(
                traced_model,
                inputs=[ct.TensorType(
                    name="input",
                    shape=example_input.shape,
                    dtype=np.float32
                )],
                classifier_config=ct.ClassifierConfig(
                    class_labels=self.model_info.get('classes', list(range(8)))
                ) if self.model_info.get('classes') else None
            )
            
            # Ajouter les métadonnées
            model_coreml.author = "NightScan"
            model_coreml.short_description = "Wildlife classification model"
            model_coreml.version = "1.0.0"
            
            if self.model_info.get('performance'):
                acc = self.model_info['performance'].get('accuracy')
                if acc:
                    model_coreml.user_defined_metadata['accuracy'] = str(acc)
            
            # Sauvegarder
            output_path = self.output_dir / 'model.mlmodel'
            model_coreml.save(str(output_path))
            
            # Taille du fichier
            file_size = output_path.stat().st_size / (1024 * 1024)
            
            self.export_results['coreml'] = {
                'path': str(output_path),
                'size_mb': file_size,
                'status': 'success'
            }
            
            logger.info(f"✅ CoreML exporté: {output_path} ({file_size:.1f} MB)")
            return str(output_path)
            
        except ImportError:
            logger.error("coremltools non installé. Installer avec: pip install coremltools")
            self.export_results['coreml'] = {
                'status': 'failed',
                'error': 'Dependencies not installed'
            }
            return None
        except Exception as e:
            logger.error(f"Erreur lors de l'export CoreML: {e}")
            self.export_results['coreml'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def export_tflite(self, quantize: bool = False) -> str:
        """
        Exporte le modèle au format TensorFlow Lite.
        
        Args:
            quantize: Quantifier le modèle (INT8)
            
        Returns:
            Chemin du fichier exporté
        """
        logger.info("Export TensorFlow Lite...")
        
        try:
            # D'abord exporter en ONNX
            onnx_path = self.export_onnx(dynamic_batch=False)
            if not onnx_path:
                return None
            
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare
            
            # Charger le modèle ONNX
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            
            # Exporter en SavedModel TensorFlow
            tf_model_path = self.output_dir / 'tf_model'
            tf_rep.export_graph(str(tf_model_path))
            
            # Convertir en TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
            
            if quantize:
                # Quantification INT8
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = self._representative_dataset_gen
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                output_filename = 'model_quantized.tflite'
            else:
                output_filename = 'model.tflite'
            
            # Convertir
            tflite_model = converter.convert()
            
            # Sauvegarder
            output_path = self.output_dir / output_filename
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Taille du fichier
            file_size = output_path.stat().st_size / (1024 * 1024)
            
            self.export_results['tflite'] = {
                'path': str(output_path),
                'size_mb': file_size,
                'quantized': quantize,
                'status': 'success'
            }
            
            logger.info(f"✅ TFLite exporté: {output_path} ({file_size:.1f} MB)")
            return str(output_path)
            
        except ImportError as e:
            logger.error(f"Dépendances manquantes: {e}")
            logger.error("Installer avec: pip install tensorflow onnx-tf")
            self.export_results['tflite'] = {
                'status': 'failed',
                'error': 'Dependencies not installed'
            }
            return None
        except Exception as e:
            logger.error(f"Erreur lors de l'export TFLite: {e}")
            self.export_results['tflite'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def _representative_dataset_gen(self):
        """Générateur de données pour la quantification TFLite."""
        for _ in range(100):
            data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            yield [data]
    
    def export_quantized_pytorch(self) -> str:
        """
        Exporte le modèle PyTorch quantifié (INT8).
        
        Returns:
            Chemin du fichier exporté
        """
        logger.info("Export PyTorch quantifié...")
        
        try:
            # Quantification dynamique
            quantized_model = torch.quantization.quantize_dynamic(
                self.model.cpu(),
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            # Sauvegarder
            output_path = self.output_dir / 'model_quantized.pth'
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'model_info': self.model_info,
                'export_type': 'pytorch_quantized',
                'quantization': 'dynamic_int8'
            }, output_path)
            
            # Taille du fichier
            file_size = output_path.stat().st_size / (1024 * 1024)
            
            # Comparer avec le modèle original
            original_size = sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)
            compression_ratio = original_size / file_size
            
            self.export_results['pytorch_quantized'] = {
                'path': str(output_path),
                'size_mb': file_size,
                'original_size_mb': original_size,
                'compression_ratio': compression_ratio,
                'status': 'success'
            }
            
            logger.info(f"✅ PyTorch quantifié exporté: {output_path} ({file_size:.1f} MB)")
            logger.info(f"   Compression: {compression_ratio:.1f}x")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la quantification PyTorch: {e}")
            self.export_results['pytorch_quantized'] = {
                'status': 'failed',
                'error': str(e)
            }
            return None
    
    def export_all(self, formats: Optional[list] = None) -> Dict[str, Any]:
        """
        Exporte le modèle dans tous les formats demandés.
        
        Args:
            formats: Liste des formats ou None pour tous
            
        Returns:
            Dictionnaire avec les résultats d'export
        """
        if formats is None:
            formats = ['pytorch', 'torchscript', 'onnx', 'pytorch_quantized']
        
        logger.info(f"Export dans les formats: {formats}")
        
        for format_name in formats:
            if format_name == 'pytorch':
                self.export_pytorch()
            elif format_name == 'torchscript':
                self.export_torchscript()
            elif format_name == 'onnx':
                self.export_onnx()
            elif format_name == 'coreml':
                self.export_coreml()
            elif format_name == 'tflite':
                self.export_tflite()
            elif format_name == 'pytorch_quantized':
                self.export_quantized_pytorch()
            else:
                logger.warning(f"Format non supporté: {format_name}")
        
        # Sauvegarder les métadonnées
        self._save_export_metadata()
        
        return self.export_results
    
    def _save_export_metadata(self):
        """Sauvegarde les métadonnées d'export."""
        metadata = {
            'export_date': datetime.now().isoformat(),
            'source_checkpoint': str(self.checkpoint_path),
            'model_info': self.model_info,
            'export_results': self.export_results
        }
        
        metadata_path = self.output_dir / 'export_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Métadonnées sauvegardées: {metadata_path}")
    
    def generate_report(self) -> str:
        """
        Génère un rapport d'export.
        
        Returns:
            Chemin du rapport
        """
        report_path = self.output_dir / 'export_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("RAPPORT D'EXPORT DU MODÈLE\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint source: {self.checkpoint_path}\n\n")
            
            # Informations du modèle
            f.write("INFORMATIONS DU MODÈLE\n")
            f.write("-"*30 + "\n")
            if 'model_size' in self.model_info:
                f.write(f"Paramètres: {self.model_info['model_size']['parameters']:,}\n")
                f.write(f"Taille théorique: {self.model_info['model_size']['size_mb']:.1f} MB\n")
            
            if 'performance' in self.model_info:
                perf = self.model_info['performance']
                if perf.get('accuracy'):
                    f.write(f"Accuracy: {perf['accuracy']:.2f}%\n")
            f.write("\n")
            
            # Résultats d'export
            f.write("RÉSULTATS D'EXPORT\n")
            f.write("-"*30 + "\n")
            
            for format_name, result in self.export_results.items():
                f.write(f"\n{format_name.upper()}:\n")
                
                if result['status'] == 'success':
                    f.write(f"  ✅ Succès\n")
                    f.write(f"  Fichier: {result.get('path', 'N/A')}\n")
                    f.write(f"  Taille: {result.get('size_mb', 0):.1f} MB\n")
                    
                    if 'compression_ratio' in result:
                        f.write(f"  Compression: {result['compression_ratio']:.1f}x\n")
                    if 'quantized' in result:
                        f.write(f"  Quantifié: {'Oui' if result['quantized'] else 'Non'}\n")
                else:
                    f.write(f"  ❌ Échec\n")
                    f.write(f"  Erreur: {result.get('error', 'Unknown')}\n")
            
            # Résumé
            f.write("\n" + "="*60 + "\n")
            f.write("RÉSUMÉ\n")
            f.write("-"*30 + "\n")
            
            successful = sum(1 for r in self.export_results.values() if r['status'] == 'success')
            total = len(self.export_results)
            f.write(f"Formats exportés avec succès: {successful}/{total}\n")
            
            # Taille totale
            total_size = sum(r.get('size_mb', 0) for r in self.export_results.values() 
                           if r['status'] == 'success')
            f.write(f"Taille totale des exports: {total_size:.1f} MB\n")
        
        logger.info(f"Rapport généré: {report_path}")
        return str(report_path)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Exporter un modèle dans différents formats")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--output_dir', type=str, default='./exported_models',
                       help='Dossier de sortie pour les modèles exportés')
    parser.add_argument('--formats', type=str, nargs='+',
                       default=['pytorch', 'torchscript', 'onnx'],
                       choices=['pytorch', 'torchscript', 'onnx', 'coreml', 
                               'tflite', 'pytorch_quantized', 'all'],
                       help='Formats d\'export')
    
    args = parser.parse_args()
    
    # Si 'all' est dans les formats, utiliser tous les formats
    if 'all' in args.formats:
        formats = None
    else:
        formats = args.formats
    
    # Créer l'exporteur
    exporter = ModelExporter(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # Exporter dans tous les formats demandés
    results = exporter.export_all(formats)
    
    # Générer le rapport
    report_path = exporter.generate_report()
    
    # Afficher le résumé
    print("\n" + "="*60)
    print("EXPORT TERMINÉ")
    print("="*60)
    
    for format_name, result in results.items():
        if result['status'] == 'success':
            print(f"✅ {format_name}: {result.get('size_mb', 0):.1f} MB")
        else:
            print(f"❌ {format_name}: {result.get('error', 'Failed')}")
    
    print(f"\nRapport complet: {report_path}")
    print(f"Dossier de sortie: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()