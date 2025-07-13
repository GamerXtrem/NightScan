#!/usr/bin/env python3
"""
Pipeline de Quantification des Modèles NightScan pour Mobile

Ce script convertit les modèles EfficientNet complets en versions légères
optimisées pour l'inférence mobile (TensorFlow Lite, Core ML).
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# TensorFlow imports for conversion
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Force CPU usage for conversion
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - TensorFlow Lite conversion will be skipped")

# Core ML imports (macOS only)
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("⚠️ Core ML Tools not available - Core ML conversion will be skipped")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class ModelQuantizationPipeline:
    """Pipeline de quantification pour les modèles NightScan."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialise le pipeline de quantification.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(self.config['output_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistiques de conversion
        self.conversion_stats = {
            'models_processed': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_size_reduction': 0,
            'accuracy_retention': []
        }
        
        logger.info(f"Pipeline de quantification initialisé - Device: {self.device}")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Charge la configuration de quantification."""
        default_config = {
            'output_directory': 'mobile_models',
            'audio_model': {
                'input_path': 'audio_training_efficientnet/models/best_model.pth',
                'model_name': 'efficientnet-b1',
                'num_classes': 6,
                'class_names': ['bird_song', 'mammal_call', 'insect_sound', 
                              'amphibian_call', 'environmental_sound', 'unknown_species'],
                'input_size': (128, 128),  # Spectrogramme mel dimensions
                'quantization_mode': 'dynamic'
            },
            'photo_model': {
                'input_path': 'picture_training_enhanced/models/best_model.pth',
                'model_name': 'efficientnet-b1',
                'num_classes': 8,
                'class_names': ['bat', 'owl', 'raccoon', 'opossum', 'deer', 'fox', 'coyote', 'unknown'],
                'input_size': (224, 224),
                'quantization_mode': 'int8'
            },
            'optimization': {
                'target_accuracy_loss': 0.05,  # Max 5% accuracy loss
                'target_size_reduction': 0.75,  # Target 75% size reduction
                'calibration_samples': 100
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    default_config.update(user_config)
                    logger.info(f"Configuration chargée depuis {config_path}")
            except Exception as e:
                logger.warning(f"Erreur lecture config: {e}. Utilisation config par défaut.")
        
        return default_config
    
    def quantize_all_models(self) -> Dict:
        """Quantifie tous les modèles configurés."""
        logger.info("🚀 Démarrage de la quantification des modèles")
        
        results = {}
        
        # Quantifier le modèle audio
        if self._model_exists('audio_model'):
            logger.info("🎵 Quantification du modèle audio...")
            results['audio'] = self.quantize_model('audio')
        else:
            logger.warning("❌ Modèle audio introuvable")
            results['audio'] = {'success': False, 'error': 'Model not found'}
        
        # Quantifier le modèle photo
        if self._model_exists('photo_model'):
            logger.info("📸 Quantification du modèle photo...")
            results['photo'] = self.quantize_model('photo')
        else:
            logger.warning("❌ Modèle photo introuvable")
            results['photo'] = {'success': False, 'error': 'Model not found'}
        
        # Générer le rapport final
        self._generate_report(results)
        
        return results
    
    def _model_exists(self, model_type: str) -> bool:
        """Vérifie si le modèle existe."""
        model_path = Path(self.config[model_type]['input_path'])
        return model_path.exists()
    
    def quantize_model(self, model_type: str) -> Dict:
        """
        Quantifie un modèle spécifique.
        
        Args:
            model_type: 'audio' ou 'photo'
            
        Returns:
            Dict: Résultats de la quantification
        """
        try:
            config = self.config[f'{model_type}_model']
            model_path = Path(config['input_path'])
            
            logger.info(f"📦 Chargement du modèle {model_type}: {model_path}")
            
            # Charger le modèle PyTorch
            model = self._load_pytorch_model(model_path, config)
            
            # Convertir en ONNX d'abord
            onnx_path = self._convert_to_onnx(model, config, model_type)
            
            results = {
                'success': True,
                'model_type': model_type,
                'original_size': model_path.stat().st_size,
                'conversions': {},
                'metadata': {
                    'model_name': config['model_name'],
                    'num_classes': config['num_classes'],
                    'class_names': config['class_names'],
                    'input_size': config['input_size'],
                    'quantization_mode': config['quantization_mode']
                }
            }
            
            # Conversion TensorFlow Lite
            if TENSORFLOW_AVAILABLE:
                tflite_result = self._convert_to_tflite(onnx_path, config, model_type)
                results['conversions']['tflite'] = tflite_result
            
            # Conversion Core ML (macOS seulement)
            if COREML_AVAILABLE:
                coreml_result = self._convert_to_coreml(onnx_path, config, model_type)
                results['conversions']['coreml'] = coreml_result
            
            # Nettoyage des fichiers temporaires
            if onnx_path.exists():
                onnx_path.unlink()
            
            self.conversion_stats['models_processed'] += 1
            self.conversion_stats['successful_conversions'] += 1
            
            logger.info(f"✅ Quantification {model_type} terminée avec succès")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur quantification {model_type}: {e}")
            self.conversion_stats['failed_conversions'] += 1
            return {
                'success': False,
                'model_type': model_type,
                'error': str(e)
            }
    
    def _load_pytorch_model(self, model_path: Path, config: Dict) -> nn.Module:
        """Charge un modèle PyTorch."""
        try:
            # Importer le modèle approprié
            if 'audio' in str(model_path):
                from audio_training_efficientnet.models.efficientnet_config import EfficientNetConfig, create_model
                model_config = EfficientNetConfig(
                    model_name=config['model_name'],
                    num_classes=config['num_classes'],
                    pretrained=False
                )
            else:
                from picture_training_enhanced.models.photo_config import PhotoConfig, create_model
                model_config = PhotoConfig(
                    model_name=config['model_name'],
                    num_classes=config['num_classes'],
                    architecture='efficientnet',
                    pretrained=False
                )
            
            # Créer et charger le modèle
            model = create_model(model_config)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle PyTorch: {e}")
            raise
    
    def _convert_to_onnx(self, model: nn.Module, config: Dict, model_type: str) -> Path:
        """Convertit un modèle PyTorch en ONNX."""
        try:
            input_size = config['input_size']
            
            # Créer input tensor exemple
            if model_type == 'audio':
                # Spectrogramme: (batch, channels, height, width)
                dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
            else:
                # Image: (batch, channels, height, width)
                dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
            
            # Chemin de sortie ONNX
            onnx_path = self.output_dir / f'{model_type}_model.onnx'
            
            # Conversion ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"✅ Modèle ONNX sauvegardé: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"Erreur conversion ONNX: {e}")
            raise
    
    def _convert_to_tflite(self, onnx_path: Path, config: Dict, model_type: str) -> Dict:
        """Convertit ONNX en TensorFlow Lite."""
        try:
            import onnx
            import tf2onnx
            
            # Charger le modèle ONNX
            onnx_model = onnx.load(str(onnx_path))
            
            # Convertir ONNX → TensorFlow
            tf_model_path = self.output_dir / f'{model_type}_model.pb'
            tf2onnx.convert.from_onnx(onnx_model, output_path=str(tf_model_path))
            
            # Charger le modèle TensorFlow
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path.parent))
            
            # Configuration de quantification
            quantization_mode = config.get('quantization_mode', 'dynamic')
            
            if quantization_mode == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
            elif quantization_mode == 'dynamic':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Conversion TensorFlow Lite
            tflite_model = converter.convert()
            
            # Sauvegarder le modèle TensorFlow Lite
            tflite_path = self.output_dir / f'{model_type}_model.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Calculer la réduction de taille
            original_size = onnx_path.stat().st_size
            tflite_size = tflite_path.stat().st_size
            size_reduction = (original_size - tflite_size) / original_size
            
            logger.info(f"✅ Modèle TensorFlow Lite sauvegardé: {tflite_path}")
            logger.info(f"📊 Réduction de taille: {size_reduction:.1%}")
            
            return {
                'success': True,
                'output_path': str(tflite_path),
                'size_bytes': tflite_size,
                'size_reduction': size_reduction,
                'quantization_mode': quantization_mode
            }
            
        except Exception as e:
            logger.error(f"Erreur conversion TensorFlow Lite: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _convert_to_coreml(self, onnx_path: Path, config: Dict, model_type: str) -> Dict:
        """Convertit ONNX en Core ML."""
        try:
            import onnx
            from onnx_coreml import convert
            
            # Charger le modèle ONNX
            onnx_model = onnx.load(str(onnx_path))
            
            # Convertir en Core ML
            input_size = config['input_size']
            
            coreml_model = convert(
                onnx_model,
                image_input_names=['input'],
                image_output_names=['output'],
                predicted_feature_name='classLabel',
                predicted_probabilities_output='classProbability',
                class_labels=config['class_names'],
                image_scale=1.0/255.0,  # Normalisation
                red_bias=-0.485/0.229,
                green_bias=-0.456/0.224,
                blue_bias=-0.406/0.225
            )
            
            # Métadonnées
            coreml_model.short_description = f'NightScan {model_type} model'
            coreml_model.input_description['input'] = f'Input {model_type} data'
            coreml_model.output_description['output'] = f'{model_type.capitalize()} classification'
            
            # Sauvegarder le modèle Core ML
            coreml_path = self.output_dir / f'{model_type}_model.mlmodel'
            coreml_model.save(str(coreml_path))
            
            # Calculer la réduction de taille
            original_size = onnx_path.stat().st_size
            coreml_size = coreml_path.stat().st_size
            size_reduction = (original_size - coreml_size) / original_size
            
            logger.info(f"✅ Modèle Core ML sauvegardé: {coreml_path}")
            logger.info(f"📊 Réduction de taille: {size_reduction:.1%}")
            
            return {
                'success': True,
                'output_path': str(coreml_path),
                'size_bytes': coreml_size,
                'size_reduction': size_reduction
            }
            
        except Exception as e:
            logger.error(f"Erreur conversion Core ML: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_report(self, results: Dict):
        """Génère un rapport de quantification."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'conversion_stats': self.conversion_stats,
            'model_results': results,
            'system_info': {
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'coreml_available': COREML_AVAILABLE,
                'device': str(self.device)
            }
        }
        
        report_path = self.output_dir / 'quantization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Rapport de quantification sauvegardé: {report_path}")
        
        # Afficher un résumé
        print("\n" + "="*50)
        print("📊 RÉSUMÉ DE LA QUANTIFICATION")
        print("="*50)
        print(f"Modèles traités: {self.conversion_stats['models_processed']}")
        print(f"Conversions réussies: {self.conversion_stats['successful_conversions']}")
        print(f"Conversions échouées: {self.conversion_stats['failed_conversions']}")
        
        for model_type, result in results.items():
            if result.get('success'):
                print(f"\n{model_type.upper()} MODEL:")
                print(f"  Original size: {result['original_size']:,} bytes")
                
                for format_type, conversion in result.get('conversions', {}).items():
                    if conversion.get('success'):
                        print(f"  {format_type.upper()}: {conversion['size_bytes']:,} bytes "
                              f"({conversion.get('size_reduction', 0):.1%} reduction)")
        
        print("\n" + "="*50)


def main():
    """Point d'entrée principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline de quantification des modèles NightScan")
    parser.add_argument("--config", type=Path, help="Fichier de configuration JSON")
    parser.add_argument("--model-type", choices=['audio', 'photo', 'all'], default='all',
                       help="Type de modèle à quantifier")
    parser.add_argument("--output-dir", type=Path, default='mobile_models',
                       help="Dossier de sortie")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Mode verbeux")
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🌙 NightScan - Pipeline de Quantification des Modèles")
    print("="*60)
    
    # Créer le pipeline
    pipeline = ModelQuantizationPipeline(args.config)
    
    # Mettre à jour le dossier de sortie si spécifié
    if args.output_dir:
        pipeline.output_dir = args.output_dir
        pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.model_type == 'all':
            results = pipeline.quantize_all_models()
        else:
            results = {args.model_type: pipeline.quantize_model(args.model_type)}
        
        # Vérifier le succès global
        success_count = sum(1 for r in results.values() if r.get('success', False))
        total_count = len(results)
        
        if success_count == total_count:
            print(f"\n✅ Quantification terminée avec succès ({success_count}/{total_count})")
        else:
            print(f"\n⚠️ Quantification terminée avec erreurs ({success_count}/{total_count})")
            
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())