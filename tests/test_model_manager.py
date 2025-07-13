"""
Tests pour le gestionnaire de modèles ML unifié.

Ce module teste tous les aspects critiques du ModelManager:
- Lifecycle des modèles (chargement/déchargement)
- Sélection automatique du device (GPU/CPU/MPS)  
- Threading safety et pool d'instances
- Gestion mémoire et cleanup
- Configuration et validation
- Gestion d'erreurs pour modèles manquants/corrompus
"""

import pytest
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import tempfile
import json

# Import the components to test
from unified_prediction_system.model_manager import (
    ModelManager,
    ModelInfo,
    AudioModelLoader,
    PhotoModelLoader,
    ModelLoadError,
    PredictionError
)


class TestModelInfo:
    """Tests pour la classe ModelInfo."""
    
    def test_model_info_creation(self):
        """Test création basique d'un ModelInfo."""
        config = {
            'class_names': ['owl', 'deer', 'fox'],
            'num_classes': 3
        }
        model_path = Path('/fake/model.pth')
        
        info = ModelInfo('audio', model_path, config)
        
        assert info.model_type == 'audio'
        assert info.model_path == model_path
        assert info.config == config
        assert info.class_names == ['owl', 'deer', 'fox']
        assert info.predictions_count == 0
        assert info.model is None
        assert info.device is None
    
    def test_update_usage_stats(self):
        """Test mise à jour des statistiques d'utilisation."""
        info = ModelInfo('audio', Path('/fake/model.pth'), {})
        initial_count = info.predictions_count
        initial_time = info.last_used
        
        time.sleep(0.01)  # Ensure time difference
        info.update_usage()
        
        assert info.predictions_count == initial_count + 1
        assert info.last_used > initial_time
    
    def test_get_stats(self):
        """Test récupération des statistiques."""
        config = {'class_names': ['test']}
        info = ModelInfo('photo', Path('/test/model.pth'), config)
        info.device = 'cpu'
        info.predictions_count = 5
        
        stats = info.get_stats()
        
        assert stats['model_type'] == 'photo'
        assert stats['model_path'] == '/test/model.pth'
        assert stats['predictions_count'] == 5
        assert stats['device'] == 'cpu'
        assert stats['class_names'] == ['test']
        assert 'loaded_at' in stats
        assert 'last_used' in stats


class TestAudioModelLoader:
    """Tests pour le chargeur de modèles audio."""
    
    @patch('unified_prediction_system.model_manager.torch')
    @patch('unified_prediction_system.model_manager.create_model')
    @patch('unified_prediction_system.model_manager.EfficientNetConfig')
    def test_load_audio_model_success(self, mock_config_class, mock_create_model, mock_torch):
        """Test chargement réussi d'un modèle audio."""
        # Setup mocks
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        mock_torch.load.return_value = {'state': 'dict'}
        
        # Create temp model file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            model_path = Path(tmp.name)
        
        try:
            config = {'num_classes': 10}
            
            result = AudioModelLoader.load_model(model_path, config)
            
            # Verify calls
            mock_config_class.assert_called_once_with(**config)
            mock_create_model.assert_called_once()
            mock_torch.load.assert_called_once_with(model_path, map_location='cpu')
            mock_model.load_state_dict.assert_called_once_with({'state': 'dict'})
            mock_model.eval.assert_called_once()
            
            assert result == mock_model
        finally:
            model_path.unlink()
    
    def test_load_audio_model_file_not_found(self):
        """Test erreur quand le fichier modèle n'existe pas."""
        non_existent_path = Path('/nonexistent/model.pth')
        config = {'num_classes': 10}
        
        with pytest.raises(ModelLoadError) as exc_info:
            AudioModelLoader.load_model(non_existent_path, config)
        
        assert "Fichier modèle audio non trouvé" in str(exc_info.value)
    
    @patch('unified_prediction_system.model_manager.torch')
    @patch('unified_prediction_system.model_manager.create_model')
    def test_load_audio_model_loading_error(self, mock_create_model, mock_torch):
        """Test gestion d'erreur lors du chargement."""
        mock_torch.load.side_effect = RuntimeError("Loading failed")
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            model_path = Path(tmp.name)
        
        try:
            config = {'num_classes': 10}
            
            with pytest.raises(ModelLoadError) as exc_info:
                AudioModelLoader.load_model(model_path, config)
            
            assert "Erreur lors du chargement du modèle audio" in str(exc_info.value)
        finally:
            model_path.unlink()


class TestModelManager:
    """Tests pour le gestionnaire principal de modèles."""
    
    def test_model_manager_initialization(self):
        """Test initialisation du ModelManager."""
        manager = ModelManager()
        
        assert manager.models == {}
        assert manager.config == {}
        assert manager.device is not None
        assert manager._lock is not None
        assert manager.max_models > 0
    
    @patch('unified_prediction_system.model_manager.torch')
    def test_device_selection_cuda(self, mock_torch):
        """Test sélection automatique GPU CUDA."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        manager = ModelManager()
        device = manager._select_device()
        
        assert 'cuda' in str(device)
    
    @patch('unified_prediction_system.model_manager.torch')
    def test_device_selection_mps(self, mock_torch):
        """Test sélection automatique GPU MPS (Apple Silicon)."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        
        manager = ModelManager()
        device = manager._select_device()
        
        assert 'mps' in str(device)
    
    @patch('unified_prediction_system.model_manager.torch')
    def test_device_selection_cpu_fallback(self, mock_torch):
        """Test fallback vers CPU quand GPU indisponible."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        
        manager = ModelManager()
        device = manager._select_device()
        
        assert str(device) == 'cpu'
    
    def test_threading_safety_load_model(self):
        """Test thread safety lors du chargement concurrent."""
        manager = ModelManager()
        results = []
        errors = []
        
        def load_model_thread(model_id):
            try:
                # Mock successful loading
                with patch.object(manager, '_load_model_from_disk') as mock_load:
                    mock_model = Mock()
                    mock_load.return_value = mock_model
                    
                    model = manager.load_model(f'test_model_{model_id}', 'audio')
                    results.append((model_id, model))
            except Exception as e:
                errors.append((model_id, e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=load_model_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors and consistent results
        assert len(errors) == 0
        assert len(results) == 5
        assert len(manager.models) <= manager.max_models
    
    def test_memory_management_max_models(self):
        """Test gestion mémoire avec limite de modèles."""
        manager = ModelManager()
        manager.max_models = 2  # Limite à 2 modèles
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            # Create mock models
            mock_models = [Mock() for _ in range(3)]
            mock_load.side_effect = mock_models
            
            # Load 3 models (should trigger cleanup)
            model1 = manager.load_model('model1', 'audio')
            model2 = manager.load_model('model2', 'audio')
            model3 = manager.load_model('model3', 'audio')
            
            # Should have max 2 models loaded
            assert len(manager.models) <= manager.max_models
    
    def test_unload_model(self):
        """Test déchargement d'un modèle."""
        manager = ModelManager()
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Load model
            manager.load_model('test_model', 'audio')
            assert 'test_model' in manager.models
            
            # Unload model
            manager.unload_model('test_model')
            assert 'test_model' not in manager.models
    
    def test_unload_nonexistent_model(self):
        """Test déchargement d'un modèle inexistant."""
        manager = ModelManager()
        
        # Should not raise error
        manager.unload_model('nonexistent_model')
        assert len(manager.models) == 0
    
    def test_cleanup_old_models(self):
        """Test nettoyage automatique des anciens modèles."""
        manager = ModelManager()
        manager.max_models = 2
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            mock_load.return_value = Mock()
            
            # Load models with different usage times
            model1 = manager.load_model('model1', 'audio')
            time.sleep(0.01)
            model2 = manager.load_model('model2', 'audio')
            time.sleep(0.01)
            
            # Force cleanup by loading a third model
            model3 = manager.load_model('model3', 'audio')
            
            # model1 should be removed (oldest)
            assert 'model1' not in manager.models
            assert 'model2' in manager.models
            assert 'model3' in manager.models
    
    def test_get_model_stats(self):
        """Test récupération des statistiques de tous les modèles."""
        manager = ModelManager()
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            mock_load.return_value = Mock()
            
            # Load some models
            manager.load_model('model1', 'audio')
            manager.load_model('model2', 'photo')
            
            stats = manager.get_model_stats()
            
            assert len(stats) == 2
            assert 'model1' in stats
            assert 'model2' in stats
            assert stats['model1']['model_type'] == 'audio'
            assert stats['model2']['model_type'] == 'photo'
    
    def test_clear_cache(self):
        """Test vidage complet du cache."""
        manager = ModelManager()
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            mock_load.return_value = Mock()
            
            # Load models
            manager.load_model('model1', 'audio')
            manager.load_model('model2', 'photo')
            assert len(manager.models) == 2
            
            # Clear cache
            manager.clear_cache()
            assert len(manager.models) == 0
    
    def test_configuration_loading(self):
        """Test chargement de configuration depuis fichier."""
        config_data = {
            'audio_models': {'default': '/path/to/audio.pth'},
            'photo_models': {'default': '/path/to/photo.pth'},
            'max_models': 5,
            'device': 'cpu'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(config_data, tmp)
            config_path = tmp.name
        
        try:
            manager = ModelManager(config_path=config_path)
            
            assert manager.config == config_data
            assert manager.max_models == 5
        finally:
            Path(config_path).unlink()
    
    def test_invalid_model_type_error(self):
        """Test erreur pour type de modèle invalide."""
        manager = ModelManager()
        
        with pytest.raises(ValueError) as exc_info:
            manager.load_model('test_model', 'invalid_type')
        
        assert "Type de modèle non supporté" in str(exc_info.value)


class TestModelManagerIntegration:
    """Tests d'intégration pour le ModelManager."""
    
    @pytest.mark.ml_integration
    def test_end_to_end_model_lifecycle(self):
        """Test complet du cycle de vie d'un modèle."""
        manager = ModelManager()
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            # Setup mock model
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Load model
            model = manager.load_model('integration_test', 'audio')
            assert model is not None
            assert 'integration_test' in manager.models
            
            # Use model (simulate prediction)
            info = manager.models['integration_test']
            info.update_usage()
            assert info.predictions_count == 1
            
            # Get stats
            stats = manager.get_model_stats()
            assert stats['integration_test']['predictions_count'] == 1
            
            # Unload model
            manager.unload_model('integration_test')
            assert 'integration_test' not in manager.models
    
    @pytest.mark.performance_critical
    def test_concurrent_model_loading_performance(self):
        """Test performance du chargement concurrent."""
        manager = ModelManager()
        num_threads = 10
        models_per_thread = 5
        
        results = []
        
        def load_models_thread(thread_id):
            thread_results = []
            for i in range(models_per_thread):
                start_time = time.time()
                
                with patch.object(manager, '_load_model_from_disk') as mock_load:
                    mock_load.return_value = Mock()
                    model = manager.load_model(f'perf_model_{thread_id}_{i}', 'audio')
                
                load_time = time.time() - start_time
                thread_results.append(load_time)
            
            results.extend(thread_results)
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=load_models_thread, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert len(results) == num_threads * models_per_thread
        assert total_time < 10.0  # Should complete within 10 seconds
        assert max(results) < 1.0  # Individual loads should be < 1 second
        assert len(manager.models) <= manager.max_models  # Memory management working


@pytest.mark.ml_unit
class TestModelManagerEdgeCases:
    """Tests pour les cas limites et erreurs."""
    
    def test_memory_pressure_handling(self):
        """Test gestion de la pression mémoire."""
        manager = ModelManager()
        manager.max_models = 1  # Force immediate cleanup
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            mock_load.return_value = Mock()
            
            # Load first model
            model1 = manager.load_model('model1', 'audio')
            assert len(manager.models) == 1
            
            # Load second model (should evict first)
            model2 = manager.load_model('model2', 'audio')
            assert len(manager.models) == 1
            assert 'model1' not in manager.models
            assert 'model2' in manager.models
    
    def test_corrupted_model_handling(self):
        """Test gestion des modèles corrompus."""
        manager = ModelManager()
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            mock_load.side_effect = ModelLoadError("Modèle corrompu")
            
            with pytest.raises(ModelLoadError):
                manager.load_model('corrupted_model', 'audio')
            
            # Verify no partial state
            assert 'corrupted_model' not in manager.models
    
    def test_device_change_handling(self):
        """Test gestion du changement de device."""
        manager = ModelManager()
        original_device = manager.device
        
        with patch.object(manager, '_load_model_from_disk') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Load model
            model = manager.load_model('test_model', 'audio')
            
            # Change device
            manager._update_device('cuda:0')
            
            # Verify model moved to new device
            assert str(manager.device) == 'cuda:0'
            # In real implementation, would verify model.to(device) was called