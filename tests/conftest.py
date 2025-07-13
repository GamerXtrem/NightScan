"""
Configuration et fixtures globales pour les tests ML.

Ce module fournit des fixtures réutilisables pour tous les tests:
- Fixtures de modèles ML mockés
- Fixtures de données de test
- Configuration de l'environnement de test
- Utilities de test communes
"""

import sys
import types
from contextlib import contextmanager
import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
from PIL import Image
import struct
from typing import Dict, Any, List


def _stub_torch():
    torch = types.ModuleType("torch")
    torch.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(Dataset=object, DataLoader=object)
    )
    torch.nn = types.SimpleNamespace(
        Module=object,
        Linear=object,
        Sequential=lambda *a: types.SimpleNamespace(state_dict=lambda: {}),
        Conv2d=object,
        ReLU=object,
        AdaptiveAvgPool2d=object,
        Flatten=object,
        Softmax=lambda *a, **k: lambda x: x,
        functional=types.SimpleNamespace(pad=lambda *a, **k: None),
    )
    torch.Tensor = object
    torch.device = type("device", (), {})
    torch.load = lambda *a, **k: {}
    torch.save = lambda *a, **k: None
    torch.manual_seed = lambda x: None
    torch.topk = lambda *a, **k: ([], [])

    @contextmanager
    def no_grad():
        yield
    torch.no_grad = no_grad
    sys.modules['torch'] = torch
    sys.modules['torch.utils'] = torch.utils
    sys.modules['torch.utils.data'] = torch.utils.data


def _stub_torchvision():
    torchvision = types.ModuleType("torchvision")
    torchvision.models = types.SimpleNamespace(
        resnet18=lambda: types.SimpleNamespace(fc=types.SimpleNamespace(in_features=0))
    )
    torchvision.transforms = types.SimpleNamespace(
        Compose=lambda *a, **k: lambda x: x,
        Lambda=lambda f: f,
        Resize=lambda *a, **k: lambda x: x,
    )
    sys.modules['torchvision'] = torchvision


def _stub_torchaudio():
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.load = lambda *a, **k: (None, 22050)
    torchaudio.functional = types.SimpleNamespace(resample=lambda x, orig_sr, sr: x)
    torchaudio.transforms = types.SimpleNamespace(
        MelSpectrogram=lambda *a, **k: lambda x: x,
        AmplitudeToDB=lambda *a, **k: lambda x: x,
    )
    sys.modules['torchaudio'] = torchaudio


def _stub_pyaudioop():
    pyaudioop = types.ModuleType("pyaudioop")

    def _return_zero(*args, **kwargs):
        return 0

    for name in [
        "max",
        "minmax",
        "avg",
        "avgpp",
        "rms",
        "bias",
        "ulaw2lin",
        "lin2ulaw",
        "lin2lin",
        "add",
        "mul",
        "ratecv",
        "cross",
    ]:
        setattr(pyaudioop, name, _return_zero)

    sys.modules["pyaudioop"] = pyaudioop


def pytest_configure(config):
    """Configuration pytest avec stubs pour dépendances ML."""
    try:
        import torch  # noqa: F401
    except Exception:
        _stub_torch()

    try:
        import torchvision  # noqa: F401
    except Exception:
        _stub_torchvision()

    try:
        import torchaudio  # noqa: F401
    except Exception:
        _stub_torchaudio()

    try:
        import pyaudioop  # noqa: F401
    except Exception:
        _stub_pyaudioop()


@pytest.fixture(scope="session")
def test_data_dir():
    """Créer un répertoire temporaire pour les données de test."""
    temp_dir = tempfile.mkdtemp(prefix="nightscan_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def mock_models_dir(test_data_dir):
    """Créer des modèles mockés pour les tests."""
    models_dir = test_data_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create mock model files
    (models_dir / "audio_model.pth").write_text("mock_audio_model")
    (models_dir / "image_model.pth").write_text("mock_image_model")
    
    return models_dir


@pytest.fixture(scope="session")
def sample_audio_files(test_data_dir):
    """Créer des fichiers audio de test valides."""
    audio_dir = test_data_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    sample_files = []
    
    for i in range(5):
        wav_file = audio_dir / f"test_audio_{i}.wav"
        create_test_audio_file(wav_file, duration=1.0 + i * 0.5)
        sample_files.append(wav_file)
    
    return sample_files


@pytest.fixture(scope="session")
def sample_image_files(test_data_dir):
    """Créer des fichiers image de test valides."""
    image_dir = test_data_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    sample_files = []
    
    for i in range(5):
        # JPEG files
        jpg_file = image_dir / f"test_image_{i}.jpg"
        create_test_image_file(jpg_file, width=640 + i * 100, height=480 + i * 50)
        sample_files.append(jpg_file)
        
        # PNG files
        png_file = image_dir / f"test_image_{i}.png"
        create_test_image_file(png_file, width=640 + i * 100, height=480 + i * 50)
        sample_files.append(png_file)
    
    return sample_files


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager pour les tests."""
    manager = Mock()
    manager.models = {}
    manager.max_models = 10
    
    def mock_load_model(model_id, model_type):
        mock_model = Mock()
        mock_model.model_id = model_id
        mock_model.model_type = model_type
        manager.models[model_id] = mock_model
        return mock_model
    
    manager.load_model.side_effect = mock_load_model
    manager.get_model.side_effect = lambda model_id: manager.models.get(model_id)
    
    # Mock predict methods
    manager.predict_audio.return_value = {
        'species': 'test_species',
        'confidence': 0.85,
        'predictions': [
            {'class': 'test_species', 'confidence': 0.85},
            {'class': 'other_species', 'confidence': 0.15}
        ]
    }
    
    manager.predict_image.return_value = {
        'species': 'test_animal',
        'confidence': 0.92,
        'bounding_boxes': [
            {
                'species': 'test_animal',
                'confidence': 0.92,
                'bbox': {'x': 100, 'y': 100, 'width': 200, 'height': 150}
            }
        ]
    }
    
    return manager


@pytest.fixture
def mock_file_detector():
    """Mock FileTypeDetector pour les tests."""
    detector = Mock()
    
    def mock_detect_file_type(file_path):
        file_path = str(file_path)
        
        mock_info = Mock()
        mock_info.file_path = file_path
        mock_info.is_valid = True
        
        if file_path.endswith(('.wav', '.mp3', '.m4a')):
            mock_info.file_type = 'audio'
            mock_info.format = 'WAV' if file_path.endswith('.wav') else 'MP3'
            mock_info.metadata = {
                'duration': 5.0,
                'sample_rate': 44100,
                'channels': 1,
                'size_bytes': 441000
            }
        elif file_path.endswith(('.jpg', '.jpeg', '.png')):
            mock_info.file_type = 'image'
            mock_info.format = 'JPEG' if 'jp' in file_path.lower() else 'PNG'
            mock_info.metadata = {
                'width': 640,
                'height': 480,
                'channels': 3,
                'size_bytes': 307200
            }
        else:
            mock_info.is_valid = False
            mock_info.file_type = 'unknown'
            mock_info.format = None
            mock_info.metadata = {}
        
        return mock_info
    
    detector.detect_file_type.side_effect = mock_detect_file_type
    return detector


@pytest.fixture
def mock_prediction_router(mock_model_manager, mock_file_detector):
    """Mock PredictionRouter pour les tests."""
    router = Mock()
    router.model_manager = mock_model_manager
    router.file_detector = mock_file_detector
    
    def mock_predict(file_path):
        file_info = mock_file_detector.detect_file_type(file_path)
        
        if not file_info.is_valid:
            raise ValueError("Type de fichier non supporté")
        
        if file_info.file_type == 'audio':
            predictions = mock_model_manager.predict_audio('mock_data')
        elif file_info.file_type == 'image':
            predictions = mock_model_manager.predict_image('mock_data')
        else:
            raise ValueError("Type de fichier non supporté")
        
        return {
            'status': 'success',
            'file_type': file_info.file_type,
            'predictions': predictions,
            'metadata': file_info.metadata,
            'processing_time': 0.123
        }
    
    router.predict.side_effect = mock_predict
    
    def mock_predict_batch(file_paths, max_workers=4):
        results = []
        for file_path in file_paths:
            try:
                result = mock_predict(file_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'status': 'error',
                    'error': str(e),
                    'file_path': str(file_path)
                })
        return results
    
    router.predict_batch.side_effect = mock_predict_batch
    return router


@pytest.fixture
def mock_unified_api(mock_prediction_router):
    """Mock UnifiedPredictionAPI pour les tests."""
    api = Mock()
    api.router = mock_prediction_router
    
    api.predict_file.side_effect = lambda file_path: mock_prediction_router.predict(file_path)
    api.predict_batch.side_effect = lambda file_paths: mock_prediction_router.predict_batch(file_paths)
    
    api.get_prediction_status.return_value = {
        'prediction_id': 'test_pred_123',
        'status': 'completed',
        'progress': 100
    }
    
    api.get_prediction_result.return_value = {
        'prediction_id': 'test_pred_123',
        'status': 'completed',
        'predictions': {'species': 'test_species', 'confidence': 0.85}
    }
    
    return api


@pytest.fixture
def performance_baseline():
    """Baseline de performance pour tests de régression."""
    return {
        'model_loading_time': 0.8,
        'prediction_latency': {
            'audio': {'avg': 0.3, 'p95': 0.8, 'p99': 1.2, 'max': 1.5},
            'image': {'avg': 0.2, 'p95': 0.6, 'p99': 1.0, 'max': 1.3}
        },
        'memory_usage': {'model_footprint': 50, 'peak_processing': 100},
        'throughput': {'sequential': 5, 'concurrent': 15}
    }


@pytest.fixture
def accuracy_baseline():
    """Baseline de précision pour tests de régression."""
    return {
        'audio_model': {
            'overall_accuracy': 0.85,
            'top1_accuracy': 0.85,
            'top3_accuracy': 0.95,
            'avg_confidence': 0.78
        },
        'image_model': {
            'overall_accuracy': 0.88,
            'top1_accuracy': 0.88,
            'top3_accuracy': 0.96,
            'avg_confidence': 0.82
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Configuration automatique de l'environnement de test."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("MODEL_PATH", "tests/models/test_model.pth")
    monkeypatch.setenv("CSV_DIR", "tests/data")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    
    # Set deterministic behavior
    np.random.seed(42)


@pytest.fixture(scope="function")
def isolated_test_env(tmp_path):
    """Environnement de test isolé pour chaque test."""
    test_dirs = {
        'models': tmp_path / 'models',
        'data': tmp_path / 'data',
        'logs': tmp_path / 'logs',
        'temp': tmp_path / 'temp'
    }
    
    for dir_path in test_dirs.values():
        dir_path.mkdir(exist_ok=True)
    
    return test_dirs


def create_test_audio_file(file_path: Path, duration: float = 1.0, sample_rate: int = 44100):
    """Utilitaire pour créer un fichier audio de test."""
    samples = int(duration * sample_rate)
    audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)
    
    with open(file_path, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<L', samples * 2 + 36))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<L', 16))
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', 1))   # Mono
        f.write(struct.pack('<L', sample_rate))
        f.write(struct.pack('<L', sample_rate * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<L', samples * 2))
        f.write(audio_data.tobytes())


def create_test_image_file(file_path: Path, width: int = 640, height: int = 480):
    """Utilitaire pour créer un fichier image de test."""
    img = Image.new('RGB', (width, height), color=(100, 150, 200))
    img.save(file_path)
