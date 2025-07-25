"""
Tests pour le module de segmentation audio
"""

import pytest
import numpy as np
import torch
import torchaudio
from pathlib import Path
import tempfile
import shutil
import json

import sys
sys.path.append(str(Path(__file__).parent.parent / "audio_training_efficientnet"))

from audio_segmentation import AudioSegmenter


@pytest.fixture
def temp_dir():
    """Crée un répertoire temporaire pour les tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def segmenter():
    """Crée une instance du segmenteur pour les tests."""
    return AudioSegmenter(
        segment_duration=3.0,
        overlap=1.0,
        min_segment_duration=1.0
    )


def create_test_audio(duration: float, sample_rate: int = 22050) -> torch.Tensor:
    """
    Crée un signal audio de test.
    
    Args:
        duration: Durée en secondes
        sample_rate: Taux d'échantillonnage
        
    Returns:
        Tensor audio (1, samples)
    """
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Signal avec plusieurs fréquences
    signal = 0.5 * torch.sin(2 * np.pi * 440 * t)  # La 440 Hz
    signal += 0.3 * torch.sin(2 * np.pi * 880 * t)  # La octave
    return signal.unsqueeze(0)


class TestAudioSegmenter:
    """Tests pour la classe AudioSegmenter."""
    
    def test_initialization(self):
        """Test l'initialisation du segmenteur."""
        segmenter = AudioSegmenter(
            segment_duration=5.0,
            overlap=1.0,
            min_segment_duration=2.0,
            sample_rate=16000
        )
        
        assert segmenter.segment_duration == 5.0
        assert segmenter.overlap == 1.0
        assert segmenter.min_segment_duration == 2.0
        assert segmenter.sample_rate == 16000
        assert segmenter.hop_duration == 4.0
    
    def test_invalid_overlap(self):
        """Test qu'un chevauchement invalide lève une exception."""
        with pytest.raises(ValueError):
            AudioSegmenter(segment_duration=3.0, overlap=3.0)
    
    def test_get_audio_info(self, segmenter, temp_dir):
        """Test l'obtention des informations audio."""
        # Créer un fichier audio de test
        audio_path = temp_dir / "test.wav"
        waveform = create_test_audio(5.0, 22050)
        torchaudio.save(str(audio_path), waveform, 22050)
        
        # Obtenir les infos
        info = segmenter.get_audio_info(audio_path)
        
        assert info['duration'] == pytest.approx(5.0, rel=0.01)
        assert info['sample_rate'] == 22050
        assert info['channels'] == 1
        assert info['num_frames'] == 5 * 22050
    
    def test_segment_short_file(self, segmenter, temp_dir):
        """Test qu'un fichier court n'est pas segmenté."""
        # Créer un fichier court (2 secondes)
        audio_path = temp_dir / "short.wav"
        waveform = create_test_audio(2.0)
        torchaudio.save(str(audio_path), waveform, 22050)
        
        # Segmenter
        output_dir = temp_dir / "segments"
        segments = segmenter.segment_audio_file(audio_path, output_dir)
        
        # Vérifier qu'aucun segment n'est créé
        assert len(segments) == 0
        assert not output_dir.exists()
    
    def test_segment_long_file(self, segmenter, temp_dir):
        """Test la segmentation d'un fichier long."""
        # Créer un fichier de 10 secondes
        audio_path = temp_dir / "long.wav"
        waveform = create_test_audio(10.0)
        torchaudio.save(str(audio_path), waveform, 22050)
        
        # Segmenter (segments de 3s avec overlap de 1s)
        output_dir = temp_dir / "segments"
        segments = segmenter.segment_audio_file(audio_path, output_dir)
        
        # Vérifier les segments créés
        assert len(segments) == 5  # 0-3, 2-5, 4-7, 6-9, 8-10 (le dernier avec padding)
        assert output_dir.exists()
        
        # Vérifier le premier segment (pas de padding)
        assert segments[0]['start_time'] == 0.0
        assert segments[0]['end_time'] == pytest.approx(3.0, rel=0.01)
        assert segments[0]['actual_duration'] == pytest.approx(3.0, rel=0.01)
        assert segments[0]['padded_duration'] == 3.0
        assert not segments[0]['has_padding']
        
        # Vérifier le chevauchement
        assert segments[1]['start_time'] == pytest.approx(2.0, rel=0.01)
        
        # Vérifier le dernier segment (devrait avoir du padding)
        last_segment = segments[-1]
        assert last_segment['actual_duration'] == pytest.approx(2.0, rel=0.01)  # 8-10s = 2s
        assert last_segment['padded_duration'] == 3.0
        assert last_segment['has_padding']
        
        # Vérifier que tous les fichiers existent et ont la bonne durée
        for segment in segments:
            segment_path = output_dir / segment['filename']
            assert segment_path.exists()
            # Vérifier que le fichier a bien la durée padded
            info = torchaudio.info(str(segment_path))
            duration = info.num_frames / info.sample_rate
            assert duration == pytest.approx(3.0, rel=0.01)
    
    def test_segment_with_resample(self, temp_dir):
        """Test la segmentation avec rééchantillonnage."""
        # Créer un segmenteur avec sample_rate spécifique
        segmenter = AudioSegmenter(
            segment_duration=3.0,
            overlap=0.0,
            sample_rate=16000
        )
        
        # Créer un fichier à 22050 Hz
        audio_path = temp_dir / "high_sr.wav"
        waveform = create_test_audio(6.0, 22050)
        torchaudio.save(str(audio_path), waveform, 22050)
        
        # Segmenter
        output_dir = temp_dir / "segments"
        segments = segmenter.segment_audio_file(audio_path, output_dir)
        
        # Vérifier que les segments sont à 16000 Hz
        assert len(segments) == 2
        for segment in segments:
            segment_path = output_dir / segment['filename']
            info = torchaudio.info(str(segment_path))
            assert info.sample_rate == 16000
    
    def test_segment_directory(self, segmenter, temp_dir):
        """Test la segmentation d'un répertoire complet."""
        # Créer une structure de test
        class1_dir = temp_dir / "class1"
        class2_dir = temp_dir / "class2"
        class1_dir.mkdir()
        class2_dir.mkdir()
        
        # Créer des fichiers audio
        files_created = []
        
        # Classe 1: 2 fichiers (1 court, 1 long)
        short_file = class1_dir / "short.wav"
        torchaudio.save(str(short_file), create_test_audio(2.0), 22050)
        files_created.append(short_file)
        
        long_file = class1_dir / "long.wav"
        torchaudio.save(str(long_file), create_test_audio(10.0), 22050)
        files_created.append(long_file)
        
        # Classe 2: 1 fichier long
        long_file2 = class2_dir / "recording.wav"
        torchaudio.save(str(long_file2), create_test_audio(8.0), 22050)
        files_created.append(long_file2)
        
        # Segmenter le répertoire
        output_dir = temp_dir / "output"
        all_segments = segmenter.segment_directory(
            temp_dir,
            output_dir,
            preserve_structure=True
        )
        
        # Vérifier les résultats
        assert len(all_segments) == 2  # Seulement les 2 fichiers longs
        
        # Vérifier la structure préservée
        assert (output_dir / "class1").exists()
        assert (output_dir / "class2").exists()
        
        # Vérifier les métadonnées
        metadata_path = output_dir / "segmentation_metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata['statistics']['total_files'] == 3
        assert metadata['statistics']['segmented_files'] == 2
        assert metadata['statistics']['total_segments'] > 0
    
    def test_padding_short_segments(self, temp_dir):
        """Test que les segments courts sont correctement paddés."""
        segmenter = AudioSegmenter(
            segment_duration=5.0,
            overlap=0.0,
            min_segment_duration=2.0
        )
        
        # Créer un fichier de 12 secondes
        audio_path = temp_dir / "test_padding.wav"
        waveform = create_test_audio(12.0)
        torchaudio.save(str(audio_path), waveform, 22050)
        
        # Segmenter
        output_dir = temp_dir / "segments"
        segments = segmenter.segment_audio_file(audio_path, output_dir)
        
        # On devrait avoir 3 segments: 0-5s, 5-10s, 10-12s (avec 3s de padding)
        assert len(segments) == 3
        
        # Vérifier les deux premiers segments (pas de padding)
        for i in range(2):
            assert segments[i]['actual_duration'] == pytest.approx(5.0, rel=0.01)
            assert segments[i]['padded_duration'] == 5.0
            assert not segments[i]['has_padding']
        
        # Vérifier le dernier segment (avec padding)
        last_segment = segments[2]
        assert last_segment['actual_duration'] == pytest.approx(2.0, rel=0.01)
        assert last_segment['padded_duration'] == 5.0
        assert last_segment['has_padding']
        
        # Vérifier que tous les fichiers ont exactement 5 secondes
        for segment in segments:
            segment_path = output_dir / segment['filename']
            info = torchaudio.info(str(segment_path))
            duration = info.num_frames / info.sample_rate
            assert duration == pytest.approx(5.0, rel=0.01)
    
    def test_min_segment_duration(self, temp_dir):
        """Test que les segments trop courts sont ignorés."""
        segmenter = AudioSegmenter(
            segment_duration=3.0,
            overlap=0.0,
            min_segment_duration=2.0
        )
        
        # Créer un fichier de 7 secondes
        audio_path = temp_dir / "medium.wav"
        waveform = create_test_audio(7.0)
        torchaudio.save(str(audio_path), waveform, 22050)
        
        # Segmenter
        output_dir = temp_dir / "segments"
        segments = segmenter.segment_audio_file(audio_path, output_dir)
        
        # On devrait avoir 2 segments: 0-3s et 3-6s
        # Le troisième segment (6-7s) est trop court (1s < 2s min)
        assert len(segments) == 2
        assert segments[0]['actual_duration'] == pytest.approx(3.0, rel=0.01)
        assert segments[0]['padded_duration'] == 3.0
        assert segments[1]['actual_duration'] == pytest.approx(3.0, rel=0.01)
        assert segments[1]['padded_duration'] == 3.0


@pytest.mark.integration
class TestIntegration:
    """Tests d'intégration avec prepare_audio_data.py"""
    
    def test_prepare_with_segmentation(self, temp_dir):
        """Test l'intégration avec prepare_audio_data.py"""
        # Cette intégration est testée manuellement
        # car elle nécessite l'import de prepare_audio_data
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])