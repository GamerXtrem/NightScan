"""
Tests pour le détecteur de type de fichier.

Ce module teste tous les aspects du FileTypeDetector:
- Détection automatique des formats audio (WAV, MP3, etc.)
- Détection des formats image (JPG, PNG, etc.)  
- Validation des spectrogrammes et fichiers ML
- Extraction précise des métadonnées
- Gestion robuste des fichiers corrompus
- Performance de détection sur volume
"""

import pytest
import tempfile
import struct
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any
import numpy as np
from PIL import Image
import io

# Import the components to test
from unified_prediction_system.file_type_detector import (
    FileTypeDetector,
    FileType,
    NightScanFile
)


class TestFileType:
    """Tests pour l'enum FileType."""
    
    def test_file_type_enum_values(self):
        """Test que tous les types de fichiers sont définis."""
        assert FileType.AUDIO_RAW.value == "audio_raw"
        assert FileType.AUDIO_SPECTROGRAM.value == "audio_spectrogram"
        assert FileType.IMAGE.value == "image"
        assert FileType.UNKNOWN.value == "unknown"
    
    def test_file_type_enum_completeness(self):
        """Test que tous les types attendus sont présents."""
        expected_types = {'audio_raw', 'audio_spectrogram', 'image', 'unknown'}
        actual_types = {ft.value for ft in FileType}
        assert actual_types == expected_types


class TestNightScanFile:
    """Tests pour la classe NightScanFile."""
    
    def test_nightscan_file_creation_nonexistent(self):
        """Test création avec fichier inexistant."""
        file_obj = NightScanFile('/nonexistent/file.wav')
        
        assert file_obj.file_path == Path('/nonexistent/file.wav')
        assert file_obj.file_type == FileType.UNKNOWN
        assert file_obj.is_valid == False
        assert "Fichier non trouvé" in file_obj.error_message
        assert file_obj.metadata == {}
    
    def test_wav_file_analysis(self):
        """Test analyse d'un fichier WAV valide."""
        # Create minimal WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Write WAV header (RIFF chunk)
            tmp.write(b'RIFF')  # ChunkID
            tmp.write(struct.pack('<L', 36))  # ChunkSize  
            tmp.write(b'WAVE')  # Format
            
            # Write fmt chunk
            tmp.write(b'fmt ')  # Subchunk1ID
            tmp.write(struct.pack('<L', 16))  # Subchunk1Size
            tmp.write(struct.pack('<H', 1))   # AudioFormat (PCM)
            tmp.write(struct.pack('<H', 1))   # NumChannels (mono)
            tmp.write(struct.pack('<L', 44100))  # SampleRate
            tmp.write(struct.pack('<L', 44100))  # ByteRate
            tmp.write(struct.pack('<H', 1))   # BlockAlign
            tmp.write(struct.pack('<H', 16))  # BitsPerSample
            
            # Write data chunk header
            tmp.write(b'data')  # Subchunk2ID
            tmp.write(struct.pack('<L', 0))  # Subchunk2Size (no data)
            
            wav_path = tmp.name
        
        try:
            file_obj = NightScanFile(wav_path)
            
            assert file_obj.file_type == FileType.AUDIO_RAW
            assert file_obj.is_valid == True
            assert file_obj.error_message is None
            assert 'sample_rate' in file_obj.metadata
            assert file_obj.metadata['sample_rate'] == 44100
            assert file_obj.metadata['channels'] == 1
            assert file_obj.metadata['format'] == 'WAV'
        finally:
            Path(wav_path).unlink()
    
    def test_jpeg_file_analysis(self):
        """Test analyse d'un fichier JPEG valide."""
        # Create minimal JPEG file
        img = Image.new('RGB', (100, 100), color='red')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp, format='JPEG')
            jpg_path = tmp.name
        
        try:
            file_obj = NightScanFile(jpg_path)
            
            assert file_obj.file_type == FileType.IMAGE
            assert file_obj.is_valid == True
            assert file_obj.error_message is None
            assert file_obj.metadata['width'] == 100
            assert file_obj.metadata['height'] == 100
            assert file_obj.metadata['format'] == 'JPEG'
            assert 'file_size' in file_obj.metadata
        finally:
            Path(jpg_path).unlink()
    
    def test_png_file_analysis(self):
        """Test analyse d'un fichier PNG valide."""
        # Create PNG file
        img = Image.new('RGBA', (200, 150), color=(0, 255, 0, 128))
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp, format='PNG')
            png_path = tmp.name
        
        try:
            file_obj = NightScanFile(png_path)
            
            assert file_obj.file_type == FileType.IMAGE
            assert file_obj.is_valid == True
            assert file_obj.metadata['width'] == 200
            assert file_obj.metadata['height'] == 150
            assert file_obj.metadata['format'] == 'PNG'
            assert file_obj.metadata['channels'] == 4  # RGBA
        finally:
            Path(png_path).unlink()
    
    def test_numpy_spectrogram_analysis(self):
        """Test analyse d'un fichier spectrogramme numpy."""
        # Create numpy spectrogram file
        spectrogram = np.random.rand(128, 256).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, spectrogram)
            npy_path = tmp.name
        
        try:
            file_obj = NightScanFile(npy_path)
            
            assert file_obj.file_type == FileType.AUDIO_SPECTROGRAM
            assert file_obj.is_valid == True
            assert file_obj.metadata['shape'] == (128, 256)
            assert file_obj.metadata['dtype'] == 'float32'
            assert file_obj.metadata['format'] == 'NPY'
        finally:
            Path(npy_path).unlink()
    
    def test_corrupted_wav_file(self):
        """Test gestion d'un fichier WAV corrompu."""
        # Create corrupted WAV (wrong header)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'INVALID_HEADER_DATA')
            corrupted_path = tmp.name
        
        try:
            file_obj = NightScanFile(corrupted_path)
            
            assert file_obj.file_type == FileType.UNKNOWN
            assert file_obj.is_valid == False
            assert "Erreur lors de l'analyse" in file_obj.error_message
        finally:
            Path(corrupted_path).unlink()
    
    def test_corrupted_image_file(self):
        """Test gestion d'un fichier image corrompu."""
        # Create corrupted image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b'NOT_AN_IMAGE_FILE')
            corrupted_path = tmp.name
        
        try:
            file_obj = NightScanFile(corrupted_path)
            
            assert file_obj.file_type == FileType.UNKNOWN
            assert file_obj.is_valid == False
            assert file_obj.error_message is not None
        finally:
            Path(corrupted_path).unlink()
    
    def test_unknown_file_extension(self):
        """Test fichier avec extension inconnue."""
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as tmp:
            tmp.write(b'Some random content')
            unknown_path = tmp.name
        
        try:
            file_obj = NightScanFile(unknown_path)
            
            assert file_obj.file_type == FileType.UNKNOWN
            assert file_obj.is_valid == False
        finally:
            Path(unknown_path).unlink()


class TestFileTypeDetector:
    """Tests pour le détecteur principal de type de fichier."""
    
    def test_detector_initialization(self):
        """Test initialisation du FileTypeDetector."""
        detector = FileTypeDetector()
        
        assert detector.supported_audio_formats is not None
        assert detector.supported_image_formats is not None
        assert len(detector.supported_audio_formats) > 0
        assert len(detector.supported_image_formats) > 0
    
    def test_detect_wav_file_type(self):
        """Test détection automatique d'un fichier WAV."""
        detector = FileTypeDetector()
        
        # Create WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Minimal WAV header
            tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            tmp.write(b'\x10\x00\x00\x00\x01\x00\x01\x00')  # PCM, mono
            tmp.write(b'\x44\xAC\x00\x00\x44\xAC\x00\x00')  # 44100 Hz
            tmp.write(b'\x01\x00\x10\x00data\x00\x00\x00\x00')
            wav_path = tmp.name
        
        try:
            file_obj = detector.detect_file_type(wav_path)
            
            assert isinstance(file_obj, NightScanFile)
            assert file_obj.file_type == FileType.AUDIO_RAW
            assert file_obj.is_valid == True
            assert 'sample_rate' in file_obj.metadata
        finally:
            Path(wav_path).unlink()
    
    def test_detect_image_file_type(self):
        """Test détection automatique d'un fichier image."""
        detector = FileTypeDetector()
        
        # Create image file
        img = Image.new('RGB', (50, 50), color='blue')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp, format='JPEG')
            img_path = tmp.name
        
        try:
            file_obj = detector.detect_file_type(img_path)
            
            assert file_obj.file_type == FileType.IMAGE
            assert file_obj.is_valid == True
            assert file_obj.metadata['width'] == 50
            assert file_obj.metadata['height'] == 50
        finally:
            Path(img_path).unlink()
    
    def test_detect_spectrogram_file_type(self):
        """Test détection automatique d'un spectrogramme."""
        detector = FileTypeDetector()
        
        # Create spectrogram file
        spec_data = np.random.rand(64, 128)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, spec_data)
            spec_path = tmp.name
        
        try:
            file_obj = detector.detect_file_type(spec_path)
            
            assert file_obj.file_type == FileType.AUDIO_SPECTROGRAM
            assert file_obj.is_valid == True
            assert file_obj.metadata['shape'] == (64, 128)
        finally:
            Path(spec_path).unlink()
    
    def test_batch_file_detection(self):
        """Test détection batch de multiples fichiers."""
        detector = FileTypeDetector()
        
        # Create multiple files
        files = []
        
        # WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            files.append((tmp.name, FileType.AUDIO_RAW))
        
        # Image file
        img = Image.new('RGB', (10, 10))
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp, format='PNG')
            files.append((tmp.name, FileType.IMAGE))
        
        # Spectrogram file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, np.random.rand(32, 64))
            files.append((tmp.name, FileType.AUDIO_SPECTROGRAM))
        
        try:
            file_paths = [f[0] for f in files]
            expected_types = [f[1] for f in files]
            
            results = detector.detect_batch(file_paths)
            
            assert len(results) == len(files)
            for i, result in enumerate(results):
                assert result.file_type == expected_types[i]
                assert result.is_valid == True
        finally:
            for file_path, _ in files:
                Path(file_path).unlink()
    
    def test_is_supported_format(self):
        """Test vérification des formats supportés."""
        detector = FileTypeDetector()
        
        # Audio formats
        assert detector.is_supported_format('test.wav') == True
        assert detector.is_supported_format('test.mp3') == True
        assert detector.is_supported_format('test.WAV') == True  # Case insensitive
        
        # Image formats  
        assert detector.is_supported_format('test.jpg') == True
        assert detector.is_supported_format('test.jpeg') == True
        assert detector.is_supported_format('test.png') == True
        assert detector.is_supported_format('test.PNG') == True  # Case insensitive
        
        # Spectrogram formats
        assert detector.is_supported_format('test.npy') == True
        
        # Unsupported formats
        assert detector.is_supported_format('test.txt') == False
        assert detector.is_supported_format('test.doc') == False
        assert detector.is_supported_format('test') == False  # No extension
    
    def test_get_file_signature(self):
        """Test lecture de signature de fichier."""
        detector = FileTypeDetector()
        
        # Create file with known signature
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'RIFF')  # WAV signature
            tmp.write(b'\x00' * 100)  # Padding
            test_path = tmp.name
        
        try:
            signature = detector._get_file_signature(test_path, 4)
            assert signature == b'RIFF'
            
            # Test larger signature
            signature = detector._get_file_signature(test_path, 10)
            assert signature.startswith(b'RIFF')
            assert len(signature) == 10
        finally:
            Path(test_path).unlink()
    
    def test_validate_wav_signature(self):
        """Test validation spécifique des signatures WAV."""
        detector = FileTypeDetector()
        
        # Valid WAV file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'RIFF\x24\x08\x00\x00WAVE')
            valid_wav = tmp.name
        
        # Invalid WAV file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'NOT_A_WAV_FILE')
            invalid_wav = tmp.name
        
        try:
            assert detector.validate_wav_signature(valid_wav) == True
            assert detector.validate_wav_signature(invalid_wav) == False
        finally:
            Path(valid_wav).unlink()
            Path(invalid_wav).unlink()
    
    def test_extract_wav_metadata(self):
        """Test extraction de métadonnées WAV détaillées."""
        detector = FileTypeDetector()
        
        # Create WAV with specific parameters
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # RIFF header
            tmp.write(b'RIFF')
            tmp.write(struct.pack('<L', 44))  # File size - 8
            tmp.write(b'WAVE')
            
            # fmt chunk
            tmp.write(b'fmt ')
            tmp.write(struct.pack('<L', 16))  # fmt chunk size
            tmp.write(struct.pack('<H', 1))   # PCM format
            tmp.write(struct.pack('<H', 2))   # Stereo
            tmp.write(struct.pack('<L', 48000))  # 48kHz sample rate
            tmp.write(struct.pack('<L', 192000)) # Byte rate
            tmp.write(struct.pack('<H', 4))   # Block align
            tmp.write(struct.pack('<H', 16))  # 16-bit
            
            # data chunk
            tmp.write(b'data')
            tmp.write(struct.pack('<L', 8))   # Data size
            tmp.write(b'\x00' * 8)            # Sample data
            
            wav_path = tmp.name
        
        try:
            metadata = detector._extract_wav_metadata(wav_path)
            
            assert metadata['sample_rate'] == 48000
            assert metadata['channels'] == 2
            assert metadata['bits_per_sample'] == 16
            assert metadata['format'] == 'WAV'
            assert 'duration' in metadata
            assert 'file_size' in metadata
        finally:
            Path(wav_path).unlink()


class TestFileTypeDetectorIntegration:
    """Tests d'intégration pour le détecteur."""
    
    @pytest.mark.ml_integration
    def test_real_world_file_detection(self):
        """Test détection sur fichiers réalistes."""
        detector = FileTypeDetector()
        
        # Create realistic audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Complete WAV file with realistic data
            sample_rate = 44100
            duration = 1  # 1 second
            samples = int(sample_rate * duration)
            
            # WAV header
            tmp.write(b'RIFF')
            tmp.write(struct.pack('<L', 36 + samples * 2))
            tmp.write(b'WAVE')
            tmp.write(b'fmt ')
            tmp.write(struct.pack('<L', 16))
            tmp.write(struct.pack('<H', 1))   # PCM
            tmp.write(struct.pack('<H', 1))   # Mono
            tmp.write(struct.pack('<L', sample_rate))
            tmp.write(struct.pack('<L', sample_rate * 2))
            tmp.write(struct.pack('<H', 2))
            tmp.write(struct.pack('<H', 16))
            tmp.write(b'data')
            tmp.write(struct.pack('<L', samples * 2))
            
            # Generate sine wave data
            for i in range(samples):
                sample = int(32767 * np.sin(2 * np.pi * 440 * i / sample_rate))
                tmp.write(struct.pack('<h', sample))
            
            wav_path = tmp.name
        
        try:
            file_obj = detector.detect_file_type(wav_path)
            
            assert file_obj.file_type == FileType.AUDIO_RAW
            assert file_obj.is_valid == True
            assert abs(file_obj.metadata['duration'] - 1.0) < 0.1  # ~1 second
            assert file_obj.metadata['sample_rate'] == 44100
        finally:
            Path(wav_path).unlink()
    
    @pytest.mark.performance_critical
    def test_detection_performance(self):
        """Test performance de détection sur volume."""
        detector = FileTypeDetector()
        
        # Create many small files
        files = []
        for i in range(100):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                files.append(tmp.name)
        
        try:
            import time
            start_time = time.time()
            
            results = detector.detect_batch(files)
            
            detection_time = time.time() - start_time
            
            # Performance assertions
            assert len(results) == 100
            assert detection_time < 5.0  # Should detect 100 files in < 5 seconds
            assert all(r.file_type == FileType.AUDIO_RAW for r in results)
        finally:
            for file_path in files:
                Path(file_path).unlink()


@pytest.mark.ml_unit
class TestFileTypeDetectorEdgeCases:
    """Tests pour les cas limites."""
    
    def test_empty_file_handling(self):
        """Test gestion des fichiers vides."""
        detector = FileTypeDetector()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Empty file
            empty_path = tmp.name
        
        try:
            file_obj = detector.detect_file_type(empty_path)
            
            assert file_obj.file_type == FileType.UNKNOWN
            assert file_obj.is_valid == False
            assert file_obj.error_message is not None
        finally:
            Path(empty_path).unlink()
    
    def test_very_large_file_handling(self):
        """Test gestion des très gros fichiers."""
        detector = FileTypeDetector()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Create large file (simulate)
            tmp.write(b'RIFF')
            tmp.write(struct.pack('<L', 1000000000))  # 1GB simulated
            tmp.write(b'WAVEfmt ')
            large_path = tmp.name
        
        try:
            file_obj = detector.detect_file_type(large_path)
            
            # Should handle gracefully without loading entire file
            assert file_obj.file_type in [FileType.AUDIO_RAW, FileType.UNKNOWN]
            # Should not crash or consume excessive memory
        finally:
            Path(large_path).unlink()
    
    def test_permission_denied_handling(self):
        """Test gestion des erreurs de permission."""
        detector = FileTypeDetector()
        
        # Test with non-existent file (simulates permission issues)
        with pytest.raises(FileNotFoundError):
            detector.detect_file_type('/root/restricted_file.wav')
    
    def test_concurrent_detection(self):
        """Test détection concurrente thread-safe."""
        import threading
        detector = FileTypeDetector()
        results = []
        errors = []
        
        # Create test file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            test_path = tmp.name
        
        def detect_thread(thread_id):
            try:
                file_obj = detector.detect_file_type(test_path)
                results.append((thread_id, file_obj.file_type))
            except Exception as e:
                errors.append((thread_id, e))
        
        try:
            # Start multiple threads
            threads = []
            for i in range(10):
                thread = threading.Thread(target=detect_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify results
            assert len(errors) == 0
            assert len(results) == 10
            assert all(result[1] == FileType.AUDIO_RAW for result in results)
        finally:
            Path(test_path).unlink()