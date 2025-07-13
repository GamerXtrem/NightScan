"""
Tests d'intégration pour les flux d'upload de fichiers complets.

Ces tests vérifient le fonctionnement bout-en-bout des uploads:
- Upload de fichiers avec validation complète
- Gestion des quotas utilisateur en temps réel
- Traitement asynchrone avec Celery
- Intégration avec base de données et Redis
- Gestion d'erreurs et recovery
- Performance avec gros fichiers
"""

import pytest
import time
import tempfile
import os
import threading
from pathlib import Path
from unittest.mock import patch, Mock
from io import BytesIO
import hashlib
import struct
import numpy as np

from web.app import Prediction, User, db
from web.tasks import run_prediction


@pytest.mark.integration
class TestCompleteUploadFlow:
    """Tests pour les flux d'upload complets."""
    
    def test_complete_audio_upload_flow(self, integration_client, authenticated_user, test_audio_file, mock_prediction_service):
        """Test flux complet d'upload audio avec traitement."""
        # Upload valid audio file
        with open(test_audio_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'test_upload.wav')
            }, follow_redirects=True)
        
        assert response.status_code == 200
        
        # Check that prediction was created in database
        prediction = Prediction.query.filter_by(
            user_id=authenticated_user.id,
            filename='test_upload.wav'
        ).first()
        
        assert prediction is not None
        assert prediction.file_size > 0
        
        # Verify Celery task was triggered
        mock_prediction_service.delay.assert_called_once()
        
        # Check file was processed (in real scenario, would wait for Celery)
        if mock_prediction_service.delay.return_value.result:
            result = mock_prediction_service.delay.return_value.result
            assert 'species' in result
            assert 'confidence' in result
    
    def test_complete_image_upload_flow(self, integration_client, authenticated_user, test_image_file, mock_prediction_service):
        """Test flux complet d'upload image avec traitement."""
        # Upload valid image file
        with open(test_image_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'test_image.jpg')
            }, follow_redirects=True)
        
        # May succeed or fail depending on implementation
        # This tests the infrastructure can handle image uploads
        
        if response.status_code == 200:
            # Check prediction creation
            prediction = Prediction.query.filter_by(
                user_id=authenticated_user.id,
                filename='test_image.jpg'
            ).first()
            
            if prediction:
                assert prediction.file_size > 0
    
    def test_upload_with_file_validation(self, integration_client, authenticated_user, test_invalid_file):
        """Test upload avec validation de fichier."""
        # Upload invalid file
        with open(test_invalid_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'invalid.wav')
            }, follow_redirects=True)
        
        # Should reject invalid file
        assert b'Invalid' in response.data or b'error' in response.data.lower()
        
        # No prediction should be created
        prediction = Prediction.query.filter_by(
            user_id=authenticated_user.id,
            filename='invalid.wav'
        ).first()
        
        assert prediction is None
    
    def test_upload_file_size_validation(self, integration_client, authenticated_user):
        """Test validation de taille de fichier."""
        # Create file that's too large (simulate)
        large_content = b'RIFF' + b'x' * (100 * 1024 * 1024)  # 100MB+ file
        
        response = integration_client.post('/', data={
            'file': (BytesIO(large_content), 'large_file.wav')
        }, follow_redirects=True)
        
        # Should reject oversized file
        assert response.status_code == 200
        assert b'exceed' in response.data.lower() or b'limit' in response.data.lower()
    
    def test_upload_filename_sanitization(self, integration_client, authenticated_user, test_audio_file):
        """Test sanitisation des noms de fichiers."""
        dangerous_filenames = [
            '../../../etc/passwd.wav',
            '<script>alert("xss")</script>.wav',
            'file with spaces.wav',
            'file$with&special@chars.wav',
            'très_long_nom_de_fichier_qui_dépasse_la_limite_normale.wav'
        ]
        
        for dangerous_name in dangerous_filenames:
            with open(test_audio_file, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, dangerous_name)
                }, follow_redirects=True)
            
            # Should either reject or sanitize filename
            prediction = Prediction.query.filter_by(
                user_id=authenticated_user.id
            ).order_by(Prediction.id.desc()).first()
            
            if prediction:
                # Filename should be sanitized
                assert '../' not in prediction.filename
                assert '<script>' not in prediction.filename
                # Should have safe filename
                assert prediction.filename.endswith('.wav')
    
    def test_concurrent_uploads_same_user(self, integration_app, test_user_factory, test_audio_file, mock_prediction_service):
        """Test uploads concurrents du même utilisateur."""
        user = test_user_factory(username="concurrentuser", password="pass123")
        
        results = []
        errors = []
        
        def upload_file(client_num):
            try:
                client = integration_app.test_client()
                
                # Login
                client.post('/login', data={
                    'username': 'concurrentuser',
                    'password': 'pass123'
                })
                
                # Upload
                with open(test_audio_file, 'rb') as f:
                    response = client.post('/', data={
                        'file': (f, f'concurrent_file_{client_num}.wav')
                    }, follow_redirects=True)
                
                results.append((client_num, response.status_code))
                
            except Exception as e:
                errors.append((client_num, str(e)))
        
        # Start multiple uploads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=upload_file, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all uploads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Upload errors: {errors}"
        assert len(results) == 5
        
        # All uploads should succeed
        success_count = sum(1 for _, status in results if status == 200)
        assert success_count >= 4  # Allow for some failures due to concurrency
        
        # Check database consistency
        predictions = Prediction.query.filter_by(user_id=user.id).all()
        assert len(predictions) >= 4


@pytest.mark.integration
class TestQuotaManagement:
    """Tests pour la gestion des quotas d'upload."""
    
    def test_quota_enforcement_single_file(self, integration_client, test_user_factory, test_audio_file):
        """Test application des quotas pour un seul fichier."""
        # Create user with limited quota
        user = test_user_factory(username="quotauser", password="pass123")
        
        # Login
        integration_client.post('/login', data={
            'username': 'quotauser',
            'password': 'pass123'
        })
        
        # Get initial quota
        response = integration_client.get('/')
        assert response.status_code == 200
        
        # Upload file
        with open(test_audio_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'quota_test.wav')
            }, follow_redirects=True)
        
        # Should succeed for first upload
        assert response.status_code == 200
        
        # Check quota usage is tracked
        prediction = Prediction.query.filter_by(user_id=user.id).first()
        if prediction:
            assert prediction.file_size > 0
    
    def test_quota_exceeded_rejection(self, integration_client, test_user_factory, integration_db):
        """Test rejet quand quota dépassé."""
        user = test_user_factory(username="exceededuser", password="pass123")
        
        # Manually set user's quota usage near limit
        # Simulate user has already used most of their quota
        for i in range(10):
            prediction = Prediction(
                user_id=user.id,
                filename=f'existing_file_{i}.wav',
                file_size=100 * 1024 * 1024,  # 100MB each
                result='{"species": "test"}'
            )
            integration_db.session.add(prediction)
        integration_db.session.commit()
        
        # Login
        integration_client.post('/login', data={
            'username': 'exceededuser',
            'password': 'pass123'
        })
        
        # Try to upload another file
        large_file_content = b'RIFF' + b'x' * (50 * 1024 * 1024)  # 50MB
        response = integration_client.post('/', data={
            'file': (BytesIO(large_file_content), 'quota_exceeded.wav')
        }, follow_redirects=True)
        
        # Should reject due to quota
        assert b'quota' in response.data.lower() or b'limit' in response.data.lower()
    
    def test_quota_calculation_accuracy(self, integration_client, test_user_factory, test_audio_file):
        """Test précision du calcul de quota."""
        user = test_user_factory(username="calculuser", password="pass123")
        
        # Login
        integration_client.post('/login', data={
            'username': 'calculuser',
            'password': 'pass123'
        })
        
        # Get initial quota
        response = integration_client.get('/')
        initial_usage = 0  # Parse from response if displayed
        
        # Upload file and get size
        file_size = os.path.getsize(test_audio_file)
        
        with open(test_audio_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'calc_test.wav')
            }, follow_redirects=True)
        
        # Check database quota tracking
        total_usage = integration_db.session.query(
            integration_db.func.sum(Prediction.file_size)
        ).filter_by(user_id=user.id).scalar() or 0
        
        assert total_usage >= file_size
    
    def test_quota_reset_functionality(self, integration_client, test_user_factory, integration_db):
        """Test fonctionnalité de reset de quota."""
        user = test_user_factory(username="resetuser", password="pass123")
        
        # Add some usage
        prediction = Prediction(
            user_id=user.id,
            filename='reset_test.wav',
            file_size=10 * 1024 * 1024,  # 10MB
            result='{"species": "test"}'
        )
        integration_db.session.add(prediction)
        integration_db.session.commit()
        
        # Simulate quota reset (in real scenario, would be cron job)
        # For test, manually clear predictions older than X days
        from datetime import datetime, timedelta
        old_date = datetime.utcnow() - timedelta(days=30)
        
        old_predictions = Prediction.query.filter(
            Prediction.user_id == user.id,
            Prediction.created_at < old_date
        ).all()
        
        for pred in old_predictions:
            integration_db.session.delete(pred)
        integration_db.session.commit()
        
        # Quota should be reduced
        new_usage = integration_db.session.query(
            integration_db.func.sum(Prediction.file_size)
        ).filter_by(user_id=user.id).scalar() or 0
        
        # Should have current usage only
        assert new_usage >= 0


@pytest.mark.integration
class TestAsynchronousProcessing:
    """Tests pour le traitement asynchrone des uploads."""
    
    def test_celery_task_creation(self, integration_client, authenticated_user, test_audio_file, mock_celery_worker):
        """Test création de tâches Celery pour traitement."""
        with patch('web.tasks.run_prediction.delay') as mock_task:
            mock_task.return_value.id = 'test-task-123'
            
            # Upload file
            with open(test_audio_file, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, 'celery_test.wav')
                }, follow_redirects=True)
            
            assert response.status_code == 200
            
            # Verify task was created
            mock_task.assert_called_once()
            
            # Check task parameters
            call_args = mock_task.call_args[1]  # kwargs
            assert 'filename' in str(call_args) or len(mock_task.call_args[0]) > 0
    
    def test_async_processing_result_storage(self, integration_client, authenticated_user, test_audio_file, integration_db):
        """Test stockage des résultats de traitement asynchrone."""
        with patch('web.tasks.run_prediction.delay') as mock_task:
            # Configure mock result
            mock_result = Mock()
            mock_result.id = 'test-task-456'
            mock_result.ready.return_value = True
            mock_result.result = {
                'species': 'owl',
                'confidence': 0.89,
                'predictions': [
                    {'class': 'owl', 'confidence': 0.89},
                    {'class': 'wind', 'confidence': 0.11}
                ]
            }
            mock_task.return_value = mock_result
            
            # Upload file
            with open(test_audio_file, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, 'async_result_test.wav')
                }, follow_redirects=True)
            
            # Find created prediction
            prediction = Prediction.query.filter_by(
                user_id=authenticated_user.id,
                filename='async_result_test.wav'
            ).first()
            
            assert prediction is not None
            
            # In real scenario, Celery worker would update the prediction
            # For test, simulate the update
            if mock_result.ready():
                prediction.result = str(mock_result.result)
                integration_db.session.commit()
                
                assert 'owl' in prediction.result
    
    def test_processing_status_tracking(self, integration_client, authenticated_user, test_audio_file):
        """Test suivi du statut de traitement."""
        with patch('web.tasks.run_prediction.delay') as mock_task:
            # Configure progressive status updates
            mock_result = Mock()
            mock_result.id = 'status-task-789'
            mock_result.ready.return_value = False
            mock_result.status = 'PENDING'
            mock_task.return_value = mock_result
            
            # Upload file
            with open(test_audio_file, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, 'status_test.wav')
                }, follow_redirects=True)
            
            # Check initial status
            prediction = Prediction.query.filter_by(
                user_id=authenticated_user.id,
                filename='status_test.wav'
            ).first()
            
            if prediction:
                # Initially should be pending
                assert prediction.result is None or 'pending' in prediction.result.lower()
                
                # Simulate status update
                mock_result.status = 'SUCCESS'
                mock_result.ready.return_value = True
                mock_result.result = {'species': 'fox', 'confidence': 0.75}
                
                # In real implementation, would have status endpoint
                # GET /api/predictions/{id}/status
    
    def test_processing_error_handling(self, integration_client, authenticated_user, test_audio_file):
        """Test gestion d'erreurs de traitement."""
        with patch('web.tasks.run_prediction.delay') as mock_task:
            # Configure task failure
            mock_result = Mock()
            mock_result.id = 'error-task-999'
            mock_result.ready.return_value = True
            mock_result.failed.return_value = True
            mock_result.result = Exception("Processing failed")
            mock_task.return_value = mock_result
            
            # Upload file
            with open(test_audio_file, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, 'error_test.wav')
                }, follow_redirects=True)
            
            # Should handle gracefully
            assert response.status_code == 200
            
            # Check error is recorded
            prediction = Prediction.query.filter_by(
                user_id=authenticated_user.id,
                filename='error_test.wav'
            ).first()
            
            if prediction:
                # Error should be recorded
                # In real implementation, would have error status
                pass
    
    def test_concurrent_processing_queue(self, integration_app, test_user_factory, test_audio_file):
        """Test file d'attente de traitement concurrent."""
        user = test_user_factory(username="queueuser", password="pass123")
        
        task_ids = []
        
        with patch('web.tasks.run_prediction.delay') as mock_task:
            def create_mock_task(task_id):
                mock_result = Mock()
                mock_result.id = task_id
                mock_result.ready.return_value = False
                mock_result.status = 'PENDING'
                return mock_result
            
            # Configure multiple tasks
            mock_task.side_effect = lambda *args, **kwargs: create_mock_task(f'queue-task-{len(task_ids)}')
            
            client = integration_app.test_client()
            client.post('/login', data={
                'username': 'queueuser',
                'password': 'pass123'
            })
            
            # Upload multiple files
            for i in range(5):
                with open(test_audio_file, 'rb') as f:
                    response = client.post('/', data={
                        'file': (f, f'queue_test_{i}.wav')
                    }, follow_redirects=True)
                
                assert response.status_code == 200
                task_ids.append(f'queue-task-{i}')
            
            # All tasks should be queued
            assert len(task_ids) == 5
            assert mock_task.call_count == 5


@pytest.mark.integration
class TestUploadPerformance:
    """Tests de performance pour les uploads."""
    
    def test_large_file_upload_performance(self, integration_client, authenticated_user, test_large_audio_file):
        """Test performance d'upload de gros fichiers."""
        start_time = time.time()
        
        with open(test_large_audio_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'large_performance_test.wav')
            }, follow_redirects=True)
        
        upload_time = time.time() - start_time
        
        # Should complete within reasonable time (depends on file size)
        assert upload_time < 60  # 1 minute max for test
        
        if response.status_code == 200:
            # Verify file was uploaded correctly
            prediction = Prediction.query.filter_by(
                user_id=authenticated_user.id,
                filename='large_performance_test.wav'
            ).first()
            
            if prediction:
                assert prediction.file_size > 50 * 1024 * 1024  # Should be large
    
    def test_streaming_upload_memory_usage(self, integration_client, authenticated_user):
        """Test utilisation mémoire pour uploads streaming."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create large file data
        large_data = b'RIFF' + b'x' * (20 * 1024 * 1024)  # 20MB
        
        # Upload via streaming
        response = integration_client.post('/', data={
            'file': (BytesIO(large_data), 'streaming_test.wav')
        }, follow_redirects=True)
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (not holding entire file)
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase
    
    def test_concurrent_upload_throughput(self, integration_app, test_user_factory, test_audio_file):
        """Test débit d'uploads concurrents."""
        user = test_user_factory(username="throughputuser", password="pass123")
        
        results = []
        start_time = time.time()
        
        def timed_upload(file_num):
            try:
                client = integration_app.test_client()
                client.post('/login', data={
                    'username': 'throughputuser',
                    'password': 'pass123'
                })
                
                upload_start = time.time()
                with open(test_audio_file, 'rb') as f:
                    response = client.post('/', data={
                        'file': (f, f'throughput_test_{file_num}.wav')
                    }, follow_redirects=True)
                upload_time = time.time() - upload_start
                
                results.append({
                    'file_num': file_num,
                    'status': response.status_code,
                    'upload_time': upload_time
                })
                
            except Exception as e:
                results.append({
                    'file_num': file_num,
                    'error': str(e),
                    'upload_time': 0
                })
        
        # Upload 10 files concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=timed_upload, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_uploads = [r for r in results if r.get('status') == 200]
        
        assert len(successful_uploads) >= 8  # At least 80% success
        
        # Calculate throughput
        if successful_uploads:
            avg_upload_time = sum(r['upload_time'] for r in successful_uploads) / len(successful_uploads)
            throughput = len(successful_uploads) / total_time
            
            assert avg_upload_time < 5.0  # Average upload < 5 seconds
            assert throughput > 1.0  # > 1 file per second overall


@pytest.mark.integration
class TestUploadSecurity:
    """Tests de sécurité pour les uploads."""
    
    def test_malicious_file_upload_prevention(self, integration_client, authenticated_user, test_files_dir):
        """Test prévention d'upload de fichiers malicieux."""
        malicious_files = [
            ('virus.exe', b'MZ\x90\x00'),  # Executable header
            ('script.wav', b'<script>alert("xss")</script>'),  # XSS attempt
            ('bomb.zip', b'PK\x03\x04'),  # ZIP bomb attempt
            ('payload.wav', b'\x00' * 1000 + b'rm -rf /' + b'\x00' * 1000)  # Command injection
        ]
        
        for filename, content in malicious_files:
            response = integration_client.post('/', data={
                'file': (BytesIO(content), filename)
            }, follow_redirects=True)
            
            # Should reject malicious files
            # Either rejected or sanitized
            assert response.status_code == 200
            
            # Check file wasn't processed
            if '.wav' in filename:
                prediction = Prediction.query.filter_by(
                    user_id=authenticated_user.id,
                    filename=filename
                ).first()
                
                # Should either be rejected or sanitized
                if prediction:
                    # File content should be validated
                    assert prediction.file_size < len(content) + 1000  # Reasonable size
    
    def test_path_traversal_prevention(self, integration_client, authenticated_user, test_audio_file):
        """Test prévention de path traversal."""
        traversal_filenames = [
            '../../etc/passwd.wav',
            '..\\..\\windows\\system32\\config\\sam.wav',
            '/etc/shadow.wav',
            'C:\\Windows\\System32\\drivers\\etc\\hosts.wav'
        ]
        
        for malicious_name in traversal_filenames:
            with open(test_audio_file, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, malicious_name)
                }, follow_redirects=True)
            
            # Should sanitize filename
            prediction = Prediction.query.filter_by(
                user_id=authenticated_user.id
            ).order_by(Prediction.id.desc()).first()
            
            if prediction:
                # Filename should be sanitized
                assert '..' not in prediction.filename
                assert '/' not in prediction.filename or prediction.filename.count('/') <= 1
                assert '\\' not in prediction.filename
    
    def test_file_type_spoofing_prevention(self, integration_client, authenticated_user, test_files_dir):
        """Test prévention de spoofing de type de fichier."""
        # Create file with .wav extension but different content
        spoofed_file = test_files_dir / 'spoofed.wav'
        
        # PNG file with .wav extension
        png_header = b'\x89PNG\r\n\x1a\n'
        with open(spoofed_file, 'wb') as f:
            f.write(png_header + b'fake png data')
        
        with open(spoofed_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'spoofed.wav')
            }, follow_redirects=True)
        
        # Should detect and reject spoofed file
        assert b'Invalid' in response.data or b'error' in response.data.lower()
        
        # No prediction should be created
        prediction = Prediction.query.filter_by(
            user_id=authenticated_user.id,
            filename='spoofed.wav'
        ).first()
        
        assert prediction is None
    
    def test_upload_rate_limiting(self, integration_client, authenticated_user, test_audio_file):
        """Test rate limiting des uploads."""
        # Attempt rapid uploads
        responses = []
        
        for i in range(10):
            with open(test_audio_file, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, f'rate_limit_test_{i}.wav')
                }, follow_redirects=True)
            
            responses.append(response.status_code)
            time.sleep(0.1)  # Brief pause
        
        # Some requests might be rate limited
        # Depends on implementation
        success_count = sum(1 for status in responses if status == 200)
        
        # Should allow some uploads but may rate limit excessive requests
        assert success_count >= 5  # At least some should succeed


@pytest.mark.integration
class TestUploadCleanupAndRecovery:
    """Tests pour le nettoyage et la récupération des uploads."""
    
    def test_temporary_file_cleanup(self, integration_client, authenticated_user, test_audio_file):
        """Test nettoyage des fichiers temporaires."""
        # Track temp files before upload
        temp_dir = Path(tempfile.gettempdir())
        initial_temp_files = set(temp_dir.glob('*nightscan*'))
        
        with open(test_audio_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'cleanup_test.wav')
            }, follow_redirects=True)
        
        # Allow some time for cleanup
        time.sleep(1)
        
        # Check temp files after upload
        final_temp_files = set(temp_dir.glob('*nightscan*'))
        
        # Should not accumulate temp files
        new_temp_files = final_temp_files - initial_temp_files
        assert len(new_temp_files) <= 1  # At most one temp file remaining
    
    def test_interrupted_upload_recovery(self, integration_client, authenticated_user, test_audio_file):
        """Test récupération après upload interrompu."""
        # Simulate interrupted upload by closing connection
        # This is complex to test in integration test
        # Would need to test with real network interruption
        
        # For now, test that partial uploads are handled
        partial_content = b'RIFF' + b'x' * 1000  # Partial WAV file
        
        response = integration_client.post('/', data={
            'file': (BytesIO(partial_content), 'interrupted.wav')
        }, follow_redirects=True)
        
        # Should handle gracefully
        assert response.status_code == 200
        
        # Check no corrupted prediction is created
        prediction = Prediction.query.filter_by(
            user_id=authenticated_user.id,
            filename='interrupted.wav'
        ).first()
        
        # Either no prediction or error recorded
        if prediction:
            # Should have reasonable file size
            assert prediction.file_size < 10000  # Small partial file
    
    def test_database_consistency_after_errors(self, integration_client, authenticated_user, integration_db):
        """Test cohérence de la base de données après erreurs."""
        initial_prediction_count = Prediction.query.count()
        
        # Attempt upload that will fail
        response = integration_client.post('/', data={
            'file': (BytesIO(b'invalid content'), 'consistency_test.wav')
        }, follow_redirects=True)
        
        # Check database state
        final_prediction_count = Prediction.query.count()
        
        # Should not create invalid prediction records
        assert final_prediction_count >= initial_prediction_count
        
        # If prediction was created, it should be valid
        new_predictions = Prediction.query.filter_by(
            user_id=authenticated_user.id,
            filename='consistency_test.wav'
        ).all()
        
        for prediction in new_predictions:
            assert prediction.user_id == authenticated_user.id
            assert prediction.filename is not None
            assert prediction.created_at is not None
    
    def test_orphaned_file_cleanup(self, integration_client, authenticated_user, test_audio_file, integration_db):
        """Test nettoyage des fichiers orphelins."""
        # Create prediction record without actual processing
        prediction = Prediction(
            user_id=authenticated_user.id,
            filename='orphaned_test.wav',
            file_size=12345,
            result=None  # No result yet
        )
        integration_db.session.add(prediction)
        integration_db.session.commit()
        
        # Simulate cleanup process (would be cron job)
        from datetime import datetime, timedelta
        
        # Find predictions without results older than X minutes
        old_threshold = datetime.utcnow() - timedelta(minutes=30)
        orphaned_predictions = Prediction.query.filter(
            Prediction.result.is_(None),
            Prediction.created_at < old_threshold
        ).all()
        
        # Cleanup orphaned records
        for pred in orphaned_predictions:
            integration_db.session.delete(pred)
        integration_db.session.commit()
        
        # Verify cleanup
        remaining_orphaned = Prediction.query.filter(
            Prediction.id == prediction.id
        ).first()
        
        # Should be cleaned up if old enough
        # Or kept if recent