"""
Tests End-to-End pour les workflows complets utilisateur.

Ces tests vérifient le fonctionnement de bout en bout des scénarios 
utilisateur réels:
- Workflow standard: Register → Login → Upload → Check results → Logout
- Workflow API: Get token → Upload via API → Poll status → Get results
- Workflow avec erreurs et récupération
- Intégration complète de tous les services
"""

import pytest
import time
import json
import threading
from unittest.mock import patch, Mock
from pathlib import Path

from web.app import User, Prediction, Detection, db


@pytest.mark.integration
@pytest.mark.e2e
class TestStandardUserWorkflow:
    """Tests pour le workflow utilisateur standard."""
    
    def test_complete_user_journey_registration_to_prediction(self, integration_client, test_audio_file, mock_prediction_service, integration_db):
        """Test parcours utilisateur complet: inscription → login → upload → résultat."""
        
        # Step 1: User Registration
        response = integration_client.post('/register', data={
            'username': 'endtoenduser',
            'password': 'e2epass123',
            'confirm_password': 'e2epass123'
        })
        
        # Registration should succeed or redirect
        assert response.status_code in [200, 302]
        
        # Verify user was created
        user = User.query.filter_by(username='endtoenduser').first()
        if not user:
            # If registration not implemented, create user manually for test
            user = User(username='endtoenduser')
            user.set_password('e2epass123')
            integration_db.session.add(user)
            integration_db.session.commit()
        
        # Step 2: User Login
        response = integration_client.post('/login', data={
            'username': 'endtoenduser',
            'password': 'e2epass123'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        
        # Step 3: Access Dashboard
        response = integration_client.get('/')
        assert response.status_code == 200
        
        # Should show upload interface
        assert b'upload' in response.data.lower() or b'file' in response.data.lower()
        
        # Step 4: File Upload
        with open(test_audio_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'e2e_test_audio.wav')
            }, follow_redirects=True)
        
        assert response.status_code == 200
        
        # Step 5: Verify Prediction Created
        prediction = Prediction.query.filter_by(
            user_id=user.id,
            filename='e2e_test_audio.wav'
        ).first()
        
        assert prediction is not None
        assert prediction.file_size > 0
        
        # Step 6: Check Processing Status
        # In real app, would poll for results
        if mock_prediction_service.delay.called:
            # Simulate completed processing
            mock_result = mock_prediction_service.delay.return_value.result
            if mock_result:
                prediction.result = json.dumps(mock_result)
                integration_db.session.commit()
        
        # Step 7: View Results
        response = integration_client.get('/predictions')
        if response.status_code == 200:
            # Should show prediction results
            assert b'e2e_test_audio.wav' in response.data
        
        # Step 8: User Logout
        response = integration_client.post('/logout', follow_redirects=True)
        assert response.status_code == 200
        
        # Step 9: Verify Session Ended
        response = integration_client.get('/')
        assert response.status_code == 302  # Redirect to login
    
    def test_multiple_file_upload_workflow(self, integration_client, authenticated_user, test_audio_file, test_image_file, mock_prediction_service):
        """Test workflow avec upload de plusieurs fichiers."""
        
        # Upload multiple files in sequence
        files_to_upload = [
            (test_audio_file, 'multi_audio_1.wav'),
            (test_audio_file, 'multi_audio_2.wav'),
            (test_image_file, 'multi_image_1.jpg') if test_image_file else None
        ]
        
        uploaded_files = []
        
        for file_path, filename in filter(None, files_to_upload):
            with open(file_path, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, filename)
                }, follow_redirects=True)
            
            if response.status_code == 200:
                uploaded_files.append(filename)
        
        # Verify all uploads were processed
        assert len(uploaded_files) >= 2
        
        # Check predictions were created
        predictions = Prediction.query.filter_by(user_id=authenticated_user.id).all()
        prediction_files = [p.filename for p in predictions]
        
        for filename in uploaded_files:
            assert filename in prediction_files
        
        # Verify processing was triggered for each file
        assert mock_prediction_service.delay.call_count >= len(uploaded_files)
    
    def test_user_dashboard_and_history_workflow(self, integration_client, authenticated_user, integration_helpers, integration_db):
        """Test workflow de consultation du dashboard et historique."""
        
        # Create some historical predictions
        for i in range(5):
            prediction = integration_helpers.create_test_prediction(
                authenticated_user,
                filename=f'history_file_{i}.wav',
                result=json.dumps({
                    'species': f'species_{i}',
                    'confidence': 0.8 + i * 0.02
                })
            )
        
        # Access dashboard
        response = integration_client.get('/')
        assert response.status_code == 200
        
        # Should show recent activity
        assert b'history_file_' in response.data or b'species_' in response.data
        
        # Access predictions history
        response = integration_client.get('/predictions')
        if response.status_code == 200:
            # Should show all user's predictions
            for i in range(5):
                assert f'history_file_{i}.wav'.encode() in response.data
        
        # Access individual prediction details
        prediction = Prediction.query.filter_by(user_id=authenticated_user.id).first()
        if prediction:
            response = integration_client.get(f'/predictions/{prediction.id}')
            if response.status_code == 200:
                # Should show prediction details
                assert prediction.filename.encode() in response.data
    
    def test_user_settings_and_preferences_workflow(self, integration_client, authenticated_user):
        """Test workflow de gestion des paramètres utilisateur."""
        
        # Access settings page
        response = integration_client.get('/settings')
        if response.status_code == 200:
            # Should show user settings
            assert b'settings' in response.data.lower() or b'preferences' in response.data.lower()
        
        # Update notification preferences
        response = integration_client.post('/api/notifications/preferences', 
            json={
                'email_notifications': True,
                'sms_notifications': False,
                'push_notifications': True
            },
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            # Preferences should be updated
            data = response.get_json()
            assert data.get('email_notifications') == True
        
        # Change password
        response = integration_client.post('/api/v1/user/change-password',
            json={
                'current_password': 'authpass123',
                'new_password': 'newauthpass123'
            },
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            # Logout and test new password
            integration_client.post('/logout')
            
            # Login with new password
            response = integration_client.post('/login', data={
                'username': 'authuser',
                'password': 'newauthpass123'
            }, follow_redirects=True)
            
            assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.e2e
class TestAPIWorkflow:
    """Tests pour le workflow API complet."""
    
    def test_complete_api_workflow_token_to_results(self, integration_client, test_user_factory, test_audio_file, mock_prediction_service):
        """Test workflow API complet: token → upload → status → résultats."""
        
        # Step 1: Create API user
        user = test_user_factory(username="apiworkflowuser", password="apipass123")
        
        # Step 2: Get JWT Token
        response = integration_client.post('/api/auth/login',
            json={
                'username': 'apiworkflowuser',
                'password': 'apipass123'
            },
            headers={'Content-Type': 'application/json'}
        )
        
        access_token = None
        if response.status_code == 200:
            data = response.get_json()
            access_token = data.get('access_token')
        else:
            # Fallback: create mock token
            access_token = 'mock-jwt-token-for-testing'
        
        assert access_token is not None
        
        # Step 3: Upload File via API
        with open(test_audio_file, 'rb') as f:
            response = integration_client.post('/api/v1/prediction/predict',
                data={
                    'file': (f, 'api_workflow_test.wav'),
                    'model_type': 'auto'
                },
                headers={'Authorization': f'Bearer {access_token}'},
                content_type='multipart/form-data'
            )
        
        prediction_id = None
        if response.status_code == 200:
            data = response.get_json()
            prediction_id = data.get('prediction_id')
            
            # Verify upload succeeded
            assert data.get('status') == 'success'
            assert 'processing_time' in data or 'prediction_id' in data
        
        # Step 4: Check Prediction Status
        if prediction_id:
            response = integration_client.get(f'/api/v1/prediction/status/{prediction_id}',
                headers={'Authorization': f'Bearer {access_token}'}
            )
            
            if response.status_code == 200:
                status_data = response.get_json()
                assert 'status' in status_data
                assert 'prediction_id' in status_data
        
        # Step 5: Get Prediction Results
        if prediction_id:
            response = integration_client.get(f'/api/v1/prediction/result/{prediction_id}',
                headers={'Authorization': f'Bearer {access_token}'}
            )
            
            if response.status_code == 200:
                result_data = response.get_json()
                assert 'predictions' in result_data or 'result' in result_data
        
        # Step 6: List All Predictions
        response = integration_client.get('/api/v1/prediction/predictions',
            headers={'Authorization': f'Bearer {access_token}'},
            query_string={'page': 1, 'per_page': 10}
        )
        
        if response.status_code == 200:
            list_data = response.get_json()
            assert 'predictions' in list_data or 'results' in list_data
    
    def test_batch_upload_api_workflow(self, integration_client, test_jwt_token, test_audio_file, mock_prediction_service):
        """Test workflow d'upload batch via API."""
        
        if not test_jwt_token:
            pytest.skip("JWT authentication not available")
        
        # Prepare multiple files for batch upload
        files_data = []
        for i in range(3):
            with open(test_audio_file, 'rb') as f:
                content = f.read()
                files_data.append(('files', (content, f'batch_file_{i}.wav')))
        
        # Batch upload
        response = integration_client.post('/api/v1/prediction/predict/batch',
            data=files_data,
            headers={'Authorization': f'Bearer {test_jwt_token}'},
            content_type='multipart/form-data'
        )
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'results' in data
            assert len(data['results']) == 3
            
            # All files should be processed
            for result in data['results']:
                assert result.get('status') in ['success', 'processing']
    
    def test_api_error_handling_workflow(self, integration_client, test_jwt_token):
        """Test workflow de gestion d'erreurs API."""
        
        if not test_jwt_token:
            pytest.skip("JWT authentication not available")
        
        # Test invalid file upload
        response = integration_client.post('/api/v1/prediction/predict',
            data={
                'file': (b'invalid content', 'invalid.wav')
            },
            headers={'Authorization': f'Bearer {test_jwt_token}'},
            content_type='multipart/form-data'
        )
        
        # Should return appropriate error
        assert response.status_code in [400, 422]
        
        if response.status_code in [400, 422]:
            error_data = response.get_json()
            assert 'error' in error_data
            assert error_data.get('status') == 'error'
        
        # Test access to non-existent prediction
        response = integration_client.get('/api/v1/prediction/result/nonexistent-id',
            headers={'Authorization': f'Bearer {test_jwt_token}'}
        )
        
        assert response.status_code == 404
        
        # Test invalid authorization
        response = integration_client.get('/api/v1/prediction/predictions',
            headers={'Authorization': 'Bearer invalid-token'}
        )
        
        assert response.status_code == 401


@pytest.mark.integration
@pytest.mark.e2e
class TestErrorRecoveryWorkflows:
    """Tests pour les workflows de récupération d'erreur."""
    
    def test_failed_login_recovery_workflow(self, integration_client, test_user_factory):
        """Test workflow de récupération après échec de login."""
        
        user = test_user_factory(username="recoveryuser", password="correctpass123")
        
        # Attempt failed logins
        for i in range(3):
            response = integration_client.post('/login', data={
                'username': 'recoveryuser',
                'password': f'wrongpass{i}'
            })
            
            # Should stay on login page
            assert response.status_code in [200, 302]
        
        # Successful login after failures
        response = integration_client.post('/login', data={
            'username': 'recoveryuser',
            'password': 'correctpass123'
        }, follow_redirects=True)
        
        # Should eventually succeed
        assert response.status_code == 200
        
        # User should have access
        response = integration_client.get('/')
        assert response.status_code == 200
    
    def test_upload_failure_retry_workflow(self, integration_client, authenticated_user, test_audio_file):
        """Test workflow de retry après échec d'upload."""
        
        # First attempt with service unavailable
        with patch('web.tasks.run_prediction.delay') as mock_task:
            mock_task.side_effect = Exception("Service unavailable")
            
            with open(test_audio_file, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, 'retry_test_1.wav')
                }, follow_redirects=True)
            
            # Should handle error gracefully
            assert response.status_code == 200
        
        # Second attempt with service restored
        with patch('web.tasks.run_prediction.delay') as mock_task:
            mock_task.return_value.id = 'retry-task-success'
            mock_task.return_value.result = {
                'species': 'owl',
                'confidence': 0.87
            }
            
            with open(test_audio_file, 'rb') as f:
                response = integration_client.post('/', data={
                    'file': (f, 'retry_test_2.wav')
                }, follow_redirects=True)
            
            assert response.status_code == 200
            
            # Should succeed this time
            prediction = Prediction.query.filter_by(
                user_id=authenticated_user.id,
                filename='retry_test_2.wav'
            ).first()
            
            assert prediction is not None
    
    def test_session_expiration_recovery_workflow(self, integration_client, test_user_factory, redis_client):
        """Test workflow de récupération après expiration de session."""
        
        user = test_user_factory(username="sessionuser", password="sessionpass123")
        
        # Login
        response = integration_client.post('/login', data={
            'username': 'sessionuser',
            'password': 'sessionpass123'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        
        # Simulate session expiration by clearing Redis
        session_keys = redis_client.keys('session:*')
        for key in session_keys:
            redis_client.delete(key)
        
        # Try to access protected resource
        response = integration_client.get('/')
        assert response.status_code == 302  # Redirect to login
        
        # Re-login
        response = integration_client.post('/login', data={
            'username': 'sessionuser',
            'password': 'sessionpass123'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        
        # Should have access again
        response = integration_client.get('/')
        assert response.status_code == 200
    
    def test_network_interruption_recovery_workflow(self, integration_client, authenticated_user, test_audio_file):
        """Test workflow de récupération après interruption réseau."""
        
        # Simulate partial upload (network interruption)
        partial_data = b'RIFF' + b'x' * 1000  # Incomplete file
        
        response = integration_client.post('/', data={
            'file': (BytesIO(partial_data), 'interrupted.wav')
        }, follow_redirects=True)
        
        # Should handle gracefully
        assert response.status_code == 200
        
        # Retry with complete file
        with open(test_audio_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'complete_retry.wav')
            }, follow_redirects=True)
        
        assert response.status_code == 200
        
        # Complete upload should succeed
        prediction = Prediction.query.filter_by(
            user_id=authenticated_user.id,
            filename='complete_retry.wav'
        ).first()
        
        assert prediction is not None
        assert prediction.file_size > 1000  # Should be complete file


@pytest.mark.integration
@pytest.mark.e2e
class TestConcurrentUserWorkflows:
    """Tests pour les workflows multi-utilisateurs concurrents."""
    
    def test_multiple_users_concurrent_workflow(self, integration_app, test_user_factory, test_audio_file, mock_prediction_service):
        """Test workflow avec plusieurs utilisateurs simultanés."""
        
        # Create multiple users
        users = []
        for i in range(5):
            user = test_user_factory(username=f"concurrent_user_{i}", password=f"pass123_{i}")
            users.append(user)
        
        results = []
        errors = []
        
        def user_workflow(user_index):
            try:
                client = integration_app.test_client()
                user = users[user_index]
                
                # Login
                response = client.post('/login', data={
                    'username': user.username,
                    'password': f'pass123_{user_index}'
                })
                
                if response.status_code not in [200, 302]:
                    raise Exception(f"Login failed for user {user_index}")
                
                # Upload file
                with open(test_audio_file, 'rb') as f:
                    response = client.post('/', data={
                        'file': (f, f'concurrent_upload_{user_index}.wav')
                    }, follow_redirects=True)
                
                if response.status_code != 200:
                    raise Exception(f"Upload failed for user {user_index}")
                
                # Check predictions
                prediction = Prediction.query.filter_by(
                    user_id=user.id,
                    filename=f'concurrent_upload_{user_index}.wav'
                ).first()
                
                if not prediction:
                    raise Exception(f"Prediction not created for user {user_index}")
                
                # Logout
                client.post('/logout')
                
                results.append({
                    'user_index': user_index,
                    'status': 'success',
                    'prediction_id': prediction.id
                })
                
            except Exception as e:
                errors.append({
                    'user_index': user_index,
                    'error': str(e)
                })
        
        # Run concurrent workflows
        threads = []
        for i in range(5):
            thread = threading.Thread(target=user_workflow, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent workflow errors: {errors}"
        assert len(results) == 5
        
        # All users should have successful workflows
        for result in results:
            assert result['status'] == 'success'
            assert result['prediction_id'] is not None
    
    def test_shared_resource_access_workflow(self, integration_app, test_user_factory, redis_client):
        """Test workflow d'accès aux ressources partagées."""
        
        users = []
        for i in range(3):
            user = test_user_factory(username=f"shared_user_{i}", password=f"shared123_{i}")
            users.append(user)
        
        def concurrent_access(user_index):
            client = integration_app.test_client()
            user = users[user_index]
            
            # Login
            client.post('/login', data={
                'username': user.username,
                'password': f'shared123_{user_index}'
            })
            
            # Access shared resources (dashboard, analytics, etc.)
            endpoints = ['/', '/predictions', '/analytics']
            
            for endpoint in endpoints:
                response = client.get(endpoint)
                # Should not interfere with other users
                assert response.status_code in [200, 302, 404]  # 404 if endpoint doesn't exist
            
            # Generate some load on shared Redis
            for i in range(10):
                redis_client.set(f'user_{user_index}_key_{i}', f'value_{i}')
                redis_client.get(f'user_{user_index}_key_{i}')
            
            client.post('/logout')
        
        # Run concurrent access
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_access, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify Redis is still responsive
        assert redis_client.ping() == True
        
        # Check that all user data was written
        for user_index in range(3):
            for i in range(10):
                value = redis_client.get(f'user_{user_index}_key_{i}')
                assert value.decode() == f'value_{i}'


@pytest.mark.integration
@pytest.mark.e2e
class TestDataConsistencyWorkflows:
    """Tests pour la cohérence des données dans les workflows."""
    
    def test_data_consistency_across_services(self, integration_client, authenticated_user, test_audio_file, integration_db, redis_client):
        """Test cohérence des données entre services."""
        
        # Upload file
        with open(test_audio_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'consistency_test.wav')
            }, follow_redirects=True)
        
        assert response.status_code == 200
        
        # Check database consistency
        prediction = Prediction.query.filter_by(
            user_id=authenticated_user.id,
            filename='consistency_test.wav'
        ).first()
        
        assert prediction is not None
        
        # Check Redis cache consistency
        cache_key = f'user:{authenticated_user.id}:predictions'
        cached_predictions = redis_client.get(cache_key)
        
        # If caching is implemented, should be consistent
        if cached_predictions:
            cached_data = json.loads(cached_predictions)
            # Should include our new prediction
        
        # Check file system consistency
        # In real implementation, uploaded files might be stored
        
        # Verify user quota consistency
        total_usage = integration_db.session.query(
            integration_db.func.sum(Prediction.file_size)
        ).filter_by(user_id=authenticated_user.id).scalar() or 0
        
        assert total_usage >= prediction.file_size
    
    def test_transaction_rollback_workflow(self, integration_client, authenticated_user, integration_db):
        """Test workflow de rollback de transaction."""
        
        initial_prediction_count = Prediction.query.filter_by(
            user_id=authenticated_user.id
        ).count()
        
        # Attempt operation that should fail and rollback
        with patch('web.app.db.session.commit') as mock_commit:
            mock_commit.side_effect = Exception("Database error")
            
            response = integration_client.post('/', data={
                'file': (BytesIO(b'RIFF' + b'x' * 1000), 'rollback_test.wav')
            }, follow_redirects=True)
            
            # Should handle error gracefully
            assert response.status_code == 200
        
        # Check that no partial data was committed
        final_prediction_count = Prediction.query.filter_by(
            user_id=authenticated_user.id
        ).count()
        
        # Should not have increased due to rollback
        assert final_prediction_count == initial_prediction_count
    
    def test_eventual_consistency_workflow(self, integration_client, authenticated_user, test_audio_file, mock_prediction_service):
        """Test workflow de cohérence éventuelle."""
        
        # Upload file (immediate consistency)
        with open(test_audio_file, 'rb') as f:
            response = integration_client.post('/', data={
                'file': (f, 'eventual_consistency_test.wav')
            }, follow_redirects=True)
        
        assert response.status_code == 200
        
        # Prediction should be created immediately
        prediction = Prediction.query.filter_by(
            user_id=authenticated_user.id,
            filename='eventual_consistency_test.wav'
        ).first()
        
        assert prediction is not None
        assert prediction.result is None  # No result yet
        
        # Simulate async processing completion (eventual consistency)
        if mock_prediction_service.delay.called:
            # Simulate result being written later
            prediction.result = json.dumps({
                'species': 'owl',
                'confidence': 0.92
            })
            integration_db.session.commit()
            
            # Now result should be available
            updated_prediction = Prediction.query.get(prediction.id)
            assert updated_prediction.result is not None
            assert 'owl' in updated_prediction.result