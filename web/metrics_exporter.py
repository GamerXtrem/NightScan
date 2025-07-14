#!/usr/bin/env python3
"""
Exporter métriques Prometheus pour NightScan
Expose les métriques applicatives pour monitoring
"""

import time
import threading
from flask import Flask, Response
from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    generate_latest, CONTENT_TYPE_LATEST
)
import psutil
import redis
import psycopg2
from datetime import datetime
import os

# Métriques NightScan
nightscan_info = Info('nightscan_app_info', 'Information application NightScan')
nightscan_requests_total = Counter(
    'nightscan_http_requests_total',
    'Total requêtes HTTP',
    ['method', 'endpoint', 'status']
)
nightscan_request_duration = Histogram(
    'nightscan_http_request_duration_seconds',
    'Durée requêtes HTTP',
    ['method', 'endpoint']
)
nightscan_uploads_total = Counter(
    'nightscan_uploads_total',
    'Total uploads fichiers',
    ['file_type', 'status']
)
nightscan_predictions_total = Counter(
    'nightscan_predictions_total',
    'Total prédictions ML',
    ['model_type', 'status']
)
nightscan_ml_queue_length = Gauge(
    'nightscan_ml_queue_length',
    'Longueur queue prédictions ML'
)
nightscan_ml_model_accuracy = Gauge(
    'nightscan_ml_model_accuracy',
    'Précision modèles ML',
    ['model_name']
)
nightscan_active_users = Gauge(
    'nightscan_active_users',
    'Utilisateurs actifs'
)
nightscan_database_connections = Gauge(
    'nightscan_database_connections',
    'Connexions base données actives'
)
nightscan_redis_memory_usage = Gauge(
    'nightscan_redis_memory_usage_bytes',
    'Utilisation mémoire Redis'
)

class NightScanMetricsExporter:
    """Exporter métriques personnalisées NightScan"""
    
    def __init__(self):
        self.redis_client = None
        self.db_conn = None
        self.setup_connections()
        self.setup_app_info()
        
        # Thread collecte métriques
        self.metrics_thread = threading.Thread(target=self.collect_metrics_loop)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
    
    def setup_connections(self):
        """Configurer connexions Redis et PostgreSQL"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
        except Exception as e:
            print(f"Erreur connexion Redis: {e}")
            self.redis_client = None
        
        try:
            db_url = os.getenv('SQLALCHEMY_DATABASE_URI', 
                             'postgresql://user:pass@localhost/nightscan')
            self.db_conn = psycopg2.connect(db_url)
        except Exception as e:
            print(f"Erreur connexion PostgreSQL: {e}")
            self.db_conn = None
    
    def setup_app_info(self):
        """Configurer informations application"""
        nightscan_info.info({
            'version': '1.0.0',
            'environment': os.getenv('NIGHTSCAN_ENV', 'development'),
            'build_date': datetime.now().isoformat(),
            'python_version': psutil.sys.version.split()[0]
        })
    
    def collect_system_metrics(self):
        """Collecter métriques système"""
        try:
            # Utilisation mémoire
            memory = psutil.virtual_memory()
            nightscan_system_memory_usage = Gauge(
                'nightscan_system_memory_usage_percent',
                'Utilisation mémoire système'
            )
            nightscan_system_memory_usage.set(memory.percent)
            
            # Utilisation CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            nightscan_system_cpu_usage = Gauge(
                'nightscan_system_cpu_usage_percent',
                'Utilisation CPU système'
            )
            nightscan_system_cpu_usage.set(cpu_percent)
            
            # Espace disque
            disk = psutil.disk_usage('/')
            nightscan_system_disk_usage = Gauge(
                'nightscan_system_disk_usage_percent',
                'Utilisation disque système'
            )
            nightscan_system_disk_usage.set(
                (disk.used / disk.total) * 100
            )
            
        except Exception as e:
            print(f"Erreur collecte métriques système: {e}")
    
    def collect_redis_metrics(self):
        """Collecter métriques Redis"""
        if not self.redis_client:
            return
            
        try:
            info = self.redis_client.info()
            
            # Utilisation mémoire
            nightscan_redis_memory_usage.set(info.get('used_memory', 0))
            
            # Connexions
            redis_connections = Gauge(
                'nightscan_redis_connections',
                'Connexions Redis actives'
            )
            redis_connections.set(info.get('connected_clients', 0))
            
            # Hits/misses cache
            redis_hits = Counter(
                'nightscan_redis_hits_total',
                'Total hits cache Redis'
            )
            redis_misses = Counter(
                'nightscan_redis_misses_total',
                'Total misses cache Redis'
            )
            
            redis_hits._value._value = info.get('keyspace_hits', 0)
            redis_misses._value._value = info.get('keyspace_misses', 0)
            
        except Exception as e:
            print(f"Erreur collecte métriques Redis: {e}")
    
    def collect_database_metrics(self):
        """Collecter métriques base de données"""
        if not self.db_conn:
            return
            
        try:
            cursor = self.db_conn.cursor()
            
            # Connexions actives
            cursor.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
            )
            active_connections = cursor.fetchone()[0]
            nightscan_database_connections.set(active_connections)
            
            # Taille base données
            cursor.execute(
                "SELECT pg_database_size(current_database())"
            )
            db_size = cursor.fetchone()[0]
            db_size_gauge = Gauge(
                'nightscan_database_size_bytes',
                'Taille base données'
            )
            db_size_gauge.set(db_size)
            
            # Requêtes lentes
            cursor.execute("""
                SELECT count(*) FROM pg_stat_activity 
                WHERE state = 'active' 
                AND now() - query_start > interval '5 seconds'
            """)
            slow_queries = cursor.fetchone()[0]
            slow_queries_gauge = Gauge(
                'nightscan_database_slow_queries',
                'Requêtes lentes (>5s)'
            )
            slow_queries_gauge.set(slow_queries)
            
            cursor.close()
            
        except Exception as e:
            print(f"Erreur collecte métriques DB: {e}")
            # Reconnexion si nécessaire
            try:
                self.db_conn = psycopg2.connect(
                    os.getenv('SQLALCHEMY_DATABASE_URI')
                )
            except:
                pass
    
    def collect_ml_metrics(self):
        """Collecter métriques ML spécifiques"""
        try:
            # Simuler métriques ML (à adapter selon implémentation)
            
            # Queue predictions (depuis Redis si utilisé)
            if self.redis_client:
                queue_length = self.redis_client.llen('ml_prediction_queue')
                nightscan_ml_queue_length.set(queue_length or 0)
            
            # Précision modèles (valeurs exemples)
            models = ['audio_light', 'audio_heavy', 'photo_light', 'photo_heavy']
            accuracies = [0.95, 0.97, 0.94, 0.96]  # À récupérer des vrais modèles
            
            for model, accuracy in zip(models, accuracies):
                nightscan_ml_model_accuracy.labels(model_name=model).set(accuracy)
            
        except Exception as e:
            print(f"Erreur collecte métriques ML: {e}")
    
    def collect_business_metrics(self):
        """Collecter métriques business"""
        try:
            if not self.db_conn:
                return
            
            cursor = self.db_conn.cursor()
            
            # Utilisateurs actifs (connectés dernière heure)
            cursor.execute("""
                SELECT count(DISTINCT user_id) 
                FROM user_sessions 
                WHERE last_activity > now() - interval '1 hour'
            """)
            result = cursor.fetchone()
            if result:
                nightscan_active_users.set(result[0])
            
            # Uploads réussis aujourd'hui
            cursor.execute("""
                SELECT file_type, count(*) 
                FROM uploads 
                WHERE created_at::date = current_date 
                AND status = 'success'
                GROUP BY file_type
            """)
            for file_type, count in cursor.fetchall():
                nightscan_uploads_total.labels(
                    file_type=file_type, 
                    status='success'
                )._value._value = count
            
            # Prédictions réussies aujourd'hui
            cursor.execute("""
                SELECT model_type, count(*) 
                FROM predictions 
                WHERE created_at::date = current_date 
                AND status = 'completed'
                GROUP BY model_type
            """)
            for model_type, count in cursor.fetchall():
                nightscan_predictions_total.labels(
                    model_type=model_type,
                    status='completed'
                )._value._value = count
            
            cursor.close()
            
        except Exception as e:
            print(f"Erreur collecte métriques business: {e}")
    
    def collect_metrics_loop(self):
        """Boucle collecte métriques (exécutée en arrière-plan)"""
        while True:
            try:
                self.collect_system_metrics()
                self.collect_redis_metrics()
                self.collect_database_metrics()
                self.collect_ml_metrics()
                self.collect_business_metrics()
                
                time.sleep(30)  # Collecte toutes les 30 secondes
                
            except Exception as e:
                print(f"Erreur boucle métriques: {e}")
                time.sleep(30)

# Instance globale
metrics_exporter = NightScanMetricsExporter()

def create_metrics_app():
    """Créer application Flask pour exposition métriques"""
    app = Flask(__name__)
    
    @app.route('/metrics')
    def metrics():
        """Endpoint métriques Prometheus"""
        return Response(
            generate_latest(),
            mimetype=CONTENT_TYPE_LATEST
        )
    
    @app.route('/health')
    def health():
        """Endpoint santé"""
        return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
    
    return app

# Fonctions utilitaires pour instrumenter code applicatif
def track_request(method, endpoint, status_code):
    """Tracker requête HTTP"""
    nightscan_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status=str(status_code)
    ).inc()

def track_request_duration(method, endpoint, duration):
    """Tracker durée requête"""
    nightscan_request_duration.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)

def track_upload(file_type, status):
    """Tracker upload fichier"""
    nightscan_uploads_total.labels(
        file_type=file_type,
        status=status
    ).inc()

def track_prediction(model_type, status):
    """Tracker prédiction ML"""
    nightscan_predictions_total.labels(
        model_type=model_type,
        status=status
    ).inc()

if __name__ == '__main__':
    # Démarrage standalone
    app = create_metrics_app()
    print("🚀 Démarrage exporter métriques NightScan")
    print("📊 Métriques disponibles sur http://localhost:8080/metrics")
    app.run(host='0.0.0.0', port=8080, debug=False)