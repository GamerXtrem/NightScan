#!/usr/bin/env python3
"""
Script de Load Testing Production pour NightScan
Test 1000+ utilisateurs simultanés pour validation production
"""

import asyncio
import aiohttp
import time
import json
import random
import tempfile
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class LoadTestConfig:
    """Configuration du load test"""
    base_url: str = "http://localhost:8000"
    api_url: str = "http://localhost:8001"
    prediction_url: str = "http://localhost:8002"
    
    # Paramètres de charge
    total_users: int = 1000
    concurrent_users: int = 100
    test_duration: int = 300  # 5 minutes
    ramp_up_time: int = 60    # 1 minute
    
    # Endpoints à tester
    endpoints_web: List[str] = None
    endpoints_api: List[str] = None
    
    def __post_init__(self):
        if self.endpoints_web is None:
            self.endpoints_web = [
                "/",
                "/login", 
                "/register",
                "/dashboard",
                "/upload",
                "/health"
            ]
        
        if self.endpoints_api is None:
            self.endpoints_api = [
                "/api/v1/health",
                "/api/v1/status",
                "/api/v1/predictions",
                "/api/predict"
            ]

@dataclass
class TestResult:
    """Résultat d'un test individuel"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error: str = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class LoadTestMetrics:
    """Collecteur de métriques load test"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        self.lock = threading.Lock()
    
    def add_result(self, result: TestResult):
        """Ajouter résultat thread-safe"""
        with self.lock:
            self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Générer résumé des métriques"""
        if not self.results:
            return {"error": "Aucun résultat"}
        
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        
        response_times = [r.response_time for r in self.results if r.success]
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Percentiles
            sorted_times = sorted(response_times)
            p50 = sorted_times[int(len(sorted_times) * 0.5)]
            p90 = sorted_times[int(len(sorted_times) * 0.9)]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50 = p90 = p95 = p99 = 0
        
        # Taux de succès par endpoint
        endpoint_stats = {}
        for result in self.results:
            if result.endpoint not in endpoint_stats:
                endpoint_stats[result.endpoint] = {'total': 0, 'success': 0, 'avg_time': 0}
            endpoint_stats[result.endpoint]['total'] += 1
            if result.success:
                endpoint_stats[result.endpoint]['success'] += 1
        
        # Calculer moyennes par endpoint
        for endpoint, stats in endpoint_stats.items():
            endpoint_times = [r.response_time for r in self.results 
                            if r.endpoint == endpoint and r.success]
            if endpoint_times:
                stats['avg_time'] = sum(endpoint_times) / len(endpoint_times)
            stats['success_rate'] = (stats['success'] / stats['total']) * 100
        
        duration = (self.end_time - self.start_time) if (self.end_time and self.start_time) else 0
        requests_per_second = total_requests / duration if duration > 0 else 0
        
        return {
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": (successful_requests / total_requests) * 100,
                "requests_per_second": requests_per_second,
                "test_duration": duration
            },
            "response_times": {
                "average": avg_response_time,
                "minimum": min_response_time,
                "maximum": max_response_time,
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "p99": p99
            },
            "endpoint_stats": endpoint_stats,
            "errors": self._get_error_summary()
        }
    
    def _get_error_summary(self) -> Dict[str, int]:
        """Résumé des erreurs"""
        error_counts = {}
        for result in self.results:
            if not result.success and result.error:
                error_counts[result.error] = error_counts.get(result.error, 0) + 1
        return error_counts

class NightScanLoadTester:
    """Load tester principal pour NightScan"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics = LoadTestMetrics()
        self.session = None
        
    async def create_session(self):
        """Créer session HTTP async"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(
            limit=200,  # Pool de connexions
            limit_per_host=50,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
    
    async def close_session(self):
        """Fermer session HTTP"""
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, url: str, **kwargs) -> TestResult:
        """Faire une requête HTTP et mesurer performance"""
        start_time = time.time()
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                # Lire contenu pour simulation réaliste
                await response.read()
                
                end_time = time.time()
                response_time = end_time - start_time
                
                return TestResult(
                    endpoint=url,
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    success=200 <= response.status < 400,
                    timestamp=start_time
                )
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            return TestResult(
                endpoint=url,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e)[:100],  # Limiter taille erreur
                timestamp=start_time
            )
    
    async def test_web_endpoints(self, user_id: int):
        """Tester endpoints web pour un utilisateur"""
        for endpoint in self.config.endpoints_web:
            url = f"{self.config.base_url}{endpoint}"
            
            # Ajouter délai aléatoire pour simulation réaliste
            await asyncio.sleep(random.uniform(0.1, 2.0))
            
            result = await self.make_request("GET", url)
            self.metrics.add_result(result)
    
    async def test_api_endpoints(self, user_id: int):
        """Tester endpoints API pour un utilisateur"""
        for endpoint in self.config.endpoints_api:
            url = f"{self.config.api_url}{endpoint}"
            
            await asyncio.sleep(random.uniform(0.1, 1.0))
            
            result = await self.make_request("GET", url)
            self.metrics.add_result(result)
    
    async def test_file_upload(self, user_id: int):
        """Tester upload de fichier (simulation)"""
        upload_url = f"{self.config.base_url}/upload"
        
        # Créer fichier temporaire simulé
        fake_audio_data = b"fake audio data " * 1000  # ~16KB
        
        data = aiohttp.FormData()
        data.add_field('file', fake_audio_data, filename=f'test_audio_{user_id}.wav')
        
        result = await self.make_request("POST", upload_url, data=data)
        self.metrics.add_result(result)
    
    async def test_prediction_api(self, user_id: int):
        """Tester API de prédiction"""
        predict_url = f"{self.config.prediction_url}/api/predict"
        
        # Données de test pour prédiction
        test_data = {
            "audio_data": "base64_encoded_audio_data_simulation",
            "user_id": user_id
        }
        
        result = await self.make_request("POST", predict_url, json=test_data)
        self.metrics.add_result(result)
    
    async def simulate_user_session(self, user_id: int):
        """Simuler session utilisateur complète"""
        try:
            # Séquence réaliste d'actions utilisateur
            
            # 1. Page d'accueil
            await self.test_web_endpoints(user_id)
            
            # 2. API calls
            await self.test_api_endpoints(user_id)
            
            # 3. Upload fichier (probabilité 30%)
            if random.random() < 0.3:
                await self.test_file_upload(user_id)
            
            # 4. Prédiction (probabilité 50%)
            if random.random() < 0.5:
                await self.test_prediction_api(user_id)
                
        except Exception as e:
            # Log erreur utilisateur mais continue
            error_result = TestResult(
                endpoint="user_session",
                method="SIMULATION",
                status_code=0,
                response_time=0,
                success=False,
                error=f"User session error: {str(e)[:100]}"
            )
            self.metrics.add_result(error_result)
    
    async def run_load_test(self):
        """Exécuter load test principal"""
        print("🚀 DÉMARRAGE LOAD TEST NIGHTSCAN")
        print("=" * 50)
        print(f"Configuration:")
        print(f"  Utilisateurs total: {self.config.total_users}")
        print(f"  Utilisateurs simultanés: {self.config.concurrent_users}")
        print(f"  Durée test: {self.config.test_duration}s")
        print(f"  Ramp-up: {self.config.ramp_up_time}s")
        print()
        
        await self.create_session()
        
        self.metrics.start_time = time.time()
        
        try:
            # Phase de montée en charge
            print("📈 Phase ramp-up...")
            tasks = []
            
            users_per_batch = self.config.concurrent_users
            total_batches = (self.config.total_users + users_per_batch - 1) // users_per_batch
            
            for batch in range(total_batches):
                batch_start = batch * users_per_batch
                batch_end = min((batch + 1) * users_per_batch, self.config.total_users)
                
                print(f"  Lancement batch {batch + 1}/{total_batches} (utilisateurs {batch_start}-{batch_end})")
                
                # Créer tâches pour ce batch
                batch_tasks = []
                for user_id in range(batch_start, batch_end):
                    task = asyncio.create_task(self.simulate_user_session(user_id))
                    batch_tasks.append(task)
                
                tasks.extend(batch_tasks)
                
                # Délai entre batches pour ramp-up
                if batch < total_batches - 1:
                    delay = self.config.ramp_up_time / total_batches
                    await asyncio.sleep(delay)
            
            print(f"⏳ Attente completion {len(tasks)} tâches utilisateur...")
            
            # Attendre completion avec timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.test_duration
                )
            except asyncio.TimeoutError:
                print("⚠️ Timeout atteint, arrêt des tâches en cours...")
                
                # Annuler tâches restantes
                for task in tasks:
                    if not task.done():
                        task.cancel()
        
        finally:
            self.metrics.end_time = time.time()
            await self.close_session()
        
        print("✅ Load test terminé!")
        return self.metrics.get_summary()
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Générer rapport détaillé"""
        report = []
        report.append("🧪 RAPPORT LOAD TEST NIGHTSCAN")
        report.append("=" * 60)
        report.append("")
        
        # Résumé
        summary = results["summary"]
        report.append("📊 RÉSUMÉ GÉNÉRAL")
        report.append("-" * 30)
        report.append(f"Total requêtes: {summary['total_requests']:,}")
        report.append(f"Requêtes réussies: {summary['successful_requests']:,}")
        report.append(f"Requêtes échouées: {summary['failed_requests']:,}")
        report.append(f"Taux de succès: {summary['success_rate']:.2f}%")
        report.append(f"Req/seconde: {summary['requests_per_second']:.2f}")
        report.append(f"Durée test: {summary['test_duration']:.1f}s")
        report.append("")
        
        # Temps de réponse
        times = results["response_times"]
        report.append("⏱️ TEMPS DE RÉPONSE")
        report.append("-" * 30)
        report.append(f"Moyenne: {times['average']:.3f}s")
        report.append(f"Minimum: {times['minimum']:.3f}s")
        report.append(f"Maximum: {times['maximum']:.3f}s")
        report.append(f"P50 (médiane): {times['p50']:.3f}s")
        report.append(f"P90: {times['p90']:.3f}s")
        report.append(f"P95: {times['p95']:.3f}s")
        report.append(f"P99: {times['p99']:.3f}s")
        report.append("")
        
        # Stats par endpoint
        endpoint_stats = results["endpoint_stats"]
        report.append("🎯 PERFORMANCE PAR ENDPOINT")
        report.append("-" * 30)
        for endpoint, stats in endpoint_stats.items():
            report.append(f"{endpoint}:")
            report.append(f"  Requêtes: {stats['total']:,}")
            report.append(f"  Succès: {stats['success_rate']:.1f}%")
            report.append(f"  Temps moyen: {stats['avg_time']:.3f}s")
        report.append("")
        
        # Erreurs
        if results["errors"]:
            report.append("❌ ERREURS DÉTECTÉES")
            report.append("-" * 30)
            for error, count in results["errors"].items():
                report.append(f"  {error}: {count:,} occurrences")
            report.append("")
        
        # Évaluation
        report.append("🎯 ÉVALUATION PRODUCTION")
        report.append("-" * 30)
        
        success_rate = summary['success_rate']
        avg_response = times['average']
        p95_response = times['p95']
        
        if success_rate >= 99.5 and avg_response < 0.5 and p95_response < 2.0:
            status = "🟢 EXCELLENT - Prêt production"
        elif success_rate >= 99.0 and avg_response < 1.0 and p95_response < 3.0:
            status = "🟡 BON - Acceptable production"
        elif success_rate >= 95.0 and avg_response < 2.0:
            status = "🟠 MOYEN - Optimisations recommandées"
        else:
            status = "🔴 INSUFFISANT - Améliorations requises"
        
        report.append(f"Statut: {status}")
        report.append("")
        
        report.append("📋 RECOMMANDATIONS")
        report.append("-" * 30)
        if success_rate < 99.0:
            report.append("• Améliorer stabilité (taux succès < 99%)")
        if avg_response > 1.0:
            report.append("• Optimiser temps réponse moyen")
        if p95_response > 3.0:
            report.append("• Réduire latence P95")
        if summary['requests_per_second'] < 100:
            report.append("• Améliorer throughput")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Sauvegarder résultats"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"load_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Résultats sauvegardés: {filename}")

async def main():
    """Point d'entrée principal"""
    # Configuration par défaut
    config = LoadTestConfig(
        total_users=1000,
        concurrent_users=100,
        test_duration=300,  # 5 minutes
        ramp_up_time=60     # 1 minute
    )
    
    # Créer et lancer load tester
    tester = NightScanLoadTester(config)
    
    try:
        results = await tester.run_load_test()
        
        # Générer et afficher rapport
        report = tester.generate_report(results)
        print(report)
        
        # Sauvegarder résultats
        tester.save_results(results)
        
        return 0 if results["summary"]["success_rate"] >= 95.0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu par utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur load test: {e}")
        return 1

if __name__ == "__main__":
    import sys
    
    # Vérifier si aiohttp est disponible
    try:
        import aiohttp
    except ImportError:
        print("❌ Module aiohttp requis pour load testing")
        print("Installation: pip install aiohttp")
        sys.exit(1)
    
    # Lancer test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)