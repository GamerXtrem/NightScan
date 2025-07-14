#!/usr/bin/env python3
"""
Load Test Simple pour NightScan
Test de charge basique sans dépendances externes
"""

import urllib.request
import urllib.error
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SimpleTestResult:
    """Résultat test simple"""
    url: str
    status_code: int
    response_time: float
    success: bool
    error: str = None

class SimpleLoadTester:
    """Load tester simple avec urllib"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[SimpleTestResult] = []
        self.lock = threading.Lock()
    
    def make_request(self, endpoint: str, timeout: int = 10) -> SimpleTestResult:
        """Faire une requête HTTP simple"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                response.read()  # Lire contenu
                
                end_time = time.time()
                return SimpleTestResult(
                    url=url,
                    status_code=response.getcode(),
                    response_time=end_time - start_time,
                    success=200 <= response.getcode() < 400
                )
                
        except urllib.error.HTTPError as e:
            end_time = time.time()
            return SimpleTestResult(
                url=url,
                status_code=e.code,
                response_time=end_time - start_time,
                success=False,
                error=f"HTTP {e.code}"
            )
        except Exception as e:
            end_time = time.time()
            return SimpleTestResult(
                url=url,
                status_code=0,
                response_time=end_time - start_time,
                success=False,
                error=str(e)[:50]
            )
    
    def add_result(self, result: SimpleTestResult):
        """Ajouter résultat thread-safe"""
        with self.lock:
            self.results.append(result)
    
    def test_endpoints(self, endpoints: List[str], num_requests: int = 10):
        """Tester endpoints avec multiple requêtes"""
        print(f"🧪 Test {len(endpoints)} endpoints × {num_requests} requêtes")
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            
            for endpoint in endpoints:
                for i in range(num_requests):
                    future = executor.submit(self.make_request, endpoint)
                    futures.append(future)
            
            # Collecter résultats
            for future in as_completed(futures):
                try:
                    result = future.result()
                    self.add_result(result)
                except Exception as e:
                    print(f"Erreur future: {e}")
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        """Analyser résultats"""
        if not self.results:
            return {"error": "Aucun résultat"}
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        
        response_times = [r.response_time for r in self.results if r.success]
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
        else:
            avg_time = max_time = min_time = 0
        
        # Erreurs par type
        errors = {}
        for result in self.results:
            if not result.success and result.error:
                errors[result.error] = errors.get(result.error, 0) + 1
        
        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": (successful / total) * 100 if total > 0 else 0,
            "avg_response_time": avg_time,
            "min_response_time": min_time,
            "max_response_time": max_time,
            "errors": errors
        }

def run_basic_load_test():
    """Lancer test de charge basique"""
    print("🚀 LOAD TEST SIMPLE NIGHTSCAN")
    print("=" * 40)
    
    # Endpoints à tester
    endpoints = [
        "/",
        "/health", 
        "/api/v1/health",
        "/login",
        "/static/css/style.css"  # Test fichier statique
    ]
    
    tester = SimpleLoadTester()
    
    print("📊 Test disponibilité endpoints...")
    
    # Test préliminaire - 1 requête par endpoint
    for endpoint in endpoints:
        result = tester.make_request(endpoint, timeout=5)
        status = "✅" if result.success else "❌"
        print(f"  {status} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    # Reset pour test principal
    tester.results = []
    
    print("\n⚡ Load test principal...")
    start_time = time.time()
    
    # Test de charge - 50 requêtes par endpoint
    results = tester.test_endpoints(endpoints, num_requests=50)
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    print(f"\n📈 RÉSULTATS ({test_duration:.1f}s)")
    print("-" * 40)
    print(f"Total requêtes: {results['total_requests']}")
    print(f"Succès: {results['successful_requests']}")
    print(f"Échecs: {results['failed_requests']}")
    print(f"Taux succès: {results['success_rate']:.1f}%")
    print(f"Temps moyen: {results['avg_response_time']:.3f}s")
    print(f"Temps min: {results['min_response_time']:.3f}s")
    print(f"Temps max: {results['max_response_time']:.3f}s")
    print(f"Req/sec: {results['total_requests']/test_duration:.1f}")
    
    if results['errors']:
        print(f"\n❌ Erreurs:")
        for error, count in results['errors'].items():
            print(f"  {error}: {count}")
    
    # Évaluation
    print(f"\n🎯 ÉVALUATION")
    print("-" * 40)
    
    success_rate = results['success_rate']
    avg_time = results['avg_response_time']
    
    if success_rate >= 95 and avg_time < 1.0:
        evaluation = "🟢 EXCELLENT"
        recommendation = "Système prêt pour charge production"
    elif success_rate >= 90 and avg_time < 2.0:
        evaluation = "🟡 BON"
        recommendation = "Performance acceptable"
    elif success_rate >= 80:
        evaluation = "🟠 MOYEN"
        recommendation = "Optimisations recommandées"
    else:
        evaluation = "🔴 INSUFFISANT"
        recommendation = "Amélioration requise avant production"
    
    print(f"Statut: {evaluation}")
    print(f"Recommandation: {recommendation}")
    
    return success_rate >= 90

def test_specific_nightscan_features():
    """Tester fonctionnalités spécifiques NightScan"""
    print(f"\n🌙 TEST FONCTIONNALITÉS NIGHTSCAN")
    print("-" * 40)
    
    tester = SimpleLoadTester()
    
    # Test endpoints spécifiques
    nightscan_endpoints = [
        "/",                    # Page d'accueil
        "/api/v1/health",      # Health check API
        "/api/status",         # Status
        "/login",              # Login
        "/dashboard"           # Dashboard (peut être protégé)
    ]
    
    results = []
    for endpoint in nightscan_endpoints:
        result = tester.make_request(endpoint, timeout=5)
        results.append(result)
        
        status_icon = "✅" if result.success else "❌"
        status_detail = ""
        
        if result.success:
            if result.response_time < 0.5:
                time_icon = "⚡"
            elif result.response_time < 1.0:
                time_icon = "🟢"
            elif result.response_time < 2.0:
                time_icon = "🟡"
            else:
                time_icon = "🔴"
            
            status_detail = f"({result.response_time:.3f}s {time_icon})"
        else:
            status_detail = f"(Error: {result.error or 'Unknown'})"
        
        print(f"  {status_icon} {endpoint} {status_detail}")
    
    # Évaluation spécifique
    working_endpoints = sum(1 for r in results if r.success)
    total_endpoints = len(results)
    
    print(f"\n📊 Résumé: {working_endpoints}/{total_endpoints} endpoints fonctionnels")
    
    if working_endpoints >= total_endpoints * 0.8:
        print("🟢 NightScan répond correctement")
    else:
        print("🔴 Problèmes détectés avec NightScan")
    
    return working_endpoints >= total_endpoints * 0.8

def main():
    """Point d'entrée principal"""
    try:
        print("Starting NightScan load test...")
        
        # Test basique
        basic_success = run_basic_load_test()
        
        # Test fonctionnalités
        features_success = test_specific_nightscan_features()
        
        overall_success = basic_success and features_success
        
        print(f"\n🏁 RÉSULTAT FINAL")
        print("=" * 40)
        if overall_success:
            print("🟢 Load test RÉUSSI - Système stable")
            return 0
        else:
            print("🔴 Load test ÉCHOUÉ - Investigation requise")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)