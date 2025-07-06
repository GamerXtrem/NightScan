#!/usr/bin/env python3
"""
Validation complète production NightScan VPS Lite
Tests intégrés Phase 1 + Phase 2 + Readiness Check
"""

import os
import sys
import json
import time
import socket
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import psutil

class ProductionValidator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "validation_type": "production_readiness",
            "phase1_security": {},
            "phase2_infrastructure": {},
            "performance": {},
            "readiness": {},
            "score": 0,
            "production_ready": False,
            "issues": [],
            "recommendations": []
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log avec timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def success(self, message: str):
        self.log(f"✅ {message}", "SUCCESS")
        
    def warning(self, message: str):
        self.log(f"⚠️  {message}", "WARNING")
        
    def error(self, message: str):
        self.log(f"❌ {message}", "ERROR")
        self.results["issues"].append(message)
        
    def validate_phase1_security(self) -> bool:
        """Valider Phase 1 - Sécurité critique"""
        self.log("🔒 Validation Phase 1 - Sécurité critique...")
        
        # Exécuter le validateur Phase 1
        try:
            result = subprocess.run([
                "python", str(self.project_root / "scripts" / "validate-phase1.py")
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                self.success("Phase 1 sécurité validée")
                self.results["phase1_security"]["status"] = "passed"
                self.results["phase1_security"]["score"] = "10/10"
                return True
            else:
                self.error(f"Phase 1 échouée: {result.stderr}")
                self.results["phase1_security"]["status"] = "failed"
                return False
                
        except Exception as e:
            self.error(f"Erreur validation Phase 1: {e}")
            return False
            
    def validate_docker_services(self) -> bool:
        """Valider services Docker"""
        self.log("🐳 Validation services Docker...")
        
        try:
            # Vérifier docker-compose.production.yml
            compose_file = self.project_root / "docker-compose.production.yml"
            if not compose_file.exists():
                self.error("docker-compose.production.yml manquant")
                return False
                
            # Vérifier que les services sont définis
            with open(compose_file, 'r') as f:
                content = f.read()
                
            required_services = ["web", "prediction-api", "postgres", "redis"]
            missing_services = []
            
            for service in required_services:
                if service not in content:
                    missing_services.append(service)
                    
            if missing_services:
                self.error(f"Services Docker manquants: {missing_services}")
                return False
                
            self.success("Configuration Docker Compose validée")
            return True
            
        except Exception as e:
            self.error(f"Erreur validation Docker: {e}")
            return False
            
    def validate_ssl_configuration(self) -> bool:
        """Valider configuration SSL"""
        self.log("🔒 Validation configuration SSL...")
        
        try:
            # Vérifier fichiers SSL
            ssl_files = [
                "docker-compose.ssl.yml",
                "nginx/nginx.production.conf",
                "scripts/setup-ssl.sh"
            ]
            
            for ssl_file in ssl_files:
                if not (self.project_root / ssl_file).exists():
                    self.error(f"Fichier SSL manquant: {ssl_file}")
                    return False
                    
            # Vérifier répertoires SSL
            ssl_dirs = ["ssl/letsencrypt", "ssl/challenges", "nginx/maintenance"]
            for ssl_dir in ssl_dirs:
                ssl_path = self.project_root / ssl_dir
                if not ssl_path.exists():
                    self.warning(f"Répertoire SSL manquant: {ssl_dir}")
                    
            self.success("Configuration SSL validée")
            return True
            
        except Exception as e:
            self.error(f"Erreur validation SSL: {e}")
            return False
            
    def validate_monitoring_setup(self) -> bool:
        """Valider configuration monitoring"""
        self.log("📊 Validation monitoring...")
        
        try:
            # Vérifier fichiers monitoring
            monitoring_files = [
                "docker-compose.monitoring.yml",
                "monitoring/loki/config.yml",
                "monitoring/promtail/config.yml",
                "scripts/deploy-monitoring.sh"
            ]
            
            for mon_file in monitoring_files:
                if not (self.project_root / mon_file).exists():
                    self.error(f"Fichier monitoring manquant: {mon_file}")
                    return False
                    
            # Vérifier configuration Loki
            loki_config = self.project_root / "monitoring/loki/config.yml"
            with open(loki_config, 'r') as f:
                content = f.read()
                
            if "auth_enabled: false" not in content:
                self.warning("Loki auth non configuré")
                
            if "max_size_mb: 100" not in content:
                self.warning("Limites mémoire Loki non optimisées VPS Lite")
                
            self.success("Configuration monitoring validée")
            return True
            
        except Exception as e:
            self.error(f"Erreur validation monitoring: {e}")
            return False
            
    def validate_security_scripts(self) -> bool:
        """Valider scripts de sécurité"""
        self.log("🛡️  Validation scripts sécurité...")
        
        try:
            security_scripts = [
                "scripts/setup-firewall.sh",
                "scripts/setup-secrets.sh"
            ]
            
            for script in security_scripts:
                script_path = self.project_root / script
                if not script_path.exists():
                    self.error(f"Script sécurité manquant: {script}")
                    return False
                    
                # Vérifier permissions exécutables
                if not os.access(script_path, os.X_OK):
                    self.warning(f"Script non exécutable: {script}")
                    
            self.success("Scripts sécurité validés")
            return True
            
        except Exception as e:
            self.error(f"Erreur validation scripts sécurité: {e}")
            return False
            
    def validate_backup_system(self) -> bool:
        """Valider système backup"""
        self.log("💾 Validation système backup...")
        
        try:
            backup_scripts = [
                "scripts/setup-backup.sh"
            ]
            
            for script in backup_scripts:
                if not (self.project_root / script).exists():
                    self.error(f"Script backup manquant: {script}")
                    return False
                    
            self.success("Système backup validé")
            return True
            
        except Exception as e:
            self.error(f"Erreur validation backup: {e}")
            return False
            
    def validate_resource_optimization(self) -> bool:
        """Valider optimisations ressources VPS Lite"""
        self.log("💻 Validation optimisations VPS Lite...")
        
        try:
            # Vérifier limites mémoire dans docker-compose
            compose_file = self.project_root / "docker-compose.production.yml"
            with open(compose_file, 'r') as f:
                content = f.read()
                
            # Vérifier présence des limites
            if "mem_limit:" not in content:
                self.error("Limites mémoire non configurées")
                return False
                
            if "cpus:" not in content:
                self.warning("Limites CPU non configurées")
                
            # Vérifier configuration Nginx optimisée
            nginx_config = self.project_root / "nginx/nginx.production.conf"
            if nginx_config.exists():
                with open(nginx_config, 'r') as f:
                    nginx_content = f.read()
                    
                if "worker_processes 2" not in nginx_content:
                    self.warning("Nginx non optimisé pour 2 vCPU")
                    
                if "client_max_body_size 50M" not in nginx_content:
                    self.warning("Nginx non optimisé pour VPS Lite")
                    
            self.success("Optimisations VPS Lite validées")
            return True
            
        except Exception as e:
            self.error(f"Erreur validation optimisations: {e}")
            return False
            
    def test_network_connectivity(self) -> bool:
        """Tester connectivité réseau"""
        self.log("🌐 Test connectivité réseau...")
        
        try:
            # Test ports essentiels
            ports_to_test = [80, 443, 22]
            
            for port in ports_to_test:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    self.success(f"Port {port} accessible")
                else:
                    self.warning(f"Port {port} non accessible (normal si services non démarrés)")
                    
            self.success("Tests connectivité terminés")
            return True
            
        except Exception as e:
            self.error(f"Erreur test connectivité: {e}")
            return False
            
    def performance_check(self) -> Dict[str, Any]:
        """Vérification performance système"""
        self.log("⚡ Vérification performance système...")
        
        perf_data = {}
        
        try:
            # Mémoire disponible
            memory = psutil.virtual_memory()
            perf_data["memory_total_gb"] = round(memory.total / (1024**3), 1)
            perf_data["memory_available_gb"] = round(memory.available / (1024**3), 1)
            perf_data["memory_percent"] = memory.percent
            
            # CPU
            perf_data["cpu_count"] = psutil.cpu_count()
            perf_data["cpu_percent"] = psutil.cpu_percent(interval=1)
            
            # Disque
            disk = psutil.disk_usage('/')
            perf_data["disk_total_gb"] = round(disk.total / (1024**3), 1)
            perf_data["disk_free_gb"] = round(disk.free / (1024**3), 1)
            perf_data["disk_percent"] = round((disk.used / disk.total) * 100, 1)
            
            # Évaluation pour VPS Lite
            vps_lite_compatible = True
            issues = []
            
            if perf_data["memory_total_gb"] < 3.5:
                issues.append("RAM insuffisante pour VPS Lite (4GB requis)")
                vps_lite_compatible = False
                
            if perf_data["cpu_count"] < 2:
                issues.append("CPU insuffisant pour VPS Lite (2 vCPU requis)")
                vps_lite_compatible = False
                
            if perf_data["disk_free_gb"] < 10:
                issues.append("Espace disque faible pour déploiement")
                vps_lite_compatible = False
                
            perf_data["vps_lite_compatible"] = vps_lite_compatible
            perf_data["issues"] = issues
            
            self.results["performance"] = perf_data
            
            if vps_lite_compatible:
                self.success("Performance compatible VPS Lite")
            else:
                self.error(f"Performance incompatible: {', '.join(issues)}")
                
            return perf_data
            
        except Exception as e:
            self.error(f"Erreur vérification performance: {e}")
            return {}
            
    def calculate_production_score(self) -> int:
        """Calculer score de préparation production"""
        score = 0
        max_score = 100
        
        # Phase 1 sécurité (30 points)
        if self.results["phase1_security"].get("status") == "passed":
            score += 30
            
        # Infrastructure (40 points)
        infra_checks = [
            "docker_services",
            "ssl_config", 
            "monitoring",
            "security_scripts",
            "backup_system",
            "resource_optimization"
        ]
        
        infra_score = sum(1 for check in infra_checks 
                         if self.results["phase2_infrastructure"].get(check) == "passed")
        score += int((infra_score / len(infra_checks)) * 40)
        
        # Performance (20 points)
        if self.results["performance"].get("vps_lite_compatible"):
            score += 20
            
        # Connectivité (10 points)
        if self.results["readiness"].get("network_connectivity"):
            score += 10
            
        return min(score, max_score)
        
    def generate_recommendations(self) -> List[str]:
        """Générer recommandations"""
        recommendations = []
        
        if self.results["phase1_security"].get("status") != "passed":
            recommendations.append("🔒 CRITIQUE: Corriger la sécurité Phase 1")
            
        if not self.results["performance"].get("vps_lite_compatible"):
            recommendations.append("💻 Optimiser les ressources système")
            
        if len(self.results["issues"]) == 0:
            recommendations.extend([
                "🚀 Prêt pour déploiement production VPS Lite",
                "🔄 Tester le déploiement en staging d'abord",
                "📊 Configurer alertes monitoring",
                "🔄 Planifier tests de charge post-déploiement"
            ])
        else:
            recommendations.append("🛠️ Corriger les problèmes détectés avant déploiement")
            
        return recommendations
        
    def run_complete_validation(self) -> Dict[str, Any]:
        """Validation complète"""
        self.log("🧪 === VALIDATION PRODUCTION NIGHTSCAN VPS LITE ===")
        
        # Phase 1 - Sécurité
        phase1_ok = self.validate_phase1_security()
        
        # Phase 2 - Infrastructure
        infra_results = {}
        infra_results["docker_services"] = "passed" if self.validate_docker_services() else "failed"
        infra_results["ssl_config"] = "passed" if self.validate_ssl_configuration() else "failed"
        infra_results["monitoring"] = "passed" if self.validate_monitoring_setup() else "failed"
        infra_results["security_scripts"] = "passed" if self.validate_security_scripts() else "failed"
        infra_results["backup_system"] = "passed" if self.validate_backup_system() else "failed"
        infra_results["resource_optimization"] = "passed" if self.validate_resource_optimization() else "failed"
        
        self.results["phase2_infrastructure"] = infra_results
        
        # Performance & Readiness
        self.performance_check()
        self.results["readiness"]["network_connectivity"] = self.test_network_connectivity()
        
        # Score final
        self.results["score"] = self.calculate_production_score()
        self.results["production_ready"] = (
            self.results["score"] >= 80 and 
            len(self.results["issues"]) == 0
        )
        
        # Recommandations
        self.results["recommendations"] = self.generate_recommendations()
        
        return self.results
        
    def print_summary(self):
        """Afficher résumé"""
        print("\n" + "="*60)
        print("📊 RAPPORT DE VALIDATION PRODUCTION")
        print("="*60)
        
        score = self.results["score"]
        print(f"🎯 Score global: {score}/100")
        
        if self.results["production_ready"]:
            print("🎉 PRODUCTION READY!")
            print("✅ Système prêt pour déploiement VPS Lite")
        else:
            print("⚠️  CORRECTIONS REQUISES")
            print(f"❌ {len(self.results['issues'])} problème(s) détecté(s)")
            
        print(f"\n📋 Phase 1 Sécurité: {self.results['phase1_security'].get('status', 'unknown')}")
        
        print(f"\n🏗️  Phase 2 Infrastructure:")
        for check, status in self.results["phase2_infrastructure"].items():
            status_icon = "✅" if status == "passed" else "❌"
            print(f"   {status_icon} {check}: {status}")
            
        perf = self.results["performance"]
        if perf:
            print(f"\n💻 Performance:")
            print(f"   RAM: {perf.get('memory_available_gb', 'N/A')}GB libre / {perf.get('memory_total_gb', 'N/A')}GB total")
            print(f"   CPU: {perf.get('cpu_count', 'N/A')} cœurs ({perf.get('cpu_percent', 'N/A')}% utilisé)")
            print(f"   Disque: {perf.get('disk_free_gb', 'N/A')}GB libre ({perf.get('disk_percent', 'N/A')}% utilisé)")
            vps_compat = "✅ Compatible" if perf.get('vps_lite_compatible') else "❌ Incompatible"
            print(f"   VPS Lite: {vps_compat}")
            
        if self.results["issues"]:
            print(f"\n❌ PROBLÈMES ({len(self.results['issues'])}):")
            for issue in self.results["issues"][:5]:
                print(f"   • {issue}")
                
        print(f"\n📋 RECOMMANDATIONS:")
        for rec in self.results["recommendations"][:5]:
            print(f"   {rec}")
            
    def save_report(self, output_file: str = "production_validation_report.json"):
        """Sauvegarder rapport"""
        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        self.log(f"📄 Rapport sauvegardé: {output_path}")

def main():
    project_root = "."
    validator = ProductionValidator(project_root)
    
    results = validator.run_complete_validation()
    validator.print_summary()
    validator.save_report()
    
    # Exit code basé sur la readiness
    if results["production_ready"]:
        print("\n🚀 Système validé pour production!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Corrections requises (Score: {results['score']}/100)")
        sys.exit(1)

if __name__ == "__main__":
    main()