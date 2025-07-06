#!/usr/bin/env python3
"""
Audit de sécurité focalisé PRODUCTION pour NightScan VPS Lite
Vérifie uniquement les fichiers critiques pour la production
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

class ProductionSecurityAuditor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
        # Fichiers critiques pour la production
        self.production_files = [
            "docker-compose.production.yml",
            "docker-compose.monitoring.yml", 
            "config.py",
            "web/app.py",
            "scripts/deploy-vps-lite.sh",
            "scripts/setup-secrets.sh",
            "monitoring/prometheus/prometheus.yml",
            "monitoring/loki/config.yml",
            "nginx.conf",
            ".env.production.example"
        ]
        
        # Patterns secrets critiques
        self.critical_secrets = [
            "nightscan_secret",
            "redis_secret", 
            "your-secret-key-here",
            "your-csrf-secret-key",
            "admin",
            "password123",
            "secret"
        ]
        
    def audit_production_files(self):
        """Auditer uniquement les fichiers de production"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "audit_scope": "production_only",
            "files_checked": [],
            "critical_issues": [],
            "recommendations": []
        }
        
        issues_count = 0
        
        for file_path in self.production_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                results["files_checked"].append(file_path)
                issues = self._check_file_security(full_path, file_path)
                results["critical_issues"].extend(issues)
                issues_count += len(issues)
        
        # Vérifications spéciales
        issues_count += self._check_secrets_directory(results)
        issues_count += self._check_docker_security(results)
        
        # Calculer score
        max_score = 10
        penalty = len(results["critical_issues"]) * 2
        score = max(0, max_score - penalty)
        
        results["summary"] = {
            "total_files_checked": len(results["files_checked"]),
            "critical_issues_count": len(results["critical_issues"]),
            "security_score": f"{score}/10",
            "production_ready": score >= 8 and len(results["critical_issues"]) == 0
        }
        
        # Générer recommandations
        if results["critical_issues"]:
            results["recommendations"] = [
                "🚨 Remplacer tous les secrets hardcodés par des variables d'environnement",
                "🔐 S'assurer que secrets/production/.env contient des valeurs sécurisées",
                "🐳 Vérifier la configuration Docker pour la production",
                "🔒 Utiliser des mots de passe forts générés aléatoirement"
            ]
        else:
            results["recommendations"] = [
                "✅ Configuration sécurisée validée pour la production",
                "🔄 Effectuer des audits réguliers",
                "📊 Monitorer les accès et les logs"
            ]
        
        return results
    
    def _check_file_security(self, file_path: Path, relative_path: str):
        """Vérifier la sécurité d'un fichier"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
            for i, line in enumerate(lines, 1):
                # Chercher les secrets critiques (ignorer références légitimes)
                line_lower = line.lower()
                if line.strip().startswith('#'):
                    continue
                    
                # Ignorer les références légitimes aux répertoires/variables
                if any(legitimate in line_lower for legitimate in [
                    "./secrets/", "env_file:", "${admin_email}", "${domain_name}",
                    "secrets/**", "filter=git-crypt"
                ]):
                    continue
                    
                for secret in self.critical_secrets:
                    if secret in line_lower:
                        issues.append({
                            "file": relative_path,
                            "line": i,
                            "type": "hardcoded_secret",
                            "secret_type": secret,
                            "severity": "critical",
                            "context": line.strip()
                        })
                
                # Vérifier patterns dangereux
                if "password=" in line.lower() and any(danger in line.lower() for danger in ["admin", "123", "password"]):
                    issues.append({
                        "file": relative_path,
                        "line": i,
                        "type": "weak_password",
                        "severity": "high",
                        "context": line.strip()
                    })
                    
        except Exception as e:
            print(f"Erreur lors de l'audit de {file_path}: {e}")
            
        return issues
    
    def _check_secrets_directory(self, results):
        """Vérifier le répertoire secrets"""
        issues_count = 0
        secrets_dir = self.project_root / "secrets" / "production"
        
        if not secrets_dir.exists():
            results["critical_issues"].append({
                "file": "secrets/production/",
                "type": "missing_secrets_dir",
                "severity": "critical",
                "context": "Répertoire secrets de production manquant"
            })
            issues_count += 1
        else:
            env_file = secrets_dir / ".env"
            if not env_file.exists():
                results["critical_issues"].append({
                    "file": "secrets/production/.env",
                    "type": "missing_env_file", 
                    "severity": "critical",
                    "context": "Fichier .env de production manquant"
                })
                issues_count += 1
            else:
                # Vérifier que les secrets ne sont pas par défaut
                try:
                    with open(env_file, 'r') as f:
                        env_content = f.read()
                        
                    for secret in self.critical_secrets:
                        if secret in env_content:
                            results["critical_issues"].append({
                                "file": "secrets/production/.env",
                                "type": "default_secret_in_env",
                                "severity": "critical", 
                                "context": f"Secret par défaut détecté: {secret}"
                            })
                            issues_count += 1
                            
                except Exception as e:
                    print(f"Erreur lecture .env: {e}")
        
        return issues_count
    
    def _check_docker_security(self, results):
        """Vérifier la sécurité Docker"""
        issues_count = 0
        compose_file = self.project_root / "docker-compose.production.yml"
        
        if compose_file.exists():
            try:
                with open(compose_file, 'r') as f:
                    content = f.read()
                
                # Vérifier env_file
                if "env_file:" not in content:
                    results["critical_issues"].append({
                        "file": "docker-compose.production.yml",
                        "type": "missing_env_file_config",
                        "severity": "high",
                        "context": "Configuration env_file manquante"
                    })
                    issues_count += 1
                
                # Vérifier privilèges
                if "privileged: true" in content:
                    results["critical_issues"].append({
                        "file": "docker-compose.production.yml", 
                        "type": "privileged_container",
                        "severity": "critical",
                        "context": "Container en mode privilégié détecté"
                    })
                    issues_count += 1
                    
            except Exception as e:
                print(f"Erreur audit Docker: {e}")
        
        return issues_count

def main():
    project_root = "."
    auditor = ProductionSecurityAuditor(project_root)
    results = auditor.audit_production_files()
    
    # Afficher résultats
    print("🔍 AUDIT DE SÉCURITÉ PRODUCTION")
    print("=" * 40)
    print(f"Fichiers vérifiés: {results['summary']['total_files_checked']}")
    print(f"Problèmes critiques: {results['summary']['critical_issues_count']}")
    print(f"Score sécurité: {results['summary']['security_score']}")
    print(f"Prêt production: {'✅ OUI' if results['summary']['production_ready'] else '❌ NON'}")
    
    if results["critical_issues"]:
        print("\n🚨 PROBLÈMES CRITIQUES:")
        for issue in results["critical_issues"][:5]:  # Limiter à 5
            print(f"  - {issue['file']}: {issue['type']} ({issue['severity']})")
    
    print("\n📋 RECOMMANDATIONS:")
    for rec in results["recommendations"]:
        print(f"  {rec}")
    
    # Sauvegarder rapport
    with open("security_audit_production.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Code de sortie
    if results["summary"]["production_ready"]:
        print("\n🎉 PRODUCTION READY!")
        exit(0)
    else:
        print("\n⚠️  CORRECTIONS REQUISES")
        exit(1)

if __name__ == "__main__":
    main()