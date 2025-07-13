#!/usr/bin/env python3
"""
Script d'audit de sécurité pour NightScan VPS Lite
Détecte les secrets hardcodés, vulnérabilités et problèmes de sécurité
"""

import os
import re
import json
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import argparse

class SecurityAuditor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "summary": {
                "total_issues": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "secrets": [],
            "vulnerabilities": [],
            "docker_security": [],
            "file_permissions": [],
            "recommendations": []
        }
        
        # Patterns pour détecter les secrets
        self.secret_patterns = {
            "password": [
                r'password\s*[=:]\s*["\']?([^"\'\s]{8,})["\']?',
                r'PASSWORD\s*[=:]\s*["\']?([^"\'\s]{8,})["\']?',
                r'pwd\s*[=:]\s*["\']?([^"\'\s]{8,})["\']?'
            ],
            "api_key": [
                r'api[_-]?key\s*[=:]\s*["\']?([a-zA-Z0-9]{20,})["\']?',
                r'API[_-]?KEY\s*[=:]\s*["\']?([a-zA-Z0-9]{20,})["\']?',
                r'token\s*[=:]\s*["\']?([a-zA-Z0-9]{20,})["\']?'
            ],
            "secret_key": [
                r'secret[_-]?key\s*[=:]\s*["\']?([a-zA-Z0-9+/]{20,})["\']?',
                r'SECRET[_-]?KEY\s*[=:]\s*["\']?([a-zA-Z0-9+/]{20,})["\']?'
            ],
            "database_url": [
                r'postgres://[^:\s]+:[^@\s]+@[^/\s]+/[^\s]+',
                r'mysql://[^:\s]+:[^@\s]+@[^/\s]+/[^\s]+',
                r'mongodb://[^:\s]+:[^@\s]+@[^/\s]+/[^\s]+'
            ],
            "jwt_secret": [
                r'jwt[_-]?secret\s*[=:]\s*["\']?([a-zA-Z0-9+/]{20,})["\']?',
                r'JWT[_-]?SECRET\s*[=:]\s*["\']?([a-zA-Z0-9+/]{20,})["\']?'
            ],
            "private_key": [
                r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
                r'private[_-]?key\s*[=:]\s*["\']?([a-zA-Z0-9+/]{40,})["\']?'
            ]
        }
        
        # Fichiers à ignorer (focus sur production seulement)
        self.ignore_patterns = [
            r'\.git/',
            r'node_modules/',
            r'__pycache__/',
            r'\.pyc$',
            r'\.log$',
            r'\.tmp$',
            r'/tests?/',
            r'test_.*\.py$',
            r'.*_test\.py$',
            r'security_audit.*\.json$',
            r'.*_baseline\.json$',
            r'secure_secrets\.py$',
            r'\.env\.secure$',
            r'fix_critical_conflicts\.py$',
            r'security_fixes\.py$',
            r'disaster_recovery\.py$',
            r'backup_system\.py$',
            r'analyze_conflicts\.py$',
            r'NightScanPi/',
            r'ios-app/',
            r'VPS_lite/',
            r'models/',
            r'audio_training/',
            r'picture_training/',
            r'PRODUCTION_.*\.md$',
            r'TODO_.*\.md$'
        ]
        
        # Secrets connus par défaut (vulnérabilités)
        self.default_secrets = [
            "nightscan_secret",
            "redis_secret", 
            "your-secret-key-here",
            "your-csrf-secret-key",
            "admin",
            "password",
            "123456",
            "secret"
        ]

    def should_ignore_file(self, file_path: str) -> bool:
        """Vérifie si un fichier doit être ignoré"""
        for pattern in self.ignore_patterns:
            if re.search(pattern, file_path):
                return True
        return False

    def scan_file_for_secrets(self, file_path: Path) -> List[Dict]:
        """Scanne un fichier pour des secrets hardcodés"""
        if self.should_ignore_file(str(file_path)):
            return []
            
        secrets_found = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for secret_type, patterns in self.secret_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        secret_value = match.group(1) if match.groups() else match.group(0)
                        
                        # Vérifier si c'est un secret par défaut (critique)
                        severity = "high"
                        if any(default in secret_value.lower() for default in self.default_secrets):
                            severity = "critical"
                        
                        line_num = content[:match.start()].count('\n') + 1
                        
                        secrets_found.append({
                            "type": secret_type,
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": line_num,
                            "value": secret_value[:20] + "..." if len(secret_value) > 20 else secret_value,
                            "severity": severity,
                            "context": self._get_line_context(content, line_num)
                        })
                        
        except Exception as e:
            print(f"Erreur lors du scan de {file_path}: {e}")
            
        return secrets_found

    def _get_line_context(self, content: str, line_num: int) -> str:
        """Récupère le contexte autour d'une ligne"""
        lines = content.split('\n')
        if line_num <= len(lines):
            return lines[line_num - 1].strip()
        return ""

    def scan_docker_security(self) -> List[Dict]:
        """Audit de sécurité des fichiers Docker"""
        issues = []
        
        # Vérifier Dockerfile
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            issues.extend(self._audit_dockerfile(dockerfile_path))
        
        # Vérifier docker-compose files
        for compose_file in self.project_root.glob("docker-compose*.yml"):
            issues.extend(self._audit_compose_file(compose_file))
            
        return issues

    def _audit_dockerfile(self, dockerfile_path: Path) -> List[Dict]:
        """Audit spécifique du Dockerfile"""
        issues = []
        
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Vérifier USER root
                if line.startswith('USER root'):
                    issues.append({
                        "type": "docker_security",
                        "file": str(dockerfile_path.relative_to(self.project_root)),
                        "line": i,
                        "issue": "Running as root user",
                        "severity": "high",
                        "recommendation": "Use non-root user for security"
                    })
                
                # Vérifier secrets dans ENV
                if line.startswith('ENV') and any(secret in line.lower() for secret in ['password', 'secret', 'key']):
                    issues.append({
                        "type": "docker_security", 
                        "file": str(dockerfile_path.relative_to(self.project_root)),
                        "line": i,
                        "issue": "Potential secret in ENV variable",
                        "severity": "high",
                        "recommendation": "Use Docker secrets or build args"
                    })
                    
                # Vérifier COPY avec permissions trop larges
                if line.startswith('COPY') and '--chmod' in line:
                    if '777' in line or '666' in line:
                        issues.append({
                            "type": "docker_security",
                            "file": str(dockerfile_path.relative_to(self.project_root)),
                            "line": i,
                            "issue": "Overly permissive file permissions",
                            "severity": "medium",
                            "recommendation": "Use restrictive permissions"
                        })
                        
        except Exception as e:
            print(f"Erreur audit Dockerfile: {e}")
            
        return issues

    def _audit_compose_file(self, compose_path: Path) -> List[Dict]:
        """Audit spécifique des fichiers docker-compose"""
        issues = []
        
        try:
            with open(compose_path, 'r') as f:
                content = f.read()
                
            # Vérifier les volumes avec permissions dangereuses
            if 'privileged: true' in content:
                issues.append({
                    "type": "docker_security",
                    "file": str(compose_path.relative_to(self.project_root)),
                    "line": 0,
                    "issue": "Container running in privileged mode",
                    "severity": "critical",
                    "recommendation": "Remove privileged mode unless absolutely necessary"
                })
            
            # Vérifier les ports exposés
            port_pattern = r'ports:\s*\n\s*-\s*"(\d+):\d+"'
            for match in re.finditer(port_pattern, content):
                port = match.group(1)
                if port in ['22', '3306', '5432', '6379', '27017']:
                    issues.append({
                        "type": "docker_security",
                        "file": str(compose_path.relative_to(self.project_root)),
                        "line": 0,
                        "issue": f"Database/SSH port {port} exposed externally",
                        "severity": "high",
                        "recommendation": f"Do not expose port {port} externally"
                    })
                    
        except Exception as e:
            print(f"Erreur audit docker-compose: {e}")
            
        return issues

    def check_file_permissions(self) -> List[Dict]:
        """Vérifier les permissions des fichiers sensibles"""
        issues = []
        
        sensitive_files = [
            ".env",
            ".env.production", 
            ".env.staging",
            "secrets/*",
            "ssl/*",
            "backup/*"
        ]
        
        for pattern in sensitive_files:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    
                    # Vérifier si le fichier est lisible par tous
                    if mode[2] in ['4', '5', '6', '7']:  # others can read
                        issues.append({
                            "type": "file_permissions",
                            "file": str(file_path.relative_to(self.project_root)),
                            "issue": f"File readable by others (mode: {mode})",
                            "severity": "high",
                            "recommendation": "chmod 600 for sensitive files"
                        })
                        
        return issues

    def run_dependency_scan(self) -> List[Dict]:
        """Scanner les vulnérabilités dans les dépendances"""
        vulnerabilities = []
        
        # Scan Python dependencies
        requirements_files = list(self.project_root.glob("requirements*.txt"))
        for req_file in requirements_files:
            try:
                result = subprocess.run([
                    'python', '-m', 'pip', 'install', 'safety'
                ], capture_output=True, text=True, timeout=30)
                
                result = subprocess.run([
                    'safety', 'check', '-r', str(req_file), '--json'
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and result.stdout:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        vulnerabilities.append({
                            "type": "dependency_vulnerability",
                            "package": vuln.get("package"),
                            "version": vuln.get("installed_version"),
                            "vulnerability": vuln.get("vulnerability"),
                            "severity": self._map_severity(vuln.get("severity")),
                            "file": str(req_file.relative_to(self.project_root))
                        })
                        
            except Exception as e:
                print(f"Erreur scan dépendances Python: {e}")
        
        # Scan Node.js dependencies si présent
        package_json = self.project_root / "package.json"
        if package_json.exists():
            try:
                result = subprocess.run([
                    'npm', 'audit', '--json'
                ], cwd=self.project_root, capture_output=True, text=True, timeout=60)
                
                if result.stdout:
                    audit_data = json.loads(result.stdout)
                    if 'vulnerabilities' in audit_data:
                        for pkg, vuln_data in audit_data['vulnerabilities'].items():
                            vulnerabilities.append({
                                "type": "dependency_vulnerability",
                                "package": pkg,
                                "severity": vuln_data.get('severity', 'unknown'),
                                "file": "package.json"
                            })
                            
            except Exception as e:
                print(f"Erreur scan dépendances Node.js: {e}")
                
        return vulnerabilities

    def _map_severity(self, severity: str) -> str:
        """Mapper les niveaux de sévérité"""
        mapping = {
            "critical": "critical",
            "high": "high", 
            "medium": "medium",
            "moderate": "medium",
            "low": "low",
            "info": "low"
        }
        return mapping.get(severity.lower() if severity else "", "medium")

    def generate_recommendations(self) -> List[str]:
        """Générer des recommandations de sécurité"""
        recommendations = []
        
        if self.results["summary"]["critical"] > 0:
            recommendations.append("🚨 CRITIQUE: Remplacer tous les secrets par défaut immédiatement")
            
        if any(s["type"] == "password" for s in self.results["secrets"]):
            recommendations.append("🔐 Implémenter git-crypt pour chiffrer les secrets")
            recommendations.append("🔑 Utiliser des variables d'environnement pour les secrets")
            
        if any(d["issue"].startswith("Container running") for d in self.results["docker_security"]):
            recommendations.append("🐳 Revoir la configuration Docker pour réduire les privilèges")
            
        recommendations.extend([
            "🛡️ Activer fail2ban pour protéger SSH",
            "🔥 Configurer UFW firewall avec règles restrictives", 
            "📱 Implémenter monitoring des intrusions",
            "🔄 Mettre en place rotation automatique des secrets",
            "📊 Activer audit logging pour les accès privilégiés"
        ])
        
        return recommendations

    def run_full_audit(self) -> Dict[str, Any]:
        """Exécuter l'audit complet"""
        print("🔍 Début de l'audit de sécurité NightScan...")
        
        # 1. Scanner les secrets
        print("   📝 Scan des secrets hardcodés...")
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self.should_ignore_file(str(file_path)):
                secrets = self.scan_file_for_secrets(file_path)
                self.results["secrets"].extend(secrets)
        
        # 2. Audit Docker
        print("   🐳 Audit sécurité Docker...")
        self.results["docker_security"] = self.scan_docker_security()
        
        # 3. Vérifier permissions
        print("   📁 Vérification permissions fichiers...")
        self.results["file_permissions"] = self.check_file_permissions()
        
        # 4. Scanner dépendances
        print("   📦 Scan vulnérabilités dépendances...")
        self.results["vulnerabilities"] = self.run_dependency_scan()
        
        # 5. Calculer statistiques
        all_issues = (self.results["secrets"] + 
                     self.results["docker_security"] + 
                     self.results["file_permissions"] + 
                     self.results["vulnerabilities"])
        
        for issue in all_issues:
            severity = issue.get("severity", "medium")
            self.results["summary"]["total_issues"] += 1
            self.results["summary"][severity] += 1
        
        # 6. Générer recommandations
        self.results["recommendations"] = self.generate_recommendations()
        
        print(f"✅ Audit terminé: {self.results['summary']['total_issues']} problèmes détectés")
        return self.results

    def save_report(self, output_file: str = "security_audit_report.json"):
        """Sauvegarder le rapport d'audit"""
        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"📄 Rapport sauvegardé: {output_path}")

    def print_summary(self):
        """Afficher un résumé de l'audit"""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE L'AUDIT DE SÉCURITÉ")
        print("="*60)
        
        summary = self.results["summary"]
        print(f"Total des problèmes: {summary['total_issues']}")
        print(f"🔴 Critiques:        {summary['critical']}")
        print(f"🟠 Élevés:          {summary['high']}")
        print(f"🟡 Moyens:          {summary['medium']}")
        print(f"🟢 Faibles:         {summary['low']}")
        
        # Score de sécurité
        max_score = 10
        penalty = (summary['critical'] * 3 + summary['high'] * 2 + 
                  summary['medium'] * 1 + summary['low'] * 0.5)
        score = max(0, max_score - penalty)
        
        print(f"\n🎯 Score de sécurité: {score:.1f}/10")
        
        if summary['critical'] > 0:
            print("🚨 ACTION IMMÉDIATE REQUISE - Vulnérabilités critiques détectées!")
        elif summary['high'] > 0:
            print("⚠️  Corrections urgentes recommandées")
        elif summary['medium'] > 0:
            print("✅ Sécurité acceptable, améliorations recommandées")
        else:
            print("🛡️ Excellente posture de sécurité!")
            
        print("\n📋 RECOMMANDATIONS PRIORITAIRES:")
        for i, rec in enumerate(self.results["recommendations"][:5], 1):
            print(f"{i}. {rec}")


def main():
    parser = argparse.ArgumentParser(description="Audit de sécurité NightScan")
    parser.add_argument("--project-root", default=".", help="Racine du projet")
    parser.add_argument("--output", default="security_audit_report.json", help="Fichier de sortie")
    parser.add_argument("--full", action="store_true", help="Audit complet")
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor(args.project_root)
    results = auditor.run_full_audit()
    
    auditor.print_summary()
    auditor.save_report(args.output)
    
    # Exit code basé sur la sévérité
    if results["summary"]["critical"] > 0:
        exit(3)  # Critique
    elif results["summary"]["high"] > 0:
        exit(2)  # Élevé
    elif results["summary"]["medium"] > 0:
        exit(1)  # Moyen
    else:
        exit(0)  # OK


if __name__ == "__main__":
    main()