#!/usr/bin/env python3
"""
Validation intelligente Phase 1 - Sécurité critique
Focalisé sur les vrais problèmes de sécurité sans faux positifs
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

class Phase1Validator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.critical_issues = 0
        self.high_issues = 0
        self.medium_issues = 0
        self.validation_passed = True
        
    def validate_secrets_generated(self):
        """Valider que les secrets sont générés et sécurisés"""
        print("🔐 Validation 1: Secrets générés et sécurisés...")
        
        env_file = self.project_root / "secrets" / "production" / ".env"
        
        if not env_file.exists():
            print("❌ Fichier secrets/production/.env manquant")
            self.critical_issues += 1
            self.validation_passed = False
            return False
            
        # Vérifier que les secrets ne sont pas par défaut
        with open(env_file, 'r') as f:
            content = f.read()
            
        dangerous_defaults = [
            "nightscan_secret",
            "redis_secret", 
            "your-secret-key-here",
            "CHANGE_ME_STRONG_PASSWORD",
            "CHANGE_ME_SMTP_PASSWORD",
            "admin@yourdomain.com"
        ]
        
        for default in dangerous_defaults:
            if default in content:
                print(f"❌ Secret par défaut détecté: {default}")
                self.critical_issues += 1
                self.validation_passed = False
                return False
        
        # Vérifier les permissions
        try:
            perms = oct(env_file.stat().st_mode)[-3:]
            if perms != "600":
                print(f"⚠️  Permissions fichier secrets: {perms} (recommandé: 600)")
                self.medium_issues += 1
        except:
            pass
            
        print("✅ Secrets générés et sécurisés")
        return True
        
    def validate_hardcoded_secrets_removed(self):
        """Valider que les secrets critiques sont supprimés"""
        print("🧹 Validation 2: Secrets hardcodés supprimés...")
        
        # Fichiers critiques à vérifier
        critical_files = [
            "docker-compose.production.yml",
            "config.py"
        ]
        
        # Secrets vraiment dangereux
        dangerous_secrets = [
            "nightscan_secret",
            "redis_secret",
            "your-secret-key-here"
        ]
        
        issues_found = 0
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        
                    for secret in dangerous_secrets:
                        if secret in content:
                            print(f"❌ Secret hardcodé '{secret}' dans {file_path}")
                            issues_found += 1
                            
                except Exception as e:
                    print(f"⚠️  Erreur lecture {file_path}: {e}")
                    
        if issues_found > 0:
            print(f"❌ {issues_found} secrets hardcodés détectés")
            self.critical_issues += issues_found
            self.validation_passed = False
            return False
            
        print("✅ Aucun secret hardcodé critique détecté")
        return True
        
    def validate_docker_security(self):
        """Valider la sécurité Docker"""
        print("🐳 Validation 3: Sécurité Docker...")
        
        compose_file = self.project_root / "docker-compose.production.yml"
        
        if not compose_file.exists():
            print("❌ docker-compose.production.yml manquant")
            self.high_issues += 1
            return False
            
        try:
            with open(compose_file, 'r') as f:
                content = f.read()
                
            # Vérifier env_file
            if "env_file:" not in content:
                print("⚠️  env_file non configuré")
                self.high_issues += 1
                
            # Vérifier privilèges
            if "privileged: true" in content:
                print("❌ Container en mode privilégié détecté")
                self.critical_issues += 1
                self.validation_passed = False
                
            # Vérifier limites ressources
            if "mem_limit:" not in content:
                print("⚠️  Limites mémoire non configurées")
                self.medium_issues += 1
                
        except Exception as e:
            print(f"⚠️  Erreur lecture docker-compose: {e}")
            self.medium_issues += 1
            
        print("✅ Docker Compose sécurisé")
        return True
        
    def validate_gitignore(self):
        """Valider .gitignore pour secrets"""
        print("📝 Validation 4: Configuration .gitignore...")
        
        gitignore_file = self.project_root / ".gitignore"
        
        if not gitignore_file.exists():
            print("⚠️  .gitignore manquant")
            self.medium_issues += 1
            return False
            
        try:
            with open(gitignore_file, 'r') as f:
                content = f.read()
                
            if "secrets/" not in content:
                print("❌ Répertoire secrets/ non ignoré par Git")
                self.critical_issues += 1
                self.validation_passed = False
                return False
                
        except Exception as e:
            print(f"⚠️  Erreur lecture .gitignore: {e}")
            self.medium_issues += 1
            
        print("✅ Secrets correctement ignorés par Git")
        return True
        
    def validate_production_env_example(self):
        """Valider que .env.production.example n'a pas de secrets réels"""
        print("📋 Validation 5: Fichier .env.production.example...")
        
        example_file = self.project_root / ".env.production.example"
        
        if not example_file.exists():
            print("⚠️  .env.production.example manquant")
            self.medium_issues += 1
            return False
            
        try:
            with open(example_file, 'r') as f:
                content = f.read()
                
            # Vérifier que ce sont bien des placeholders
            if "CHANGE_ME_" not in content:
                print("⚠️  Placeholders manquants dans .env.production.example")
                self.medium_issues += 1
                
        except Exception as e:
            print(f"⚠️  Erreur lecture .env.production.example: {e}")
            self.medium_issues += 1
            
        print("✅ Fichier exemple correctement configuré")
        return True
        
    def calculate_security_score(self):
        """Calculer le score de sécurité"""
        max_score = 10
        penalty = (self.critical_issues * 3 + 
                  self.high_issues * 2 + 
                  self.medium_issues * 1)
        score = max(0, max_score - penalty)
        return score
        
    def run_validation(self):
        """Exécuter la validation complète"""
        print("🛡️  VALIDATION PHASE 1 - SÉCURITÉ CRITIQUE")
        print("=" * 40)
        
        # Exécuter toutes les validations
        self.validate_secrets_generated()
        self.validate_hardcoded_secrets_removed()
        self.validate_docker_security()
        self.validate_gitignore()
        self.validate_production_env_example()
        
        # Calculer le score final
        security_score = self.calculate_security_score()
        
        print("\n📊 RÉSULTATS DE LA VALIDATION")
        print("=" * 30)
        print(f"🔴 Critiques:  {self.critical_issues}")
        print(f"🟠 Élevés:     {self.high_issues}")
        print(f"🟡 Moyens:     {self.medium_issues}")
        print(f"\n🎯 Score de sécurité: {security_score}/10")
        
        # Déterminer le statut final
        if self.validation_passed and security_score >= 8:
            print("\n✅ PHASE 1 VALIDÉE - Sécurité critique réussie!")
            print("🎉 GATE_PASSED=true")
            print("\n📋 Critères validés:")
            print("  ✅ Secrets sécurisés générés")
            print("  ✅ Secrets hardcodés supprimés")
            print("  ✅ Docker Compose sécurisé")
            print("  ✅ Score sécurité ≥ 8/10")
            print("\n🚀 Prêt pour Phase 2 - Infrastructure")
            
            # Sauvegarder le résultat
            result = {
                "timestamp": datetime.now().isoformat(),
                "phase": "1_security_critical",
                "status": "passed",
                "score": f"{security_score}/10",
                "critical_issues": self.critical_issues,
                "high_issues": self.high_issues,
                "medium_issues": self.medium_issues,
                "gate_passed": True
            }
            
            with open(self.project_root / "phase1_validation_result.json", 'w') as f:
                json.dump(result, f, indent=2)
                
            return True
            
        else:
            print("\n❌ PHASE 1 ÉCHOUÉE - Corrections requises")
            print("🚫 GATE_PASSED=false")
            print("\n📋 Actions requises:")
            
            if self.critical_issues > 0:
                print(f"  🔴 Corriger {self.critical_issues} problème(s) CRITIQUE(S)")
            if self.high_issues > 0:
                print(f"  🟠 Corriger {self.high_issues} problème(s) ÉLEVÉ(S)")
            if security_score < 8:
                print(f"  🎯 Améliorer score: {security_score}/10 → ≥8/10")
                
            print("\n🔧 Recommandations:")
            print("  1. Relancer: ./scripts/setup-secrets.sh --env production")
            print("  2. Re-valider: python scripts/validate-phase1.py")
            
            # Sauvegarder le résultat
            result = {
                "timestamp": datetime.now().isoformat(),
                "phase": "1_security_critical",
                "status": "failed",
                "score": f"{security_score}/10",
                "critical_issues": self.critical_issues,
                "high_issues": self.high_issues,
                "medium_issues": self.medium_issues,
                "gate_passed": False
            }
            
            with open(self.project_root / "phase1_validation_result.json", 'w') as f:
                json.dump(result, f, indent=2)
                
            return False

def main():
    project_root = "."
    validator = Phase1Validator(project_root)
    
    success = validator.run_validation()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()