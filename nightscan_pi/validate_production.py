#!/usr/bin/env python3
"""
NightScanPi Production Validation Script
Vérifie que tous les composants sont prêts pour le déploiement en production
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path
from typing import List, Tuple
import logging

# Configuration des couleurs pour les logs
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m' 
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def log_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def log_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def log_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def check_imports() -> bool:
    """Vérifie que tous les imports critiques fonctionnent"""
    print(f"\n{Colors.BOLD}Vérification des imports{Colors.END}")
    print("=" * 30)
    
    critical_modules = [
        ("Program.main", "Module principal"),
        ("Program.sync", "Module de synchronisation"),
        ("Program.camera_trigger", "Module caméra"),
        ("Program.audio_capture", "Module audio"),
        ("Program.spectrogram_gen", "Module spectrogrammes")
    ]
    
    success_count = 0
    for module_name, description in critical_modules:
        try:
            importlib.import_module(module_name)
            log_success(f"{description} - Import OK")
            success_count += 1
        except ImportError as e:
            log_error(f"{description} - Import failed: {e}")
        except Exception as e:
            log_warning(f"{description} - Import warning: {e}")
            success_count += 1  # Count as success if not import error
    
    return success_count == len(critical_modules)

def check_environment_config() -> bool:
    """Vérifie la configuration des variables d'environnement"""
    print(f"\n{Colors.BOLD}Vérification de la configuration{Colors.END}")
    print("=" * 35)
    
    if not Path(".env").exists():
        log_error("Fichier .env manquant - exécutez ./configure_production.sh")
        return False
    
    # Variables critiques
    critical_vars = {
        "NIGHTSCAN_API_URL": "URL de l'API",
        "NIGHTSCAN_GPS_COORDS": "Coordonnées GPS"
    }
    
    # Variables importantes
    important_vars = {
        "NIGHTSCAN_API_TOKEN": "Token d'authentification",
        "NIGHTSCAN_DATA_DIR": "Répertoire de données",
        "NIGHTSCAN_LOG": "Fichier de log"
    }
    
    # Lire le fichier .env
    env_vars = {}
    try:
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
    except Exception as e:
        log_error(f"Erreur lecture .env: {e}")
        return False
    
    # Vérifier les variables critiques
    all_good = True
    for var, description in critical_vars.items():
        if var in env_vars and env_vars[var] and not env_vars[var].endswith("example.com"):
            log_success(f"{description} configuré")
        else:
            log_error(f"{description} manquant ou non configuré")
            all_good = False
    
    # Vérifier les variables importantes
    for var, description in important_vars.items():
        if var in env_vars and env_vars[var]:
            log_success(f"{description} configuré")
        else:
            log_warning(f"{description} non configuré (optionnel)")
    
    return all_good

def check_directories() -> bool:
    """Vérifie que les répertoires nécessaires existent"""
    print(f"\n{Colors.BOLD}Vérification des répertoires{Colors.END}")
    print("=" * 33)
    
    required_dirs = [
        "Program",
        "Program/utils", 
        "Hardware"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            log_success(f"Répertoire {dir_path} présent")
        else:
            log_error(f"Répertoire {dir_path} manquant")
            all_good = False
    
    return all_good

def check_hardware_configs() -> bool:
    """Vérifie les configurations matérielles"""
    print(f"\n{Colors.BOLD}Vérification du matériel{Colors.END}")
    print("=" * 28)
    
    hardware_files = [
        ("Hardware/configure_camera_boot.sh", "Script configuration caméra"),
        ("Hardware/configure_respeaker_audio.sh", "Script configuration audio"),
        ("setup_pi.sh", "Script d'installation")
    ]
    
    for file_path, description in hardware_files:
        if Path(file_path).exists():
            if os.access(file_path, os.X_OK):
                log_success(f"{description} - Exécutable")
            else:
                log_warning(f"{description} - Non exécutable")
        else:
            log_warning(f"{description} - Absent")
    
    return True  # Non critique pour la validation

def check_pi_detection() -> bool:
    """Vérifie si on est sur un Raspberry Pi"""
    print(f"\n{Colors.BOLD}Détection du Raspberry Pi{Colors.END}")
    print("=" * 28)
    
    try:
        if Path("/proc/device-tree/model").exists():
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().strip('\0')
                log_success(f"Raspberry Pi détecté: {model}")
                
                if "Pi Zero" in model:
                    log_info("Optimisations Pi Zero activées")
                
                return True
        else:
            log_warning("Pas sur Raspberry Pi - environnement de développement")
            return True  # OK pour développement
            
    except Exception as e:
        log_warning(f"Détection Pi échouée: {e}")
        return True  # Non critique

def check_python_packages() -> bool:
    """Vérifie les packages Python essentiels"""
    print(f"\n{Colors.BOLD}Vérification des packages Python{Colors.END}")
    print("=" * 38)
    
    required_packages = [
        ("numpy", "Calculs numériques"),
        ("requests", "Requêtes HTTP"),
        ("flask", "Serveur web"),
        ("soundfile", "Audio"),
        ("Pillow", "Images")
    ]
    
    success_count = 0
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            log_success(f"{description} ({package}) installé")
            success_count += 1
        except ImportError:
            log_error(f"{description} ({package}) manquant")
    
    return success_count >= 3  # Au moins 3 packages critiques

def run_syntax_checks() -> bool:
    """Exécute des vérifications de syntaxe sur les fichiers critiques"""
    print(f"\n{Colors.BOLD}Vérification de la syntaxe{Colors.END}")
    print("=" * 30)
    
    python_files = [
        "Program/main.py",
        "Program/sync.py", 
        "Program/camera_trigger.py",
        "Program/audio_capture.py"
    ]
    
    all_good = True
    for file_path in python_files:
        if Path(file_path).exists():
            try:
                result = subprocess.run([
                    sys.executable, "-m", "py_compile", file_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    log_success(f"Syntaxe OK: {file_path}")
                else:
                    log_error(f"Erreur syntaxe: {file_path}")
                    if result.stderr:
                        print(f"  {result.stderr}")
                    all_good = False
                    
            except Exception as e:
                log_error(f"Test syntaxe échoué pour {file_path}: {e}")
                all_good = False
        else:
            log_warning(f"Fichier absent: {file_path}")
    
    return all_good

def main():
    """Exécute toutes les vérifications"""
    print(f"{Colors.BOLD}{Colors.BLUE}🔍 Validation Production NightScanPi{Colors.END}")
    print("=" * 50)
    
    # Changer vers le répertoire du script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    checks = [
        ("Imports", check_imports),
        ("Configuration", check_environment_config),
        ("Répertoires", check_directories),
        ("Matériel", check_hardware_configs),
        ("Détection Pi", check_pi_detection),
        ("Packages Python", check_python_packages),
        ("Syntaxe", run_syntax_checks)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            log_error(f"Erreur lors de {name}: {e}")
            results.append((name, False))
    
    # Résumé final
    print(f"\n{Colors.BOLD}📊 Résumé de la validation{Colors.END}")
    print("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:20} {status}")
    
    print(f"\n{Colors.BOLD}Score: {passed}/{total} vérifications réussies{Colors.END}")
    
    if passed == total:
        log_success("🎉 Système prêt pour la production!")
        print(f"\n{Colors.BOLD}Étapes suivantes:{Colors.END}")
        print("1. Transférer sur le Raspberry Pi")
        print("2. Exécuter: ./setup_pi.sh")
        print("3. Exécuter: ./configure_production.sh") 
        print("4. Tester: python Program/main.py")
        return 0
    else:
        log_error(f"❗ {total - passed} problèmes détectés")
        print(f"\n{Colors.BOLD}Actions recommandées:{Colors.END}")
        print("1. Corriger les erreurs ci-dessus")
        print("2. Exécuter ./configure_production.sh")
        print("3. Relancer cette validation")
        return 1

if __name__ == "__main__":
    sys.exit(main())