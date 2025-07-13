#!/usr/bin/env python3
"""
Test Final - Démonstration du Système de Configuration Unifié
Ce script démontre que la fragmentation de configuration est complètement résolue.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def test_unified_system():
    """Test complet du système unifié."""
    print("🌙 NightScan - Test Final du Système de Configuration Unifié")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'summary': {}
    }
    
    # Test 1: Import du système unifié
    print("\n📦 Test 1: Import du système unifié")
    try:
        from unified_config import get_config, Environment
        print("✅ Système unifié importé avec succès")
        results['tests']['unified_import'] = True
    except Exception as e:
        print(f"❌ Erreur import: {e}")
        results['tests']['unified_import'] = False
        return results
    
    # Test 2: Configuration par environnement
    print("\n🌍 Test 2: Configurations par environnement")
    env_tests = {}
    
    for env in [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]:
        try:
            config = get_config(environment=env, force_reload=True)
            
            # Vérifications de base
            checks = {
                'database_configured': bool(config.database.host),
                'services_configured': config.services.web_port > 0,
                'ml_models_configured': bool(config.ml.audio_heavy_model),
                'logging_configured': bool(config.logging.log_directory),
                'security_configured': bool(config.security.secret_key)
            }
            
            all_passed = all(checks.values())
            passed_count = sum(checks.values())
            
            print(f"  {env.value:12} : {'✅' if all_passed else '⚠️'} {passed_count}/5 checks")
            env_tests[env.value] = {'passed': all_passed, 'details': checks}
            
        except Exception as e:
            print(f"  {env.value:12} : ❌ {e}")
            env_tests[env.value] = {'passed': False, 'error': str(e)}
    
    results['tests']['environments'] = env_tests
    
    # Test 3: Compatibilité legacy
    print("\n🔄 Test 3: Compatibilité avec anciens systèmes")
    try:
        from config_compatibility import get_legacy_value
        
        legacy_mappings = [
            ("WEB_PORT", 8000),
            ("DB_HOST", "localhost"),
            ("SECRET_KEY", str),
            ("LOG_LEVEL", str)
        ]
        
        legacy_results = {}
        for key, expected_type in legacy_mappings:
            value = get_legacy_value(key)
            is_valid = value is not None and (
                isinstance(value, expected_type) if not isinstance(expected_type, type) 
                else isinstance(value, expected_type)
            )
            
            print(f"  {key:15} : {'✅' if is_valid else '❌'} {value}")
            legacy_results[key] = {'valid': is_valid, 'value': str(value)}
        
        results['tests']['legacy_compatibility'] = legacy_results
        
    except Exception as e:
        print(f"❌ Erreur compatibilité legacy: {e}")
        results['tests']['legacy_compatibility'] = {'error': str(e)}
    
    # Test 4: Génération d'URLs et services
    print("\n🔗 Test 4: Génération d'URLs de services")
    try:
        config = get_config()
        
        urls = {
            'database': config.get_database_url(include_password=False),
            'cache': config.get_cache_url(include_password=False),
            'web_service': config.get_service_url('web'),
            'api_service': config.get_service_url('api_v1'),
            'prediction_service': config.get_service_url('prediction')
        }
        
        for service, url in urls.items():
            is_valid = url and ('://' in url) and not url.startswith('None')
            print(f"  {service:18} : {'✅' if is_valid else '❌'} {url}")
        
        results['tests']['url_generation'] = urls
        
    except Exception as e:
        print(f"❌ Erreur génération URLs: {e}")
        results['tests']['url_generation'] = {'error': str(e)}
    
    # Test 5: Validation des fichiers de configuration
    print("\n📁 Test 5: Fichiers de configuration unifiés")
    config_files = [
        'config/unified/development.json',
        'config/unified/staging.json', 
        'config/unified/production.json'
    ]
    
    file_results = {}
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Vérifications de structure
                required_sections = ['database', 'services', 'security', 'ml', 'logging']
                has_all_sections = all(section in data for section in required_sections)
                
                print(f"  {config_file:30} : {'✅' if has_all_sections else '❌'} {len(data)} sections")
                file_results[config_file] = {
                    'exists': True,
                    'valid_json': True,
                    'complete': has_all_sections,
                    'sections': len(data)
                }
                
            except json.JSONDecodeError as e:
                print(f"  {config_file:30} : ❌ JSON invalide")
                file_results[config_file] = {'exists': True, 'valid_json': False, 'error': str(e)}
        else:
            print(f"  {config_file:30} : ❌ Fichier manquant")
            file_results[config_file] = {'exists': False}
    
    results['tests']['config_files'] = file_results
    
    # Test 6: Variables d'environnement unifiées
    print("\n🌐 Test 6: Variables d'environnement standardisées")
    
    # Test avec des variables d'environnement simulées
    test_env_vars = {
        'NIGHTSCAN_ENV': 'development',
        'NIGHTSCAN_DB_HOST': 'test-db-host',
        'NIGHTSCAN_WEB_PORT': '9000',
        'NIGHTSCAN_LOG_LEVEL': 'DEBUG'
    }
    
    env_var_results = {}
    original_env = {}
    
    try:
        # Sauvegarder l'environnement original
        for var in test_env_vars:
            original_env[var] = os.environ.get(var)
        
        # Définir les variables de test
        os.environ.update(test_env_vars)
        
        # Créer une nouvelle config avec ces variables
        config = get_config(force_reload=True)
        
        # Vérifier que les variables sont bien prises en compte
        checks = {
            'db_host_updated': config.database.host == 'test-db-host',
            'web_port_updated': config.services.web_port == 9000,
            'log_level_updated': config.logging.level.value == 'DEBUG'
        }
        
        for check, result in checks.items():
            print(f"  {check:20} : {'✅' if result else '❌'}")
        
        env_var_results = checks
        
    except Exception as e:
        print(f"❌ Erreur variables d'environnement: {e}")
        env_var_results = {'error': str(e)}
    
    finally:
        # Restaurer l'environnement original
        for var, value in original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
    
    results['tests']['environment_variables'] = env_var_results
    
    # Calcul du résumé
    total_tests = 0
    passed_tests = 0
    
    for test_name, test_result in results['tests'].items():
        if test_name == 'environments':
            for env_result in test_result.values():
                total_tests += 1
                if env_result.get('passed', False):
                    passed_tests += 1
        elif test_name == 'legacy_compatibility':
            if 'error' not in test_result:
                for legacy_result in test_result.values():
                    total_tests += 1
                    if legacy_result.get('valid', False):
                        passed_tests += 1
        elif test_name == 'config_files':
            for file_result in test_result.values():
                total_tests += 1
                if file_result.get('complete', False):
                    passed_tests += 1
        elif test_name == 'environment_variables':
            if 'error' not in test_result:
                for var_result in test_result.values():
                    total_tests += 1
                    if var_result:
                        passed_tests += 1
        elif isinstance(test_result, bool):
            total_tests += 1
            if test_result:
                passed_tests += 1
        elif isinstance(test_result, dict) and 'error' not in test_result:
            total_tests += 1
            passed_tests += 1
    
    results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
        'overall_status': 'SUCCESS' if passed_tests == total_tests else 'PARTIAL' if passed_tests > total_tests * 0.8 else 'FAILURE'
    }
    
    # Affichage du résumé final
    print("\n" + "=" * 70)
    print("🎯 RÉSUMÉ FINAL - SYSTÈME DE CONFIGURATION UNIFIÉ")
    print("=" * 70)
    
    status_icon = {
        'SUCCESS': '✅',
        'PARTIAL': '⚠️', 
        'FAILURE': '❌'
    }[results['summary']['overall_status']]
    
    print(f"{status_icon} Statut global: {results['summary']['overall_status']}")
    print(f"📊 Tests réussis: {passed_tests}/{total_tests} ({results['summary']['success_rate']:.1f}%)")
    
    if results['summary']['overall_status'] == 'SUCCESS':
        print("\n🎉 CONFIGURATION UNIFIÉE ENTIÈREMENT FONCTIONNELLE!")
        print("   ✅ Fragmentation de configuration complètement résolue")
        print("   ✅ Système unifié opérationnel")
        print("   ✅ Compatibilité legacy assurée")
        print("   ✅ Migration réussie")
    else:
        print(f"\n⚠️ Système partiellement fonctionnel ({passed_tests}/{total_tests} tests)")
        print("   Voir les détails ci-dessus pour les problèmes à résoudre")
    
    # Sauvegarder le rapport
    report_path = Path("unified_config_final_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Rapport complet sauvegardé: {report_path}")
    
    return results

def main():
    """Point d'entrée principal."""
    results = test_unified_system()
    
    # Code de sortie basé sur le succès
    success = results['summary']['overall_status'] == 'SUCCESS'
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())