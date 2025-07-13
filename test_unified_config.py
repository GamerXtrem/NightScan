#!/usr/bin/env python3
"""
Tests et Validation du Système de Configuration Unifié
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_config import UnifiedConfig, Environment, get_config
from config_compatibility import get_legacy_config, get_legacy_value, FlaskConfigAdapter


class TestUnifiedConfig(unittest.TestCase):
    """Tests pour le système de configuration unifié."""
    
    def setUp(self):
        """Configuration des tests."""
        self.test_config = UnifiedConfig(Environment.TESTING)
    
    def test_default_configuration(self):
        """Test de la configuration par défaut."""
        config = UnifiedConfig(Environment.DEVELOPMENT)
        
        # Vérifier les valeurs par défaut
        self.assertEqual(config.database.host, "localhost")
        self.assertEqual(config.database.port, 5432)
        self.assertEqual(config.services.web_port, 8000)
        self.assertEqual(config.cache.port, 6379)
        self.assertEqual(config.ml.audio_model_classes, 6)
        self.assertEqual(config.ml.photo_model_classes, 8)
    
    def test_environment_specific_defaults(self):
        """Test des configurations spécifiques par environnement."""
        # Development
        dev_config = UnifiedConfig(Environment.DEVELOPMENT)
        self.assertEqual(dev_config.logging.level.value, "DEBUG")
        self.assertFalse(dev_config.security.session_cookie_secure)
        
        # Production
        prod_config = UnifiedConfig(Environment.PRODUCTION)
        self.assertEqual(prod_config.logging.level.value, "WARNING")
        self.assertTrue(prod_config.security.session_cookie_secure)
        
        # Testing
        test_config = UnifiedConfig(Environment.TESTING)
        self.assertEqual(test_config.database.database, "nightscan_test")
        self.assertEqual(test_config.cache.database, 1)
    
    def test_database_url_generation(self):
        """Test de génération des URLs de base de données."""
        config = self.test_config
        config.database.username = "testuser"
        config.database.password = "testpass"
        config.database.host = "testhost"
        config.database.port = 5433
        config.database.database = "testdb"
        
        # URL complète
        expected_url = "postgresql://testuser:testpass@testhost:5433/testdb"
        self.assertEqual(config.get_database_url(), expected_url)
        
        # URL sans mot de passe
        expected_safe_url = "postgresql://testuser@testhost:5433/testdb"
        self.assertEqual(config.get_database_url(include_password=False), expected_safe_url)
    
    def test_cache_url_generation(self):
        """Test de génération des URLs de cache."""
        config = self.test_config
        config.cache.host = "redis-host"
        config.cache.port = 6380
        config.cache.database = 2
        config.cache.password = "redispass"
        
        expected_url = "redis://:redispass@redis-host:6380/2"
        self.assertEqual(config.get_cache_url(), expected_url)
    
    def test_service_url_generation(self):
        """Test de génération des URLs de service."""
        config = self.test_config
        
        # Service web
        web_url = config.get_service_url("web", "example.com")
        self.assertEqual(web_url, "http://example.com:8000")
        
        # Service API
        api_url = config.get_service_url("api_v1", "api.example.com", ssl=True)
        self.assertEqual(api_url, "https://api.example.com:8001")
    
    def test_environment_variable_loading(self):
        """Test du chargement des variables d'environnement."""
        with patch.dict(os.environ, {
            'NIGHTSCAN_DB_HOST': 'env-db-host',
            'NIGHTSCAN_DB_PORT': '5434',
            'NIGHTSCAN_WEB_PORT': '8080',
            'SECRET_KEY': 'env-secret-key'
        }):
            config = UnifiedConfig(Environment.DEVELOPMENT)
            
            self.assertEqual(config.database.host, 'env-db-host')
            self.assertEqual(config.database.port, 5434)
            self.assertEqual(config.services.web_port, 8080)
            self.assertEqual(config.security.secret_key, 'env-secret-key')
    
    def test_database_url_parsing(self):
        """Test du parsing des URLs de base de données."""
        with patch.dict(os.environ, {
            'DATABASE_URL': 'postgresql://dbuser:dbpass@db.example.com:5435/proddb'
        }):
            config = UnifiedConfig(Environment.PRODUCTION)
            
            self.assertEqual(config.database.host, 'db.example.com')
            self.assertEqual(config.database.port, 5435)
            self.assertEqual(config.database.username, 'dbuser')
            self.assertEqual(config.database.password, 'dbpass')
            self.assertEqual(config.database.database, 'proddb')
    
    def test_config_file_loading(self):
        """Test du chargement depuis un fichier de configuration."""
        # Créer un fichier de config temporaire
        config_data = {
            "database": {
                "host": "file-db-host",
                "port": 5436
            },
            "services": {
                "web_port": 8090
            },
            "security": {
                "session_timeout": 7200
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_config_file = f.name
        
        try:
            config = UnifiedConfig(Environment.DEVELOPMENT, temp_config_file)
            
            self.assertEqual(config.database.host, 'file-db-host')
            self.assertEqual(config.database.port, 5436)
            self.assertEqual(config.services.web_port, 8090)
            self.assertEqual(config.security.session_timeout, 7200)
        finally:
            os.unlink(temp_config_file)
    
    def test_config_export(self):
        """Test de l'export de configuration."""
        config = self.test_config
        
        # Export sans secrets
        exported = config.to_dict(include_secrets=False)
        self.assertIn('database', exported)
        self.assertIn('services', exported)
        self.assertEqual(exported['security']['secret_key'], '***masked***')
        
        # Export avec secrets
        exported_with_secrets = config.to_dict(include_secrets=True)
        self.assertNotEqual(exported_with_secrets['security']['secret_key'], '***masked***')
    
    def test_secret_generation(self):
        """Test de la génération automatique de secrets."""
        config = UnifiedConfig(Environment.DEVELOPMENT)
        
        # Les secrets doivent être générés automatiquement
        self.assertNotEqual(config.security.secret_key, "")
        self.assertNotEqual(config.security.jwt_secret, "")
        self.assertNotEqual(config.security.encryption_key, "")
        
        # Les secrets doivent avoir une longueur minimale
        self.assertGreaterEqual(len(config.security.secret_key), 32)


class TestLegacyCompatibility(unittest.TestCase):
    """Tests pour la compatibilité avec les anciens systèmes."""
    
    def test_legacy_value_mapping(self):
        """Test du mapping des valeurs legacy."""
        wrapper = get_legacy_config()
        
        # Test de mapping des ports
        web_port = wrapper.get_legacy_value("WEB_PORT")
        self.assertIsInstance(web_port, int)
        self.assertEqual(web_port, 8000)  # Valeur par défaut
        
        # Test de mapping de la base de données
        db_host = wrapper.get_legacy_value("DB_HOST")
        self.assertEqual(db_host, "localhost")
    
    def test_legacy_environment_variables(self):
        """Test des variables d'environnement legacy."""
        with patch.dict(os.environ, {'WEB_PORT': '9000'}):
            wrapper = get_legacy_config()
            port = wrapper.get_legacy_value("WEB_PORT")
            self.assertEqual(port, '9000')  # Valeur d'environnement
    
    def test_legacy_utility_function(self):
        """Test de la fonction utilitaire legacy."""
        port = get_legacy_value("WEB_PORT", 8080)
        self.assertIsInstance(port, int)
        
        # Test avec valeur par défaut
        unknown_value = get_legacy_value("UNKNOWN_KEY", "default")
        self.assertEqual(unknown_value, "default")


class TestFlaskIntegration(unittest.TestCase):
    """Tests pour l'intégration Flask."""
    
    def test_flask_config_adapter(self):
        """Test de l'adaptateur Flask."""
        # Mock Flask app
        mock_app = MagicMock()
        mock_app.config = {}
        
        adapter = FlaskConfigAdapter(mock_app)
        
        # Vérifier que la configuration Flask a été mise à jour
        self.assertIn('SECRET_KEY', mock_app.config)
        self.assertIn('SQLALCHEMY_DATABASE_URI', mock_app.config)
        self.assertIn('NIGHTSCAN_CONFIG', mock_app.config)


class TestConfigValidation:
    """Tests de validation de la configuration."""
    
    def test_production_validation(self):
        """Test de validation pour la production."""
        # Configuration production sans secrets
        with patch.dict(os.environ, {'SECRET_KEY': ''}):
            config = UnifiedConfig(Environment.PRODUCTION)
            
            # Les secrets doivent être générés même si vides
            self.assertNotEqual(config.security.secret_key, "")
    
    def test_directory_creation(self):
        """Test de création des dossiers nécessaires."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config = UnifiedConfig(Environment.DEVELOPMENT)
            config.logging.log_directory = str(temp_path / "logs")
            config.ml.models_directory = str(temp_path / "models")
            
            # Force la création des dossiers
            config._ensure_directories()
            
            self.assertTrue((temp_path / "logs").exists())
            self.assertTrue((temp_path / "models").exists())


def run_integration_tests():
    """Tests d'intégration avec les vrais fichiers de configuration."""
    print("🔄 Tests d'intégration avec les configurations unifiées")
    print("=" * 60)
    
    environments = [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]
    
    for env in environments:
        print(f"\\n📋 Test environnement: {env.value}")
        
        try:
            # Test avec fichier de configuration unifié
            config_file = f"config/unified/{env.value}.json"
            if Path(config_file).exists():
                config = UnifiedConfig(env, config_file)
                print(f"  ✅ Configuration chargée depuis {config_file}")
                
                # Test des URLs générées
                db_url = config.get_database_url(include_password=False)
                cache_url = config.get_cache_url(include_password=False)
                web_url = config.get_service_url("web")
                
                print(f"  📊 DB: {db_url}")
                print(f"  📊 Cache: {cache_url}")
                print(f"  📊 Web: {web_url}")
                
                # Test de validation
                stats = {
                    'database_configured': bool(config.database.host),
                    'cache_configured': bool(config.cache.host),
                    'secrets_present': bool(config.security.secret_key),
                    'ml_models_configured': bool(config.ml.audio_heavy_model),
                }
                
                all_good = all(stats.values())
                status = "✅" if all_good else "⚠️"
                print(f"  {status} Validation: {sum(stats.values())}/4 OK")
                
            else:
                print(f"  ❌ Fichier de configuration manquant: {config_file}")
                
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    print("\\n🎯 Tests d'intégration terminés")


def generate_documentation():
    """Génère la documentation de configuration."""
    print("📚 Génération de la documentation de configuration")
    print("=" * 60)
    
    # Créer une instance avec tous les defaults
    config = UnifiedConfig(Environment.DEVELOPMENT)
    
    # Exporter la configuration complète
    full_config = config.to_dict(include_secrets=True)
    
    # Générer documentation markdown
    doc_lines = [
        "# NightScan - Configuration Unifiée",
        "",
        "## Vue d'ensemble",
        "",
        "Ce document décrit la configuration unifiée de NightScan qui remplace",
        "les multiples approches de configuration précédentes.",
        "",
        "## Variables d'environnement",
        "",
        "### Principales variables NIGHTSCAN_*",
        "",
        "| Variable | Description | Défaut | Exemple |",
        "|----------|-------------|--------|---------|",
    ]
    
    env_vars = [
        ("NIGHTSCAN_ENV", "Environnement de déploiement", "development", "production"),
        ("NIGHTSCAN_DB_HOST", "Hôte de la base de données", "localhost", "db.example.com"),
        ("NIGHTSCAN_DB_PORT", "Port de la base de données", "5432", "5432"),
        ("NIGHTSCAN_WEB_PORT", "Port du service web", "8000", "8080"),
        ("NIGHTSCAN_LOG_LEVEL", "Niveau de logging", "INFO", "DEBUG"),
        ("NIGHTSCAN_USE_GPU", "Utilisation du GPU", "true", "false"),
    ]
    
    for var, desc, default, example in env_vars:
        doc_lines.append(f"| `{var}` | {desc} | `{default}` | `{example}` |")
    
    doc_lines.extend([
        "",
        "## Configuration par environnement",
        "",
        "### Development",
        "```json",
        json.dumps(full_config, indent=2)[:500] + "...",
        "```",
        "",
        "## Migration depuis l'ancien système",
        "",
        "1. Utiliser `python unified_config.py migrate` pour migrer automatiquement",
        "2. Adapter le code avec `from config_compatibility import get_legacy_value`",
        "3. Mettre à jour les variables d'environnement avec le préfixe `NIGHTSCAN_`",
        "",
        "## Exemples d'utilisation",
        "",
        "```python",
        "from unified_config import get_config",
        "",
        "config = get_config()",
        "db_url = config.get_database_url()",
        "web_port = config.services.web_port",
        "```"
    ])
    
    # Sauvegarder la documentation
    doc_path = Path("docs/UNIFIED_CONFIGURATION.md")
    doc_path.parent.mkdir(exist_ok=True)
    
    with open(doc_path, 'w') as f:
        f.write('\\n'.join(doc_lines))
    
    print(f"✅ Documentation générée: {doc_path}")


def main():
    """Fonction principale de test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tests du système de configuration unifié")
    parser.add_argument("--unit", action="store_true", help="Exécuter les tests unitaires")
    parser.add_argument("--integration", action="store_true", help="Exécuter les tests d'intégration")
    parser.add_argument("--docs", action="store_true", help="Générer la documentation")
    parser.add_argument("--all", action="store_true", help="Exécuter tous les tests")
    
    args = parser.parse_args()
    
    if args.all or not any(vars(args).values()):
        args.unit = args.integration = args.docs = True
    
    print("🌙 NightScan - Tests de Configuration Unifiée")
    print("=" * 60)
    
    if args.unit:
        print("\\n🧪 Tests unitaires")
        unittest.main(argv=[''], exit=False, verbosity=2)
    
    if args.integration:
        print("\\n🔗 Tests d'intégration")
        run_integration_tests()
    
    if args.docs:
        print("\\n📚 Génération de documentation")
        generate_documentation()
    
    print("\\n🎉 Tous les tests terminés")


if __name__ == "__main__":
    main()