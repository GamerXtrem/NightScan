# NightScan - Configuration Unifiée\n\n## Vue d'ensemble\n\nCe document décrit la configuration unifiée de NightScan qui remplace\nles multiples approches de configuration précédentes.\n\n## Variables d'environnement\n\n### Principales variables NIGHTSCAN_*\n\n| Variable | Description | Défaut | Exemple |\n|----------|-------------|--------|---------|\n| `NIGHTSCAN_ENV` | Environnement de déploiement | `development` | `production` |\n| `NIGHTSCAN_DB_HOST` | Hôte de la base de données | `localhost` | `db.example.com` |\n| `NIGHTSCAN_DB_PORT` | Port de la base de données | `5432` | `5432` |\n| `NIGHTSCAN_WEB_PORT` | Port du service web | `8000` | `8080` |\n| `NIGHTSCAN_LOG_LEVEL` | Niveau de logging | `INFO` | `DEBUG` |\n| `NIGHTSCAN_USE_GPU` | Utilisation du GPU | `true` | `false` |\n\n## Configuration par environnement\n\n### Development\n```json\n{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "nightscan",
    "username": "nightscan",
    "password": "",
    "pool_size": 20,
    "max_overflow": 0,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "sslmode": "prefer",
    "sslcert": null,
    "sslkey": null,
    "sslrootcert": null
  },
  "cache": {
    "host": "localhost",
    "port": 6379,
    "database": 0,
    "username": null,
    "password": null,
    "socket_timeout": 5,
    "socket_connect_timeout"...\n```\n\n## Migration depuis l'ancien système\n\n1. Utiliser `python unified_config.py migrate` pour migrer automatiquement\n2. Adapter le code avec `from config_compatibility import get_legacy_value`\n3. Mettre à jour les variables d'environnement avec le préfixe `NIGHTSCAN_`\n\n## Exemples d'utilisation\n\n```python\nfrom unified_config import get_config\n\nconfig = get_config()\ndb_url = config.get_database_url()\nweb_port = config.services.web_port\n```