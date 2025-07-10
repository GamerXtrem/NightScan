#!/usr/bin/env python3
"""
Script de Migration des Données Existantes vers le Système de Rétention
Migre les données existantes pour être compatibles avec les nouvelles politiques de rétention.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from quota_manager import get_quota_manager
from config import get_config

logger = logging.getLogger(__name__)
config = get_config()

class DataMigrator:
    """Gestionnaire de migration des données vers le système de rétention"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.quota_manager = get_quota_manager()
        self.migration_stats = {
            'users_processed': 0,
            'predictions_updated': 0,
            'plans_updated': 0,
            'errors': []
        }
    
    def run_full_migration(self) -> Dict[str, Any]:
        """Exécute la migration complète"""
        logger.info(f"Starting data migration (dry_run={self.dry_run})")
        
        try:
            # 1. Migrer les plans pour ajouter les périodes de rétention
            logger.info("Step 1: Migrating plan features...")
            self._migrate_plan_features()
            
            # 2. Mettre à jour les prédictions sans created_at
            logger.info("Step 2: Updating predictions without created_at...")
            self._update_predictions_created_at()
            
            # 3. Initialiser les quotas pour les utilisateurs existants
            logger.info("Step 3: Initializing user quotas...")
            self._initialize_user_quotas()
            
            # 4. Vérifier la cohérence des données
            logger.info("Step 4: Verifying data consistency...")
            self._verify_data_consistency()
            
            logger.info(f"Migration completed: {self.migration_stats}")
            return {
                'success': True,
                'dry_run': self.dry_run,
                'stats': self.migration_stats
            }
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.migration_stats
            }
    
    def _migrate_plan_features(self):
        """Migre les caractéristiques des plans pour ajouter les périodes de rétention"""
        try:
            # Définir les périodes de rétention par palier
            retention_policies = {
                'free': 7,        # 7 jours
                'premium': 30,    # 30 jours  
                'enterprise': 180 # 6 mois
            }
            
            for plan_type, retention_days in retention_policies.items():
                if not self.dry_run:
                    query = """
                    UPDATE plan_features 
                    SET data_retention_days = %s 
                    WHERE plan_type = %s AND (data_retention_days IS NULL OR data_retention_days = 30)
                    """
                    
                    rows_affected = self.quota_manager._execute_query(
                        query, (retention_days, plan_type)
                    )
                    
                    logger.info(f"Updated plan {plan_type}: {rows_affected} rows affected")
                    self.migration_stats['plans_updated'] += rows_affected or 0
                else:
                    logger.info(f"[DRY RUN] Would update plan {plan_type} with {retention_days} days retention")
            
            # Créer les plans par défaut s'ils n'existent pas
            self._ensure_default_plans_exist()
            
        except Exception as e:
            error_msg = f"Error migrating plan features: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
    
    def _ensure_default_plans_exist(self):
        """S'assure que les plans par défaut existent"""
        default_plans = [
            {
                'plan_type': 'free',
                'plan_name': 'Gratuit',
                'monthly_quota': 600,
                'data_retention_days': 7,
                'price_monthly_cents': 0,
                'max_file_size_mb': 50
            },
            {
                'plan_type': 'premium', 
                'plan_name': 'Premium',
                'monthly_quota': 3000,
                'data_retention_days': 30,
                'price_monthly_cents': 999,  # 9.99€
                'max_file_size_mb': 50
            },
            {
                'plan_type': 'enterprise',
                'plan_name': 'Enterprise', 
                'monthly_quota': 100000,
                'data_retention_days': 180,
                'price_monthly_cents': 4999,  # 49.99€
                'max_file_size_mb': 50
            }
        ]
        
        for plan in default_plans:
            try:
                if not self.dry_run:
                    query = """
                    INSERT INTO plan_features (
                        plan_type, plan_name, monthly_quota, data_retention_days,
                        price_monthly_cents, max_file_size_mb, max_concurrent_uploads,
                        priority_queue, advanced_analytics, api_access,
                        email_support, phone_support, is_active
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, 1, %s, %s, %s, %s, %s, true
                    ) ON CONFLICT (plan_type) DO UPDATE SET
                        data_retention_days = EXCLUDED.data_retention_days,
                        monthly_quota = EXCLUDED.monthly_quota,
                        price_monthly_cents = EXCLUDED.price_monthly_cents
                    """
                    
                    # Définir les features selon le plan
                    priority_queue = plan['plan_type'] != 'free'
                    advanced_analytics = plan['plan_type'] == 'enterprise'
                    api_access = plan['plan_type'] != 'free'
                    email_support = plan['plan_type'] != 'free'
                    phone_support = plan['plan_type'] == 'enterprise'
                    
                    self.quota_manager._execute_query(query, (
                        plan['plan_type'], plan['plan_name'], plan['monthly_quota'],
                        plan['data_retention_days'], plan['price_monthly_cents'],
                        plan['max_file_size_mb'], priority_queue, advanced_analytics,
                        api_access, email_support, phone_support
                    ))
                    
                    logger.info(f"Ensured plan {plan['plan_type']} exists")
                else:
                    logger.info(f"[DRY RUN] Would ensure plan {plan['plan_type']} exists")
                    
            except Exception as e:
                error_msg = f"Error ensuring plan {plan['plan_type']} exists: {e}"
                logger.error(error_msg)
                self.migration_stats['errors'].append(error_msg)
    
    def _update_predictions_created_at(self):
        """Met à jour les prédictions qui n'ont pas de created_at"""
        try:
            if not self.dry_run:
                # Mettre à jour les prédictions sans created_at avec une estimation basée sur l'ID
                query = """
                UPDATE prediction 
                SET created_at = CURRENT_TIMESTAMP - (id * INTERVAL '1 hour')
                WHERE created_at IS NULL
                """
                
                rows_affected = self.quota_manager._execute_query(query)
                logger.info(f"Updated {rows_affected} predictions with estimated created_at")
                self.migration_stats['predictions_updated'] = rows_affected or 0
            else:
                # Compter combien de prédictions auraient besoin d'être mises à jour
                query = "SELECT COUNT(*) as count FROM prediction WHERE created_at IS NULL"
                result = self.quota_manager._execute_query(query, fetch_one=True)
                count = result['count'] if result else 0
                logger.info(f"[DRY RUN] Would update {count} predictions with created_at")
                
        except Exception as e:
            error_msg = f"Error updating predictions created_at: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
    
    def _initialize_user_quotas(self):
        """Initialise les quotas pour les utilisateurs existants"""
        try:
            if not self.dry_run:
                # S'assurer que tous les utilisateurs ont un plan assigné
                query = """
                INSERT INTO user_plans (user_id, plan_type, status)
                SELECT u.id, 'free', 'active'
                FROM "user" u
                LEFT JOIN user_plans up ON u.id = up.user_id
                WHERE up.user_id IS NULL
                """
                
                rows_affected = self.quota_manager._execute_query(query)
                logger.info(f"Assigned free plan to {rows_affected} users without plans")
                
                # Initialiser les quotas mensuels
                query = """
                INSERT INTO quota_usage (user_id, month, year, prediction_count)
                SELECT u.id, 
                       EXTRACT(month FROM CURRENT_TIMESTAMP),
                       EXTRACT(year FROM CURRENT_TIMESTAMP),
                       0
                FROM "user" u
                LEFT JOIN quota_usage qu ON (
                    u.id = qu.user_id 
                    AND qu.month = EXTRACT(month FROM CURRENT_TIMESTAMP)
                    AND qu.year = EXTRACT(year FROM CURRENT_TIMESTAMP)
                )
                WHERE qu.user_id IS NULL
                """
                
                rows_affected = self.quota_manager._execute_query(query)
                logger.info(f"Initialized quota for {rows_affected} users")
                self.migration_stats['users_processed'] = rows_affected or 0
                
            else:
                # Compter les utilisateurs qui auraient besoin d'initialisation
                query = """
                SELECT COUNT(*) as count
                FROM "user" u
                LEFT JOIN user_plans up ON u.id = up.user_id
                WHERE up.user_id IS NULL
                """
                result = self.quota_manager._execute_query(query, fetch_one=True)
                count = result['count'] if result else 0
                logger.info(f"[DRY RUN] Would initialize {count} users without plans")
                
        except Exception as e:
            error_msg = f"Error initializing user quotas: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
    
    def _verify_data_consistency(self):
        """Vérifie la cohérence des données après migration"""
        try:
            checks = []
            
            # Vérifier que tous les plans ont des périodes de rétention
            query = """
            SELECT plan_type, data_retention_days 
            FROM plan_features 
            WHERE is_active = true AND (data_retention_days IS NULL OR data_retention_days <= 0)
            """
            result = self.quota_manager._execute_query(query, fetch_all=True)
            if result:
                checks.append(f"❌ {len(result)} plans without valid retention days")
            else:
                checks.append("✅ All active plans have valid retention days")
            
            # Vérifier que toutes les prédictions ont created_at
            query = "SELECT COUNT(*) as count FROM prediction WHERE created_at IS NULL"
            result = self.quota_manager._execute_query(query, fetch_one=True)
            null_count = result['count'] if result else 0
            if null_count > 0:
                checks.append(f"❌ {null_count} predictions without created_at")
            else:
                checks.append("✅ All predictions have created_at")
            
            # Vérifier que tous les utilisateurs ont un plan
            query = """
            SELECT COUNT(*) as count
            FROM "user" u
            LEFT JOIN user_plans up ON u.id = up.user_id
            WHERE up.user_id IS NULL
            """
            result = self.quota_manager._execute_query(query, fetch_one=True)
            users_without_plan = result['count'] if result else 0
            if users_without_plan > 0:
                checks.append(f"❌ {users_without_plan} users without assigned plan")
            else:
                checks.append("✅ All users have assigned plans")
            
            # Afficher les résultats de vérification
            logger.info("Data consistency check results:")
            for check in checks:
                logger.info(f"  {check}")
            
            return len([c for c in checks if c.startswith("❌")]) == 0
            
        except Exception as e:
            error_msg = f"Error verifying data consistency: {e}"
            logger.error(error_msg)
            self.migration_stats['errors'].append(error_msg)
            return False
    
    def generate_migration_report(self) -> Dict[str, Any]:
        """Génère un rapport de migration"""
        try:
            # Statistiques des plans
            query = "SELECT plan_type, plan_name, data_retention_days, monthly_quota FROM plan_features WHERE is_active = true"
            plans = self.quota_manager._execute_query(query, fetch_all=True)
            
            # Statistiques des utilisateurs
            query = """
            SELECT 
                up.plan_type,
                COUNT(*) as user_count
            FROM user_plans up
            WHERE up.status = 'active'
            GROUP BY up.plan_type
            """
            user_distribution = self.quota_manager._execute_query(query, fetch_all=True)
            
            # Statistiques des prédictions
            query = """
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(CASE WHEN created_at IS NOT NULL THEN 1 END) as predictions_with_date,
                MIN(created_at) as oldest_prediction,
                MAX(created_at) as newest_prediction
            FROM prediction
            """
            prediction_stats = self.quota_manager._execute_query(query, fetch_one=True)
            
            return {
                'migration_timestamp': datetime.now().isoformat(),
                'dry_run': self.dry_run,
                'migration_stats': self.migration_stats,
                'current_plans': [dict(plan) for plan in plans] if plans else [],
                'user_distribution': [dict(dist) for dist in user_distribution] if user_distribution else [],
                'prediction_stats': dict(prediction_stats) if prediction_stats else {},
                'verification_passed': len(self.migration_stats['errors']) == 0
            }
            
        except Exception as e:
            logger.error(f"Error generating migration report: {e}")
            return {
                'error': str(e),
                'migration_timestamp': datetime.now().isoformat()
            }

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Migration des données existantes vers le système de rétention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --dry-run                    # Test de migration sans modifications
  %(prog)s --execute                    # Exécution réelle de la migration
  %(prog)s --report                     # Génère un rapport de l'état actuel
        """
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        default=True,
        help='Mode test - montre ce qui serait fait sans rien modifier (défaut)'
    )
    
    parser.add_argument(
        '--execute', 
        action='store_true',
        help='Exécuter réellement la migration (désactive dry-run)'
    )
    
    parser.add_argument(
        '--report', 
        action='store_true',
        help='Générer uniquement un rapport de l\'état actuel'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Mode verbeux'
    )
    
    parser.add_argument(
        '--output',
        help='Fichier de sortie pour le rapport JSON'
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validation des arguments
    if args.execute:
        dry_run = False
        logger.warning("⚠️  Mode EXECUTION activé - les modifications seront réelles!")
        
        # Demander confirmation
        response = input("Voulez-vous vraiment exécuter la migration? (tapez 'oui' pour confirmer): ")
        if response.lower() != 'oui':
            print("Migration annulée.")
            return 1
    else:
        dry_run = True
        logger.info("ℹ️  Mode DRY-RUN - aucune modification ne sera effectuée")
    
    # Initialisation du migrateur
    migrator = DataMigrator(dry_run=dry_run)
    
    try:
        if args.report:
            # Générer uniquement un rapport
            logger.info("📊 Generating migration report...")
            result = migrator.generate_migration_report()
        else:
            # Exécuter la migration
            logger.info("🚀 Starting migration...")
            result = migrator.run_full_migration()
            
            # Ajouter le rapport au résultat
            report = migrator.generate_migration_report()
            result['report'] = report
        
        # Affichage du résultat
        if result.get('success', True):
            print("\n✅ Opération terminée avec succès!")
            
            if 'stats' in result:
                stats = result['stats']
                print(f"📈 Statistiques:")
                print(f"  - Utilisateurs traités: {stats.get('users_processed', 0)}")
                print(f"  - Prédictions mises à jour: {stats.get('predictions_updated', 0)}")
                print(f"  - Plans mis à jour: {stats.get('plans_updated', 0)}")
                print(f"  - Erreurs: {len(stats.get('errors', []))}")
            
            if result.get('verification_passed', True):
                print("✅ Vérification de cohérence réussie")
            else:
                print("⚠️ Problèmes détectés lors de la vérification")
        
        else:
            print(f"\n❌ Opération échouée: {result.get('error', 'Erreur inconnue')}")
            return 1
        
        # Sauvegarde du résultat si demandé
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Résultat sauvegardé dans {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Migration interrompue par l'utilisateur")
        return 130
    
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        return 1

if __name__ == "__main__":
    exit(main())