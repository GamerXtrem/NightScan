#!/usr/bin/env python3
"""
Script de Nettoyage Automatique des Données par Rétention
Supprimer automatiquement les prédictions expirées selon les paliers de quota.
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from quota_manager import get_quota_manager
from config import get_config

# Configuration
config = get_config()
logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure logging for the cleanup script"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Format de logs structuré
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configuration du logger principal
    logging.basicConfig(
        level=level,
        handlers=[console_handler]
    )
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

class DataRetentionCleaner:
    """Gestionnaire du nettoyage automatique des données"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.quota_manager = get_quota_manager()
        self.stats = {
            'users_processed': 0,
            'total_deleted': 0,
            'total_size_deleted': 0,
            'errors': [],
            'notifications_sent': 0
        }
    
    def run_full_cleanup(self, user_id: int = None) -> Dict[str, Any]:
        """Exécute le nettoyage complet pour tous les utilisateurs ou un utilisateur spécifique"""
        start_time = datetime.now()
        logger.info(f"Starting data retention cleanup (dry_run={self.dry_run})")
        
        try:
            if user_id:
                # Nettoyage pour un utilisateur spécifique
                result = self._cleanup_user(user_id)
                self.stats['users_processed'] = 1
            else:
                # Nettoyage global
                result = self._cleanup_all_users()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            final_result = {
                'success': True,
                'dry_run': self.dry_run,
                'duration_seconds': duration,
                'stats': self.stats,
                'cleanup_result': result
            }
            
            logger.info(f"Cleanup completed in {duration:.2f}s: {self.stats}")
            return final_result
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }
    
    def _cleanup_all_users(self) -> Dict[str, Any]:
        """Nettoyage pour tous les utilisateurs avec des données expirées"""
        try:
            # Utiliser la fonction PostgreSQL pour un nettoyage global
            result = self.quota_manager.cleanup_expired_predictions(
                user_id=None, 
                dry_run=self.dry_run
            )
            
            self.stats['total_deleted'] = result.get('deleted_count', 0)
            self.stats['total_size_deleted'] = result.get('total_size_deleted_bytes', 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Global cleanup failed: {e}")
            self.stats['errors'].append(str(e))
            return {
                'success': False,
                'error': str(e),
                'deleted_count': 0
            }
    
    def _cleanup_user(self, user_id: int) -> Dict[str, Any]:
        """Nettoyage pour un utilisateur spécifique"""
        try:
            logger.info(f"Processing user {user_id}")
            
            # Vérifier les prédictions expirées
            expired_info = self.quota_manager.get_expired_predictions_count(user_id)
            
            if expired_info.get('error'):
                logger.warning(f"User {user_id}: {expired_info['error']}")
                return expired_info
            
            expired_count = expired_info.get('expired_predictions', 0)
            
            if expired_count == 0:
                logger.debug(f"User {user_id}: No expired predictions")
                return {
                    'success': True,
                    'user_id': user_id,
                    'deleted_count': 0,
                    'message': 'No expired predictions'
                }
            
            # Exécuter le nettoyage
            cleanup_result = self.quota_manager.cleanup_expired_predictions(
                user_id=user_id,
                dry_run=self.dry_run
            )
            
            if cleanup_result.get('success'):
                deleted_count = cleanup_result.get('deleted_count', 0)
                size_deleted = cleanup_result.get('total_size_deleted_bytes', 0)
                
                self.stats['total_deleted'] += deleted_count
                self.stats['total_size_deleted'] += size_deleted
                
                logger.info(f"User {user_id}: Cleaned {deleted_count} predictions ({size_deleted/1024/1024:.2f} MB)")
            
            return cleanup_result
            
        except Exception as e:
            error_msg = f"User {user_id} cleanup failed: {e}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return {
                'success': False,
                'user_id': user_id,
                'error': str(e)
            }
    
    def check_expiring_data_notifications(self, days_before: int = 7) -> Dict[str, Any]:
        """Vérifier et envoyer les notifications pour les données qui vont expirer"""
        logger.info(f"Checking for data expiring in {days_before} days")
        
        try:
            # Pour cette implémentation, on simule en vérifiant chaque utilisateur
            # En production, on pourrait avoir une requête SQL optimisée pour trouver tous les utilisateurs concernés
            
            notifications_sent = 0
            users_to_notify = []
            
            # Note: Dans une vraie implémentation, on récupérerait tous les user_ids de manière plus efficace
            # Pour l'instant, on se contente de cette structure
            
            return {
                'success': True,
                'notifications_sent': notifications_sent,
                'users_notified': users_to_notify,
                'days_before_expiry': days_before
            }
            
        except Exception as e:
            logger.error(f"Notification check failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'notifications_sent': 0
            }
    
    def generate_retention_report(self) -> Dict[str, Any]:
        """Génère un rapport détaillé sur l'état des rétentions"""
        try:
            logger.info("Generating retention report")
            
            # Statistiques globales via PostgreSQL
            # On utiliserait la fonction test_retention_system() créée dans la migration
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'dry_run_mode': self.dry_run,
                'cleanup_stats': self.stats,
                'retention_policies': self._get_retention_policies_summary(),
                'recommendations': self._get_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def _get_retention_policies_summary(self) -> List[Dict[str, Any]]:
        """Résumé des politiques de rétention"""
        try:
            plans = self.quota_manager.get_all_plans_with_retention()
            return [
                {
                    'plan_type': plan['plan_type'],
                    'plan_name': plan['plan_name'],
                    'retention_days': plan.get('data_retention_days', 30),
                    'retention_description': plan.get('retention_description', '30 jours'),
                    'monthly_quota': plan['monthly_quota'],
                    'price_monthly': plan['price_monthly']
                }
                for plan in plans
            ]
        except Exception as e:
            logger.error(f"Error getting retention policies: {e}")
            return []
    
    def _get_recommendations(self) -> List[str]:
        """Recommandations basées sur les statistiques de nettoyage"""
        recommendations = []
        
        if self.stats['total_deleted'] > 1000:
            recommendations.append("Considérer des notifications plus fréquentes aux utilisateurs pour les sensibiliser à la rétention")
        
        if self.stats['total_size_deleted'] > 1024 * 1024 * 1024:  # > 1GB
            recommendations.append("Volume important de données supprimées - vérifier si les paliers de rétention sont optimaux")
        
        if len(self.stats['errors']) > 0:
            recommendations.append("Des erreurs ont été détectées - vérifier les logs pour diagnostiquer les problèmes")
        
        if not recommendations:
            recommendations.append("Système de rétention fonctionne correctement")
        
        return recommendations

def main():
    """Point d'entrée principal du script"""
    parser = argparse.ArgumentParser(
        description="Nettoyage automatique des données selon les politiques de rétention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --dry-run                    # Test du nettoyage sans suppression
  %(prog)s --execute --user-id 123      # Nettoyage pour l'utilisateur 123
  %(prog)s --execute --all-users        # Nettoyage global
  %(prog)s --report                     # Génère un rapport de rétention
  %(prog)s --notify --days-before 3     # Vérifie les notifications à 3 jours
        """
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        default=True,
        help='Mode test - affiche ce qui serait supprimé sans rien supprimer (défaut)'
    )
    
    parser.add_argument(
        '--execute', 
        action='store_true',
        help='Exécuter réellement les suppressions (désactive dry-run)'
    )
    
    parser.add_argument(
        '--user-id', 
        type=int,
        help='Nettoyer uniquement pour un utilisateur spécifique'
    )
    
    parser.add_argument(
        '--all-users', 
        action='store_true',
        help='Nettoyer pour tous les utilisateurs'
    )
    
    parser.add_argument(
        '--report', 
        action='store_true',
        help='Générer un rapport de rétention sans nettoyage'
    )
    
    parser.add_argument(
        '--notify', 
        action='store_true',
        help='Vérifier et envoyer les notifications d\'expiration'
    )
    
    parser.add_argument(
        '--days-before', 
        type=int, 
        default=7,
        help='Nombre de jours avant expiration pour les notifications (défaut: 7)'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Mode verbeux'
    )
    
    parser.add_argument(
        '--log-file',
        help='Fichier de log (optionnel)'
    )
    
    parser.add_argument(
        '--output',
        help='Fichier de sortie pour le rapport JSON'
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    setup_logging(args.verbose, args.log_file)
    
    # Validation des arguments
    if args.execute:
        dry_run = False
        logger.warning("⚠️  Mode EXECUTION activé - les suppressions seront réelles!")
    else:
        dry_run = True
        logger.info("ℹ️  Mode DRY-RUN - aucune suppression ne sera effectuée")
    
    if not any([args.all_users, args.user_id, args.report, args.notify]):
        args.all_users = True  # Par défaut, traiter tous les utilisateurs
    
    # Initialisation du cleaner
    cleaner = DataRetentionCleaner(dry_run=dry_run)
    
    try:
        result = {}
        
        # Génération de rapport
        if args.report:
            logger.info("📊 Generating retention report...")
            result = cleaner.generate_retention_report()
        
        # Vérification des notifications
        elif args.notify:
            logger.info(f"📧 Checking expiring data notifications ({args.days_before} days)...")
            result = cleaner.check_expiring_data_notifications(args.days_before)
        
        # Nettoyage des données
        else:
            if args.user_id:
                logger.info(f"🧹 Starting cleanup for user {args.user_id}...")
                result = cleaner.run_full_cleanup(user_id=args.user_id)
            else:
                logger.info("🧹 Starting global cleanup...")
                result = cleaner.run_full_cleanup()
        
        # Affichage du résultat
        if result.get('success'):
            print("\n✅ Opération terminée avec succès!")
            
            # Statistiques de nettoyage
            if 'stats' in result:
                stats = result['stats']
                print(f"📈 Statistiques:")
                print(f"  - Utilisateurs traités: {stats.get('users_processed', 0)}")
                print(f"  - Prédictions supprimées: {stats.get('total_deleted', 0)}")
                print(f"  - Espace libéré: {stats.get('total_size_deleted', 0) / 1024 / 1024:.2f} MB")
                print(f"  - Erreurs: {len(stats.get('errors', []))}")
            
            # Informations spécifiques
            if 'cleanup_result' in result:
                cleanup = result['cleanup_result']
                if cleanup.get('deleted_count', 0) > 0:
                    print(f"🗑️  {cleanup['deleted_count']} prédictions supprimées")
                    print(f"💾 {cleanup.get('total_size_deleted_mb', 0)} MB libérés")
        
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
        logger.info("Opération interrompue par l'utilisateur")
        return 130
    
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        return 1

if __name__ == "__main__":
    exit(main())