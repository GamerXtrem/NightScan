#!/usr/bin/env python3
"""
Système de Notifications de Rétention des Données
Gère les notifications aux utilisateurs concernant l'expiration de leurs données.
"""

import os
import sys
import logging
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from quota_manager import get_quota_manager
from config import get_config

logger = logging.getLogger(__name__)
config = get_config()

class RetentionNotificationService:
    """Service de gestion des notifications de rétention"""
    
    def __init__(self):
        self.quota_manager = get_quota_manager()
        self.notification_history = {}
        
        # Configuration email (à adapter selon votre configuration)
        self.smtp_config = {
            'host': os.getenv('SMTP_HOST', 'localhost'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME', ''),
            'password': os.getenv('SMTP_PASSWORD', ''),
            'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
            'from_email': os.getenv('SMTP_FROM_EMAIL', 'noreply@nightscan.local')
        }
    
    def check_and_send_notifications(self, days_before_expiry: int = 7) -> Dict[str, Any]:
        """Vérifie et envoie les notifications pour les données qui vont expirer"""
        logger.info(f"Checking for data expiring in {days_before_expiry} days")
        
        results = {
            'notifications_sent': 0,
            'notifications_failed': 0,
            'users_processed': 0,
            'errors': []
        }
        
        try:
            # Obtenir tous les utilisateurs avec des données qui vont expirer
            users_to_notify = self._get_users_with_expiring_data(days_before_expiry)
            
            for user_data in users_to_notify:
                try:
                    user_id = user_data['user_id']
                    results['users_processed'] += 1
                    
                    # Vérifier si on a déjà envoyé une notification récemment
                    if self._should_skip_notification(user_id, days_before_expiry):
                        logger.debug(f"Skipping notification for user {user_id} - already notified recently")
                        continue
                    
                    # Préparer et envoyer la notification
                    notification_data = self._prepare_notification_data(user_id, user_data)
                    
                    if self._send_notification(notification_data):
                        results['notifications_sent'] += 1
                        self._record_notification_sent(user_id, days_before_expiry)
                        logger.info(f"Notification sent to user {user_id}")
                    else:
                        results['notifications_failed'] += 1
                        logger.warning(f"Failed to send notification to user {user_id}")
                
                except Exception as e:
                    error_msg = f"Error processing user {user_data.get('user_id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    results['notifications_failed'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in notification check: {e}")
            results['errors'].append(str(e))
            return results
    
    def _get_users_with_expiring_data(self, days_before: int) -> List[Dict[str, Any]]:
        """Obtient la liste des utilisateurs avec des données qui vont expirer"""
        try:
            # Note: En production, cette requête devrait être optimisée pour éviter de checker chaque utilisateur
            # On pourrait utiliser une requête SQL directe pour trouver tous les utilisateurs concernés
            
            # Pour cette implémentation simplifiée, on retourne une liste vide
            # car on n'a pas de mécanisme pour obtenir tous les user_ids facilement
            
            # Une implémentation complète inclurait une requête comme :
            # SELECT DISTINCT p.user_id, up.plan_type, pf.data_retention_days, COUNT(p.id) as expiring_count
            # FROM prediction p
            # JOIN user_plans up ON p.user_id = up.user_id
            # JOIN plan_features pf ON up.plan_type = pf.plan_type
            # WHERE EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) BETWEEN (pf.data_retention_days - days_before) AND pf.data_retention_days
            # GROUP BY p.user_id, up.plan_type, pf.data_retention_days
            # HAVING COUNT(p.id) > 0
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting users with expiring data: {e}")
            return []
    
    def check_user_expiring_data(self, user_id: int, days_before: int = 7) -> Dict[str, Any]:
        """Vérifie et potentiellement notifie un utilisateur spécifique"""
        try:
            notification_check = self.quota_manager.notify_expiring_data(user_id, days_before)
            
            if notification_check.get('should_notify'):
                # Préparer les données de notification
                notification_data = self._prepare_notification_data(user_id, notification_check)
                
                # Envoyer la notification
                if self._send_notification(notification_data):
                    self._record_notification_sent(user_id, days_before)
                    return {
                        'success': True,
                        'notification_sent': True,
                        'user_id': user_id,
                        'expiring_count': notification_check.get('expiring_count', 0)
                    }
                else:
                    return {
                        'success': False,
                        'notification_sent': False,
                        'error': 'Failed to send notification'
                    }
            else:
                return {
                    'success': True,
                    'notification_sent': False,
                    'reason': 'No expiring data or notification not needed'
                }
                
        except Exception as e:
            logger.error(f"Error checking user {user_id} expiring data: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_notification_data(self, user_id: int, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare les données pour la notification"""
        
        # Obtenir les détails de l'utilisateur (email, etc.)
        # Note: En production, on récupérerait ces infos de la base de données
        user_email = f"user{user_id}@example.com"  # Placeholder
        
        expiring_count = user_data.get('expiring_count', 0)
        retention_days = user_data.get('retention_days', 30)
        plan_type = user_data.get('plan_type', 'free')
        
        # Obtenir les options d'upgrade
        upgrade_options = user_data.get('upgrade_options', [])
        
        return {
            'user_id': user_id,
            'user_email': user_email,
            'expiring_count': expiring_count,
            'retention_days': retention_days,
            'plan_type': plan_type,
            'upgrade_options': upgrade_options,
            'notification_type': 'data_expiring'
        }
    
    def _send_notification(self, notification_data: Dict[str, Any]) -> bool:
        """Envoie la notification (email, push, etc.)"""
        try:
            # Pour cette implémentation, on se contente d'un email
            return self._send_email_notification(notification_data)
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
    
    def _send_email_notification(self, data: Dict[str, Any]) -> bool:
        """Envoie une notification par email"""
        try:
            # Créer le message email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"NightScan - Vos données expirent bientôt"
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = data['user_email']
            
            # Créer le contenu HTML
            html_content = self._generate_email_html(data)
            
            # Créer le contenu texte
            text_content = self._generate_email_text(data)
            
            # Attacher les contenus
            part1 = MIMEText(text_content, 'plain', 'utf-8')
            part2 = MIMEText(html_content, 'html', 'utf-8')
            
            msg.attach(part1)
            msg.attach(part2)
            
            # Envoyer l'email
            if self.smtp_config['username']:  # Si SMTP configuré
                with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
                    if self.smtp_config['use_tls']:
                        server.starttls()
                    
                    if self.smtp_config['username'] and self.smtp_config['password']:
                        server.login(self.smtp_config['username'], self.smtp_config['password'])
                    
                    server.send_message(msg)
                
                logger.info(f"Email notification sent to {data['user_email']}")
                return True
            else:
                # Mode simulation
                logger.info(f"[SIMULATION] Email notification would be sent to {data['user_email']}")
                logger.debug(f"Email content: {text_content}")
                return True
                
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def _generate_email_html(self, data: Dict[str, Any]) -> str:
        """Génère le contenu HTML de l'email"""
        expiring_count = data['expiring_count']
        retention_days = data['retention_days']
        plan_type = data['plan_type']
        upgrade_options = data.get('upgrade_options', [])
        
        upgrade_html = ""
        if upgrade_options:
            upgrade_html = "<h3>💎 Prolongez la rétention de vos données</h3>"
            for option in upgrade_options[:2]:  # Montrer max 2 options
                upgrade_html += f"""
                <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px;">
                    <strong>{option['plan_name']}</strong><br>
                    Rétention: {option['retention_description']}<br>
                    Prix: {option['price_monthly']}€/mois<br>
                    <a href="https://nightscan.local/dashboard#subscription" style="color: #007bff;">Voir ce plan</a>
                </div>
                """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #667eea;">🌙 NightScan - Expiration de vos données</h1>
                
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h2 style="margin-top: 0;">⚠️ Action requise</h2>
                    <p><strong>{expiring_count} de vos prédictions</strong> vont expirer dans les prochains jours.</p>
                    <p>Selon votre plan actuel ({plan_type}), vos données sont conservées pendant <strong>{retention_days} jours</strong>.</p>
                </div>
                
                <h3>🎯 Que faire maintenant ?</h3>
                <ul>
                    <li><strong>Exporter vos données</strong> importantes avant leur suppression</li>
                    <li><strong>Considérer une mise à niveau</strong> pour une rétention plus longue</li>
                    <li><strong>Vérifier vos prédictions</strong> sur votre tableau de bord</li>
                </ul>
                
                {upgrade_html}
                
                <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="margin-top: 0;">🔗 Liens utiles</h3>
                    <p>
                        <a href="https://nightscan.local/data-retention" style="color: #007bff;">Gérer vos données</a> | 
                        <a href="https://nightscan.local/dashboard" style="color: #007bff;">Tableau de bord</a> |
                        <a href="https://nightscan.local/api/v1/predictions?format=csv" style="color: #007bff;">Exporter vos données</a>
                    </p>
                </div>
                
                <hr style="margin: 30px 0;">
                <p style="font-size: 0.9em; color: #666;">
                    Cet email a été envoyé automatiquement par le système NightScan.<br>
                    Pour modifier vos préférences de notification, visitez votre tableau de bord.
                </p>
            </div>
        </body>
        </html>
        """
    
    def _generate_email_text(self, data: Dict[str, Any]) -> str:
        """Génère le contenu texte de l'email"""
        expiring_count = data['expiring_count']
        retention_days = data['retention_days']
        plan_type = data['plan_type']
        
        return f"""
NightScan - Expiration de vos données
=====================================

⚠️ ACTION REQUISE

{expiring_count} de vos prédictions vont expirer dans les prochains jours.

Selon votre plan actuel ({plan_type}), vos données sont conservées pendant {retention_days} jours.

Que faire maintenant ?
- Exporter vos données importantes avant leur suppression
- Considérer une mise à niveau pour une rétention plus longue
- Vérifier vos prédictions sur votre tableau de bord

Liens utiles :
- Gestion des données : https://nightscan.local/data-retention
- Tableau de bord : https://nightscan.local/dashboard
- Export des données : https://nightscan.local/api/v1/predictions?format=csv

Cet email a été envoyé automatiquement par le système NightScan.
        """
    
    def _should_skip_notification(self, user_id: int, days_before: int) -> bool:
        """Vérifie si on doit ignorer la notification (déjà envoyée récemment)"""
        # Vérifier si on a envoyé une notification dans les dernières 24h
        notification_key = f"{user_id}_{days_before}"
        last_sent = self.notification_history.get(notification_key)
        
        if last_sent:
            time_since_last = datetime.now() - last_sent
            return time_since_last < timedelta(hours=24)
        
        return False
    
    def _record_notification_sent(self, user_id: int, days_before: int):
        """Enregistre qu'une notification a été envoyée"""
        notification_key = f"{user_id}_{days_before}"
        self.notification_history[notification_key] = datetime.now()
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des notifications"""
        return {
            'notifications_sent_today': len([
                timestamp for timestamp in self.notification_history.values()
                if (datetime.now() - timestamp).days == 0
            ]),
            'total_notifications_tracked': len(self.notification_history),
            'smtp_configured': bool(self.smtp_config['username'])
        }

# Instance globale
_notification_service = None

def get_notification_service() -> RetentionNotificationService:
    """Obtient l'instance globale du service de notifications"""
    global _notification_service
    if _notification_service is None:
        _notification_service = RetentionNotificationService()
    return _notification_service

# Script principal pour les tests
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test du système de notifications de rétention")
    parser.add_argument('--user-id', type=int, help='Tester pour un utilisateur spécifique')
    parser.add_argument('--days-before', type=int, default=7, help='Jours avant expiration')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbeux')
    
    args = parser.parse_args()
    
    # Configuration du logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    service = get_notification_service()
    
    if args.user_id:
        print(f"Test de notification pour l'utilisateur {args.user_id}")
        result = service.check_user_expiring_data(args.user_id, args.days_before)
        print(f"Résultat: {json.dumps(result, indent=2)}")
    else:
        print("Test global de vérification des notifications")
        result = service.check_and_send_notifications(args.days_before)
        print(f"Résultat: {json.dumps(result, indent=2)}")
    
    stats = service.get_notification_stats()
    print(f"Statistiques: {json.dumps(stats, indent=2)}")