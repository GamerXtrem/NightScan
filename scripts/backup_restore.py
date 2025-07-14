#!/usr/bin/env python3
"""
Script restoration backups NightScan
Restoration complète ou sélective des backups
"""

import os
import sys
import subprocess
import shutil
import tempfile
import gzip
import tarfile
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

class NightScanRestore:
    """Gestionnaire restoration backups NightScan"""
    
    def __init__(self, backup_dir="/opt/nightscan/backups"):
        self.backup_dir = Path(backup_dir)
        self.temp_dir = None
        
    def list_available_backups(self):
        """Lister backups disponibles"""
        print("🔍 BACKUPS DISPONIBLES")
        print("=" * 50)
        
        backups = {}
        
        # Backups quotidiens
        daily_dir = self.backup_dir / "daily"
        if daily_dir.exists():
            daily_backups = []
            for backup_file in daily_dir.glob("database_*.sql.gz"):
                timestamp = backup_file.stem.split("_")[1]
                date_obj = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                daily_backups.append({
                    'timestamp': timestamp,
                    'date': date_obj,
                    'file': backup_file,
                    'type': 'daily'
                })
            
            daily_backups.sort(key=lambda x: x['date'], reverse=True)
            backups['daily'] = daily_backups[:10]  # 10 plus récents
        
        # Backups hebdomadaires
        weekly_dir = self.backup_dir / "weekly"
        if weekly_dir.exists():
            weekly_backups = []
            for week_dir in weekly_dir.iterdir():
                if week_dir.is_dir():
                    db_file = week_dir / "database_*.sql.gz"
                    db_files = list(week_dir.glob("database_*.sql.gz"))
                    if db_files:
                        weekly_backups.append({
                            'week': week_dir.name,
                            'path': week_dir,
                            'type': 'weekly'
                        })
            
            backups['weekly'] = weekly_backups[-4:]  # 4 plus récents
        
        # Backups mensuels
        monthly_dir = self.backup_dir / "monthly"
        if monthly_dir.exists():
            monthly_backups = []
            for month_dir in monthly_dir.iterdir():
                if month_dir.is_dir():
                    db_files = list(month_dir.glob("database_*.sql.gz"))
                    if db_files:
                        monthly_backups.append({
                            'month': month_dir.name,
                            'path': month_dir,
                            'type': 'monthly'
                        })
            
            backups['monthly'] = monthly_backups[-6:]  # 6 plus récents
        
        # Affichage
        if 'daily' in backups:
            print("\n📅 BACKUPS QUOTIDIENS:")
            for i, backup in enumerate(backups['daily']):
                age = datetime.now() - backup['date']
                print(f"  {i+1:2d}. {backup['date'].strftime('%Y-%m-%d %H:%M')} "
                      f"({age.days} jours)")
        
        if 'weekly' in backups:
            print("\n📅 BACKUPS HEBDOMADAIRES:")
            for i, backup in enumerate(backups['weekly']):
                print(f"  {i+1:2d}. Semaine {backup['week']}")
        
        if 'monthly' in backups:
            print("\n📅 BACKUPS MENSUELS:")
            for i, backup in enumerate(backups['monthly']):
                print(f"  {i+1:2d}. Mois {backup['month']}")
        
        return backups
    
    def select_backup(self, backups, backup_type=None, index=None):
        """Sélectionner backup à restaurer"""
        if backup_type and index:
            # Sélection automatique
            if backup_type in backups and index <= len(backups[backup_type]):
                return backups[backup_type][index - 1]
            else:
                raise ValueError(f"Backup {backup_type}#{index} non trouvé")
        
        # Sélection interactive
        print("\n🎯 SÉLECTION BACKUP")
        print("1. Quotidien")
        print("2. Hebdomadaire") 
        print("3. Mensuel")
        
        choice = input("Type de backup (1-3): ").strip()
        
        if choice == "1" and 'daily' in backups:
            print("\nBackups quotidiens:")
            for i, backup in enumerate(backups['daily']):
                print(f"  {i+1}. {backup['date'].strftime('%Y-%m-%d %H:%M')}")
            
            idx = int(input("Numéro du backup: ")) - 1
            return backups['daily'][idx]
            
        elif choice == "2" and 'weekly' in backups:
            print("\nBackups hebdomadaires:")
            for i, backup in enumerate(backups['weekly']):
                print(f"  {i+1}. Semaine {backup['week']}")
            
            idx = int(input("Numéro du backup: ")) - 1
            return backups['weekly'][idx]
            
        elif choice == "3" and 'monthly' in backups:
            print("\nBackups mensuels:")
            for i, backup in enumerate(backups['monthly']):
                print(f"  {i+1}. Mois {backup['month']}")
            
            idx = int(input("Numéro du backup: ")) - 1
            return backups['monthly'][idx]
        
        else:
            raise ValueError("Sélection invalide")
    
    def prepare_restore_environment(self):
        """Préparer environnement restoration"""
        print("🔧 Préparation environnement restoration...")
        
        # Créer répertoire temporaire
        self.temp_dir = Path(tempfile.mkdtemp(prefix="nightscan_restore_"))
        print(f"Répertoire temporaire: {self.temp_dir}")
        
        # Vérifier services actifs
        active_services = []
        
        # Vérifier PostgreSQL
        try:
            subprocess.run(['pg_isready'], check=True, capture_output=True)
            active_services.append('postgresql')
        except subprocess.CalledProcessError:
            print("⚠️ PostgreSQL non accessible")
        
        # Vérifier Redis
        try:
            subprocess.run(['redis-cli', 'ping'], check=True, capture_output=True)
            active_services.append('redis')
        except subprocess.CalledProcessError:
            print("⚠️ Redis non accessible")
        
        print(f"Services actifs: {', '.join(active_services)}")
        return active_services
    
    def backup_current_state(self):
        """Backup état actuel avant restoration"""
        print("💾 Backup état actuel avant restoration...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pre_restore_dir = self.backup_dir / f"pre_restore_{timestamp}"
        pre_restore_dir.mkdir(exist_ok=True)
        
        # Backup database actuelle
        try:
            db_backup = pre_restore_dir / "current_database.sql.gz"
            cmd = f"pg_dump {os.getenv('NIGHTSCAN_DATABASE_URI')} | gzip > {db_backup}"
            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ Database sauvegardée: {db_backup}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Échec backup database: {e}")
        
        # Backup Redis actuel
        try:
            redis_backup = pre_restore_dir / "current_redis.rdb"
            subprocess.run(['redis-cli', '--rdb', str(redis_backup)], check=True)
            print(f"✅ Redis sauvegardé: {redis_backup}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Échec backup Redis: {e}")
        
        return pre_restore_dir
    
    def restore_database(self, backup_info, target_db=None):
        """Restaurer base de données"""
        print("🗄️ Restoration base de données...")
        
        # Déterminer fichier backup
        if backup_info['type'] == 'daily':
            db_file = backup_info['file']
        else:
            # Chercher fichier database dans répertoire
            db_files = list(backup_info['path'].glob("database_*.sql.gz"))
            if not db_files:
                raise FileNotFoundError("Fichier database non trouvé")
            db_file = db_files[0]
        
        if not db_file.exists():
            raise FileNotFoundError(f"Backup database non trouvé: {db_file}")
        
        # Base données cible
        if not target_db:
            target_db = os.getenv('NIGHTSCAN_DATABASE_URI')
            if not target_db:
                raise ValueError("NIGHTSCAN_DATABASE_URI non défini")
        
        print(f"Fichier backup: {db_file}")
        print(f"Base cible: {target_db.split('@')[-1] if '@' in target_db else target_db}")
        
        # Confirmation
        confirm = input("Confirmer restoration database? (oui/non): ").strip().lower()
        if confirm not in ['oui', 'o', 'yes', 'y']:
            print("Restoration annulée")
            return False
        
        try:
            # Extraction et restoration
            print("Extraction et restoration en cours...")
            cmd = f"zcat {db_file} | psql {target_db}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Database restaurée avec succès")
                return True
            else:
                print(f"❌ Échec restoration database:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Erreur restoration database: {e}")
            return False
    
    def restore_redis(self, backup_info):
        """Restaurer Redis"""
        print("💾 Restoration Redis...")
        
        # Déterminer fichier backup
        if backup_info['type'] == 'daily':
            backup_dir = backup_info['file'].parent
        else:
            backup_dir = backup_info['path']
        
        redis_files = list(backup_dir.glob("redis_*.rdb.gz"))
        if not redis_files:
            print("⚠️ Backup Redis non trouvé")
            return False
        
        redis_file = redis_files[0]
        print(f"Fichier backup Redis: {redis_file}")
        
        # Confirmation
        confirm = input("Confirmer restoration Redis? (oui/non): ").strip().lower()
        if confirm not in ['oui', 'o', 'yes', 'y']:
            print("Restoration Redis annulée")
            return False
        
        try:
            # Arrêter Redis temporairement
            print("Arrêt Redis...")
            subprocess.run(['redis-cli', 'SHUTDOWN', 'NOSAVE'], check=False)
            
            # Décompresser et copier RDB
            with gzip.open(redis_file, 'rb') as f_in:
                with open('/var/lib/redis/dump.rdb', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Redémarrer Redis
            print("Redémarrage Redis...")
            subprocess.run(['systemctl', 'start', 'redis'], check=True)
            
            # Vérifier
            subprocess.run(['redis-cli', 'ping'], check=True)
            print("✅ Redis restauré avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur restoration Redis: {e}")
            # Redémarrer Redis en cas d'échec
            subprocess.run(['systemctl', 'start', 'redis'], check=False)
            return False
    
    def restore_user_data(self, backup_info, target_dir="/opt/nightscan"):
        """Restaurer données utilisateur"""
        print("📁 Restoration données utilisateur...")
        
        # Déterminer fichier backup
        if backup_info['type'] == 'daily':
            backup_dir = backup_info['file'].parent
        else:
            backup_dir = backup_info['path']
        
        data_files = list(backup_dir.glob("user_data_*.tar.gz"))
        if not data_files:
            print("⚠️ Backup données utilisateur non trouvé")
            return False
        
        data_file = data_files[0]
        print(f"Fichier backup: {data_file}")
        
        # Confirmation
        confirm = input("Confirmer restoration données utilisateur? (oui/non): ").strip().lower()
        if confirm not in ['oui', 'o', 'yes', 'y']:
            print("Restoration données annulée")
            return False
        
        try:
            # Backup répertoires existants
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_existing = Path(f"/tmp/nightscan_existing_{timestamp}")
            
            for subdir in ['uploads', 'models', 'config']:
                src = Path(target_dir) / subdir
                if src.exists():
                    dst = backup_existing / subdir
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src, dst)
            
            print(f"Données existantes sauvegardées dans: {backup_existing}")
            
            # Extraction
            print("Extraction données utilisateur...")
            with tarfile.open(data_file, 'r:gz') as tar:
                tar.extractall(path='/')
            
            print("✅ Données utilisateur restaurées")
            return True
            
        except Exception as e:
            print(f"❌ Erreur restoration données: {e}")
            return False
    
    def restore_system_config(self, backup_info):
        """Restaurer configuration système"""
        print("⚙️ Restoration configuration système...")
        
        # Déterminer fichier backup
        if backup_info['type'] == 'daily':
            backup_dir = backup_info['file'].parent
        else:
            backup_dir = backup_info['path']
        
        config_files = list(backup_dir.glob("system_config_*.tar.gz"))
        if not config_files:
            print("⚠️ Backup configuration système non trouvé")
            return False
        
        config_file = config_files[0]
        print(f"Fichier backup: {config_file}")
        
        # Confirmation (critique)
        print("⚠️ ATTENTION: Restoration configuration système critique")
        confirm = input("Confirmer restoration config système? (oui/non): ").strip().lower()
        if confirm not in ['oui', 'o', 'yes', 'y']:
            print("Restoration configuration annulée")
            return False
        
        try:
            # Extraction avec précaution
            print("Extraction configuration système...")
            with tarfile.open(config_file, 'r:gz') as tar:
                # Lister contenu avant extraction
                members = tar.getmembers()
                print("Fichiers à restaurer:")
                for member in members:
                    print(f"  {member.name}")
                
                # Confirmation finale
                final_confirm = input("Procéder à l'extraction? (oui/non): ").strip().lower()
                if final_confirm in ['oui', 'o', 'yes', 'y']:
                    tar.extractall(path='/')
                    print("✅ Configuration système restaurée")
                    print("⚠️ Redémarrage services requis")
                    return True
                else:
                    print("Extraction annulée")
                    return False
            
        except Exception as e:
            print(f"❌ Erreur restoration config: {e}")
            return False
    
    def verify_restoration(self):
        """Vérifier restoration"""
        print("🔍 Vérification restoration...")
        
        checks = []
        
        # Test connexion database
        try:
            db_uri = os.getenv('NIGHTSCAN_DATABASE_URI')
            if db_uri:
                import psycopg2
                conn = psycopg2.connect(db_uri)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM information_schema.tables")
                table_count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                checks.append(f"✅ Database: {table_count} tables")
            else:
                checks.append("⚠️ Database: URI non défini")
        except Exception as e:
            checks.append(f"❌ Database: {e}")
        
        # Test Redis
        try:
            import redis
            r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            info = r.info()
            checks.append(f"✅ Redis: {info.get('used_memory_human', 'OK')}")
        except Exception as e:
            checks.append(f"❌ Redis: {e}")
        
        # Test fichiers
        critical_files = [
            '/opt/nightscan/config/current.json',
            '/opt/nightscan/unified_config.py',
            '/opt/nightscan/web/app.py'
        ]
        
        for file_path in critical_files:
            if Path(file_path).exists():
                checks.append(f"✅ Fichier: {file_path}")
            else:
                checks.append(f"❌ Fichier manquant: {file_path}")
        
        print("\nRésultats vérification:")
        for check in checks:
            print(f"  {check}")
        
        # Score global
        success_count = sum(1 for check in checks if check.startswith("✅"))
        total_count = len(checks)
        score = (success_count / total_count) * 100
        
        print(f"\n📊 Score restoration: {score:.1f}% ({success_count}/{total_count})")
        
        return score >= 80
    
    def cleanup_temp_files(self):
        """Nettoyage fichiers temporaires"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Nettoyage: {self.temp_dir}")

def main():
    parser = argparse.ArgumentParser(description='Restoration backups NightScan')
    parser.add_argument('--list', action='store_true', 
                       help='Lister backups disponibles')
    parser.add_argument('--type', choices=['daily', 'weekly', 'monthly'],
                       help='Type de backup à restaurer')
    parser.add_argument('--index', type=int,
                       help='Index du backup à restaurer')
    parser.add_argument('--database-only', action='store_true',
                       help='Restaurer seulement la database')
    parser.add_argument('--redis-only', action='store_true',
                       help='Restaurer seulement Redis')
    parser.add_argument('--data-only', action='store_true',
                       help='Restaurer seulement les données utilisateur')
    parser.add_argument('--target-db', 
                       help='Base de données cible pour restoration')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulation sans modifications réelles')
    
    args = parser.parse_args()
    
    restore = NightScanRestore()
    
    try:
        # Lister backups
        backups = restore.list_available_backups()
        
        if args.list:
            return 0
        
        if not any(backups.values()):
            print("❌ Aucun backup disponible")
            return 1
        
        # Sélectionner backup
        selected_backup = restore.select_backup(backups, args.type, args.index)
        
        print(f"\n🎯 BACKUP SÉLECTIONNÉ:")
        if selected_backup['type'] == 'daily':
            print(f"   Date: {selected_backup['date'].strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"   Période: {selected_backup.get('week', selected_backup.get('month'))}")
        print(f"   Type: {selected_backup['type']}")
        
        if args.dry_run:
            print("\n🔍 MODE SIMULATION - Aucune modification")
            return 0
        
        # Préparation
        restore.prepare_restore_environment()
        pre_restore_backup = restore.backup_current_state()
        
        print(f"\n💾 Backup pré-restoration: {pre_restore_backup}")
        
        # Restoration sélective ou complète
        success = True
        
        if args.database_only:
            success = restore.restore_database(selected_backup, args.target_db)
        elif args.redis_only:
            success = restore.restore_redis(selected_backup)
        elif args.data_only:
            success = restore.restore_user_data(selected_backup)
        else:
            # Restoration complète
            print("\n🚀 RESTORATION COMPLÈTE")
            print("=" * 50)
            
            success &= restore.restore_database(selected_backup, args.target_db)
            success &= restore.restore_redis(selected_backup)
            success &= restore.restore_user_data(selected_backup)
            
            # Configuration système optionnelle
            config_confirm = input("\nRestaurer configuration système? (oui/non): ").strip().lower()
            if config_confirm in ['oui', 'o', 'yes', 'y']:
                success &= restore.restore_system_config(selected_backup)
        
        # Vérification
        if success:
            print("\n🔍 VÉRIFICATION FINALE")
            print("=" * 50)
            verification_ok = restore.verify_restoration()
            
            if verification_ok:
                print("\n🎉 RESTORATION RÉUSSIE!")
                print("Redémarrage des services recommandé")
            else:
                print("\n⚠️ RESTORATION PARTIELLE")
                print("Vérification manuelle requise")
        else:
            print("\n❌ ÉCHEC RESTORATION")
            print(f"Backup pré-restoration disponible: {pre_restore_backup}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Restoration interrompue")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur restoration: {e}")
        return 1
    finally:
        restore.cleanup_temp_files()

if __name__ == '__main__':
    sys.exit(main())