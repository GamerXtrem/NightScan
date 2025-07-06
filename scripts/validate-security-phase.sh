#!/bin/bash

# Script de validation Phase 1 - Sécurité critique
# Usage: ./validate-security-phase.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Variables de validation
VALIDATION_PASSED=true
CRITICAL_ISSUES=0
HIGH_ISSUES=0
MEDIUM_ISSUES=0

# Validation 1: Vérifier que les secrets sont générés
validate_secrets_generated() {
    log "🔐 Validation 1: Vérification des secrets générés..."
    
    if [ ! -f "$PROJECT_ROOT/secrets/production/.env" ]; then
        error "Fichier secrets/production/.env manquant"
        VALIDATION_PASSED=false
        CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
        return 1
    fi
    
    # Vérifier que les secrets ne sont pas par défaut
    if grep -q "nightscan_secret\|redis_secret\|your-secret-key-here" "$PROJECT_ROOT/secrets/production/.env"; then
        error "Secrets par défaut détectés dans .env"
        VALIDATION_PASSED=false
        CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
        return 1
    fi
    
    # Vérifier les permissions
    PERMS=$(stat -c %a "$PROJECT_ROOT/secrets/production/.env" 2>/dev/null || stat -f %A "$PROJECT_ROOT/secrets/production/.env" 2>/dev/null)
    if [ "$PERMS" != "600" ]; then
        warn "Permissions fichier secrets: $PERMS (recommandé: 600)"
        MEDIUM_ISSUES=$((MEDIUM_ISSUES + 1))
    fi
    
    success "Secrets générés et sécurisés"
    return 0
}

# Validation 2: Vérifier que les secrets hardcodés sont supprimés
validate_hardcoded_secrets_removed() {
    log "🧹 Validation 2: Vérification suppression secrets hardcodés..."
    
    HARDCODED_COUNT=0
    
    # Rechercher dans les fichiers critiques
    CRITICAL_FILES=("docker-compose.production.yml" "config.py" "web/app.py")
    
    for file in "${CRITICAL_FILES[@]}"; do
        if [ -f "$PROJECT_ROOT/$file" ]; then
            if grep -q "nightscan_secret\|redis_secret" "$PROJECT_ROOT/$file"; then
                error "Secrets hardcodés détectés dans $file"
                HARDCODED_COUNT=$((HARDCODED_COUNT + 1))
            fi
        fi
    done
    
    if [ $HARDCODED_COUNT -gt 0 ]; then
        error "$HARDCODED_COUNT fichiers avec secrets hardcodés"
        VALIDATION_PASSED=false
        CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
        return 1
    fi
    
    success "Aucun secret hardcodé détecté dans les fichiers critiques"
    return 0
}

# Validation 3: Vérifier Docker Compose sécurisé
validate_docker_compose_security() {
    log "🐳 Validation 3: Sécurité Docker Compose..."
    
    COMPOSE_FILE="$PROJECT_ROOT/docker-compose.production.yml"
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        error "docker-compose.production.yml manquant"
        VALIDATION_PASSED=false
        HIGH_ISSUES=$((HIGH_ISSUES + 1))
        return 1
    fi
    
    # Vérifier env_file
    if ! grep -q "env_file:" "$COMPOSE_FILE"; then
        warn "env_file non configuré dans docker-compose"
        HIGH_ISSUES=$((HIGH_ISSUES + 1))
    fi
    
    # Vérifier les limites de ressources
    if ! grep -q "mem_limit:" "$COMPOSE_FILE"; then
        warn "Limites mémoire non configurées"
        MEDIUM_ISSUES=$((MEDIUM_ISSUES + 1))
    fi
    
    # Vérifier security_opt
    if ! grep -q "security_opt:" "$COMPOSE_FILE"; then
        warn "Options de sécurité Docker non configurées"
        HIGH_ISSUES=$((HIGH_ISSUES + 1))
    fi
    
    success "Docker Compose partiellement sécurisé"
    return 0
}

# Validation 4: Vérifier .gitignore pour secrets
validate_gitignore() {
    log "📝 Validation 4: Configuration .gitignore..."
    
    if [ ! -f "$PROJECT_ROOT/.gitignore" ]; then
        warn ".gitignore manquant"
        MEDIUM_ISSUES=$((MEDIUM_ISSUES + 1))
        return 1
    fi
    
    if ! grep -q "secrets/" "$PROJECT_ROOT/.gitignore"; then
        error "Répertoire secrets/ non ignoré par Git"
        VALIDATION_PASSED=false
        CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
        return 1
    fi
    
    success "Secrets correctement ignorés par Git"
    return 0
}

# Validation 5: Audit de sécurité final
validate_security_audit() {
    log "🔍 Validation 5: Audit de sécurité final..."
    
    if [ -f "$PROJECT_ROOT/scripts/security-audit.py" ]; then
        # Exécuter audit et capturer le score
        if python "$PROJECT_ROOT/scripts/security-audit.py" --project-root "$PROJECT_ROOT" > /tmp/security_audit.log 2>&1; then
            AUDIT_EXIT_CODE=0
        else
            AUDIT_EXIT_CODE=$?
        fi
        
        # Analyser les résultats
        case $AUDIT_EXIT_CODE in
            0)
                success "Audit de sécurité: EXCELLENT (0 problème)"
                ;;
            1)
                warn "Audit de sécurité: MOYEN (problèmes mineurs)"
                MEDIUM_ISSUES=$((MEDIUM_ISSUES + 1))
                ;;
            2)
                warn "Audit de sécurité: ÉLEVÉ (corrections nécessaires)"
                HIGH_ISSUES=$((HIGH_ISSUES + 1))
                ;;
            3)
                error "Audit de sécurité: CRITIQUE (corrections urgentes)"
                VALIDATION_PASSED=false
                CRITICAL_ISSUES=$((CRITICAL_ISSUES + 1))
                ;;
        esac
    else
        warn "Script d'audit de sécurité non trouvé"
        MEDIUM_ISSUES=$((MEDIUM_ISSUES + 1))
    fi
    
    return 0
}

# Validation 6: Vérifier les permissions des scripts
validate_script_permissions() {
    log "🔐 Validation 6: Permissions des scripts..."
    
    SCRIPTS=(
        "scripts/setup-secrets.sh"
        "scripts/deploy-vps-lite.sh"
        "scripts/security-audit.py"
    )
    
    for script in "${SCRIPTS[@]}"; do
        if [ -f "$PROJECT_ROOT/$script" ]; then
            if [ ! -x "$PROJECT_ROOT/$script" ]; then
                warn "$script n'est pas exécutable"
                MEDIUM_ISSUES=$((MEDIUM_ISSUES + 1))
            fi
        fi
    done
    
    success "Permissions des scripts vérifiées"
    return 0
}

# Calculer le score de sécurité
calculate_security_score() {
    local max_score=10
    local penalty=$((CRITICAL_ISSUES * 3 + HIGH_ISSUES * 2 + MEDIUM_ISSUES * 1))
    local score=$((max_score - penalty))
    
    if [ $score -lt 0 ]; then
        score=0
    fi
    
    echo $score
}

# Fonction principale
main() {
    log "🛡️  VALIDATION PHASE 1 - SÉCURITÉ CRITIQUE"
    log "=========================================="
    
    cd "$PROJECT_ROOT"
    
    # Exécuter toutes les validations
    validate_secrets_generated
    validate_hardcoded_secrets_removed  
    validate_docker_compose_security
    validate_gitignore
    validate_security_audit
    validate_script_permissions
    
    # Calculer le score final
    SECURITY_SCORE=$(calculate_security_score)
    
    echo ""
    log "📊 RÉSULTATS DE LA VALIDATION"
    echo "============================="
    echo "🔴 Critiques:  $CRITICAL_ISSUES"
    echo "🟠 Élevés:     $HIGH_ISSUES"  
    echo "🟡 Moyens:     $MEDIUM_ISSUES"
    echo ""
    echo "🎯 Score de sécurité: $SECURITY_SCORE/10"
    
    # Déterminer le statut final
    if [ "$VALIDATION_PASSED" = true ] && [ $SECURITY_SCORE -ge 8 ]; then
        echo ""
        success "🎉 PHASE 1 VALIDÉE - Sécurité critique réussie!"
        echo "✅ GATE_PASSED=true"
        echo ""
        echo "📋 Critères validés:"
        echo "  ✅ Secrets sécurisés générés"
        echo "  ✅ Secrets hardcodés supprimés"
        echo "  ✅ Docker Compose sécurisé"
        echo "  ✅ Score sécurité ≥ 8/10"
        echo ""
        echo "🚀 Prêt pour Phase 2 - Infrastructure"
        exit 0
    else
        echo ""
        error "❌ PHASE 1 ÉCHOUÉE - Corrections requises"
        echo "❌ GATE_PASSED=false"
        echo ""
        echo "📋 Actions requises:"
        
        if [ $CRITICAL_ISSUES -gt 0 ]; then
            echo "  🔴 Corriger $CRITICAL_ISSUES problème(s) CRITIQUE(S)"
        fi
        
        if [ $HIGH_ISSUES -gt 0 ]; then
            echo "  🟠 Corriger $HIGH_ISSUES problème(s) ÉLEVÉ(S)"
        fi
        
        if [ $SECURITY_SCORE -lt 8 ]; then
            echo "  🎯 Améliorer score sécurité: $SECURITY_SCORE/10 → ≥8/10"
        fi
        
        echo ""
        echo "🔧 Recommandations:"
        echo "  1. Relancer: ./scripts/setup-secrets.sh --env production"
        echo "  2. Vérifier: python scripts/security-audit.py --full"
        echo "  3. Re-valider: ./scripts/validate-security-phase.sh"
        
        exit 1
    fi
}

main "$@"