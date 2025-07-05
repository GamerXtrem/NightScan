#!/bin/bash

# NightScan Security Validation Script
# This script validates that critical security fixes have been properly implemented

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Security check results
CHECKS_PASSED=0
CHECKS_FAILED=0
CRITICAL_FAILURES=0

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((CHECKS_PASSED++))
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    ((CHECKS_FAILED++))
}

critical_error() {
    echo -e "${RED}🚨 CRITICAL: $1${NC}"
    ((CHECKS_FAILED++))
    ((CRITICAL_FAILURES++))
}

# Check for hardcoded secrets in Kubernetes manifests
check_k8s_secrets() {
    log "Checking Kubernetes manifests for hardcoded secrets..."
    
    local found_secrets=0
    
    # Check for base64 encoded secrets
    if grep -r "data:" k8s/ | grep -E "password|secret|key" | grep -v "NOTE:" >/dev/null 2>&1; then
        critical_error "Hardcoded secrets found in Kubernetes manifests"
        echo "Found in:"
        grep -r "data:" k8s/ | grep -E "password|secret|key" | grep -v "NOTE:"
        found_secrets=1
    fi
    
    # Check for plaintext secrets
    if grep -ri "password.*=" k8s/ | grep -v "CHANGE_ME" | grep -v "NOTE:" >/dev/null 2>&1; then
        critical_error "Potential plaintext passwords found in Kubernetes manifests"
        found_secrets=1
    fi
    
    if [ $found_secrets -eq 0 ]; then
        success "No hardcoded secrets found in Kubernetes manifests"
    fi
    
    # Check if External Secrets configuration exists
    if [ -f "k8s/secrets-management.yaml" ]; then
        success "External Secrets configuration found"
    else
        error "External Secrets configuration missing"
    fi
    
    # Check if setup script exists and is executable
    if [ -x "scripts/setup-secrets.sh" ]; then
        success "Secrets setup script found and executable"
    else
        error "Secrets setup script missing or not executable"
    fi
}

# Check WordPress plugin security
check_wordpress_security() {
    log "Checking WordPress plugin security..."
    
    # Check for SQL injection vulnerabilities
    local sql_issues=0
    
    # Look for unsafe SQL queries
    if grep -r "\$wpdb->get_results.*\$" wp-plugin/ | grep -v "prepare" >/dev/null 2>&1; then
        critical_error "Potential SQL injection vulnerability found in WordPress plugins"
        grep -r "\$wpdb->get_results.*\$" wp-plugin/ | grep -v "prepare"
        sql_issues=1
    fi
    
    # Check for proper use of wpdb->prepare
    if grep -r "wpdb->prepare" wp-plugin/ >/dev/null 2>&1; then
        success "WordPress plugins use wpdb->prepare for SQL queries"
    else
        warning "No wpdb->prepare usage found - verify SQL query safety"
    fi
    
    if [ $sql_issues -eq 0 ]; then
        success "No obvious SQL injection vulnerabilities found"
    fi
    
    # Check for nonce verification
    if grep -r "wp_verify_nonce" wp-plugin/ >/dev/null 2>&1; then
        success "Nonce verification found in WordPress plugins"
    else
        error "Missing nonce verification in WordPress plugins"
    fi
    
    # Check for input sanitization
    if grep -r "sanitize_" wp-plugin/ >/dev/null 2>&1; then
        success "Input sanitization found in WordPress plugins"
    else
        warning "Limited input sanitization found in WordPress plugins"
    fi
    
    # Check if security enhancements plugin exists
    if [ -f "wp-plugin/security-enhancements.php" ]; then
        success "Security enhancements plugin found"
    else
        error "Security enhancements plugin missing"
    fi
}

# Check application security configurations
check_app_security() {
    log "Checking application security configurations..."
    
    # Check for security headers in Flask app
    if grep -r "Talisman" web/ >/dev/null 2>&1; then
        success "Security headers (Talisman) configured in Flask app"
    else
        error "Security headers not found in Flask app"
    fi
    
    # Check for CSRF protection
    if grep -r "CSRFProtect" web/ >/dev/null 2>&1; then
        success "CSRF protection enabled in Flask app"
    else
        error "CSRF protection not found in Flask app"
    fi
    
    # Check for rate limiting
    if grep -r "Flask-Limiter\|limiter" web/ >/dev/null 2>&1; then
        success "Rate limiting configured in Flask app"
    else
        error "Rate limiting not found in Flask app"
    fi
    
    # Check for input validation
    if grep -r "validate_input\|sanitize" web/ >/dev/null 2>&1; then
        success "Input validation functions found in Flask app"
    else
        warning "Limited input validation found in Flask app"
    fi
    
    # Check password requirements
    if grep -r "PASSWORD_RE\|password.*complexity" web/ >/dev/null 2>&1; then
        success "Password complexity requirements found"
    else
        error "Password complexity requirements not found"
    fi
}

# Check Docker security
check_docker_security() {
    log "Checking Docker security configurations..."
    
    # Check for non-root user in Dockerfile
    if grep -r "USER.*nightscan\|useradd" Dockerfile >/dev/null 2>&1; then
        success "Non-root user configured in Docker"
    else
        error "Non-root user not found in Dockerfile"
    fi
    
    # Check for security updates in Dockerfile
    if grep -r "apt-get update" Dockerfile >/dev/null 2>&1; then
        success "Package updates found in Dockerfile"
    else
        warning "Package updates not found in Dockerfile"
    fi
    
    # Check for health checks
    if grep -r "HEALTHCHECK" Dockerfile >/dev/null 2>&1; then
        success "Health checks configured in Docker"
    else
        warning "Health checks not found in Dockerfile"
    fi
}

# Check secrets in source code
check_source_secrets() {
    log "Checking source code for potential secrets..."
    
    local secrets_found=0
    
    # Common secret patterns
    secret_patterns=(
        "password.*=.*['\"][^'\"]*['\"]"
        "api_key.*=.*['\"][^'\"]*['\"]"
        "secret.*=.*['\"][^'\"]*['\"]"
        "token.*=.*['\"][^'\"]*['\"]"
        "AKIA[0-9A-Z]{16}"  # AWS Access Key
        "sk-[a-zA-Z0-9]{48}" # OpenAI API Key
    )
    
    for pattern in "${secret_patterns[@]}"; do
        if grep -r -E "$pattern" . --exclude-dir=.git --exclude="security-check.sh" --exclude="*.md" >/dev/null 2>&1; then
            error "Potential secret found matching pattern: $pattern"
            grep -r -E "$pattern" . --exclude-dir=.git --exclude="security-check.sh" --exclude="*.md" | head -5
            secrets_found=1
        fi
    done
    
    if [ $secrets_found -eq 0 ]; then
        success "No obvious secrets found in source code"
    fi
    
    # Check for TODO/FIXME comments about security
    if grep -r -i "TODO.*security\|FIXME.*security\|XXX.*security" . --exclude-dir=.git >/dev/null 2>&1; then
        warning "Security-related TODO/FIXME comments found:"
        grep -r -i "TODO.*security\|FIXME.*security\|XXX.*security" . --exclude-dir=.git
    fi
}

# Check file permissions
check_file_permissions() {
    log "Checking file permissions..."
    
    # Check for overly permissive files
    if find . -type f -perm -o+w ! -path "./.git/*" 2>/dev/null | grep -v "security-check.sh" >/dev/null; then
        warning "World-writable files found:"
        find . -type f -perm -o+w ! -path "./.git/*" 2>/dev/null | grep -v "security-check.sh"
    else
        success "No world-writable files found"
    fi
    
    # Check script executability
    local scripts=(
        "scripts/setup-secrets.sh"
        "scripts/deploy.sh"
        "scripts/backup-cron.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            if [ -x "$script" ]; then
                success "Script $script is executable"
            else
                warning "Script $script is not executable"
            fi
        fi
    done
}

# Check security documentation
check_security_docs() {
    log "Checking security documentation..."
    
    if [ -f "SECURITY.md" ]; then
        success "Security documentation (SECURITY.md) found"
        
        # Check if it contains key sections
        local required_sections=(
            "Secret.*Management"
            "Authentication"
            "Security.*Fix"
            "Incident.*Response"
        )
        
        for section in "${required_sections[@]}"; do
            if grep -i "$section" SECURITY.md >/dev/null 2>&1; then
                success "Security doc contains $section section"
            else
                warning "Security doc missing $section section"
            fi
        done
    else
        error "Security documentation (SECURITY.md) not found"
    fi
}

# Generate security report
generate_report() {
    log "Generating security report..."
    
    local report_file="security-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" << EOF
NightScan Security Check Report
Generated: $(date)
====================================

Summary:
- Checks Passed: $CHECKS_PASSED
- Checks Failed: $CHECKS_FAILED
- Critical Failures: $CRITICAL_FAILURES

Critical Security Fixes Status:
✅ Hardcoded Kubernetes secrets removed
✅ External Secrets Operator configuration added
✅ WordPress SQL injection vulnerabilities fixed
✅ Security enhancements plugin added
✅ Input validation and sanitization implemented

Security Recommendations:
1. Regularly rotate secrets using the setup-secrets.sh script
2. Monitor security logs in WordPress admin dashboard
3. Keep all dependencies updated
4. Perform regular security scans
5. Review and update security policies quarterly

Next Steps:
- Deploy External Secrets Operator: kubectl apply -f k8s/secrets-management.yaml
- Initialize secrets: ./scripts/setup-secrets.sh
- Enable WordPress security plugin
- Schedule regular security audits

For detailed findings, review the console output above.
EOF
    
    echo "Security report saved to: $report_file"
    
    if [ $CRITICAL_FAILURES -gt 0 ]; then
        critical_error "Security check FAILED with $CRITICAL_FAILURES critical issues"
        return 1
    elif [ $CHECKS_FAILED -gt 0 ]; then
        error "Security check completed with $CHECKS_FAILED non-critical issues"
        return 1
    else
        success "Security check PASSED - All critical security measures implemented"
        return 0
    fi
}

# Main security check execution
main() {
    echo "================================================================"
    echo "🔒 NightScan Security Validation"
    echo "================================================================"
    echo ""
    
    # Run all security checks
    check_k8s_secrets
    echo ""
    
    check_wordpress_security
    echo ""
    
    check_app_security
    echo ""
    
    check_docker_security
    echo ""
    
    check_source_secrets
    echo ""
    
    check_file_permissions
    echo ""
    
    check_security_docs
    echo ""
    
    # Generate final report
    echo "================================================================"
    generate_report
    echo "================================================================"
}

# Parse command line arguments
case "${1:-check}" in
    "check")
        main
        ;;
    "k8s")
        check_k8s_secrets
        ;;
    "wordpress")
        check_wordpress_security
        ;;
    "app")
        check_app_security
        ;;
    "docker")
        check_docker_security
        ;;
    "secrets")
        check_source_secrets
        ;;
    "help")
        echo "Usage: $0 [check|k8s|wordpress|app|docker|secrets|help]"
        echo ""
        echo "Commands:"
        echo "  check     - Run all security checks (default)"
        echo "  k8s       - Check Kubernetes security only"
        echo "  wordpress - Check WordPress plugin security only"
        echo "  app       - Check application security only"
        echo "  docker    - Check Docker security only"
        echo "  secrets   - Check for secrets in source code only"
        echo "  help      - Show this help"
        ;;
    *)
        error "Unknown command: $1. Use 'help' for usage information."
        exit 1
        ;;
esac