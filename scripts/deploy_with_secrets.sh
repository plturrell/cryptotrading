#!/bin/bash

# 🚀 Comprehensive Vercel Deployment with Secret Management
# This script handles the complete deployment workflow including secret management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENVIRONMENT="production"

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_banner() {
    echo "🚀 VERCEL DEPLOYMENT WITH SECRET MANAGEMENT"
    echo "==========================================="
    echo "Project: $(basename "$PROJECT_ROOT")"
    echo "Environment: $DEPLOY_ENVIRONMENT"
    echo "Timestamp: $(date)"
    echo ""
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for required commands
    local missing_commands=()
    
    for cmd in python3 node npm vercel; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        log_error "Missing required commands: ${missing_commands[*]}"
        echo "Please install the missing commands and try again."
        exit 1
    fi
    
    # Check for Python dependencies
    if ! python3 -c "import cryptography" &> /dev/null; then
        log_warning "Cryptography library not found, installing..."
        pip3 install cryptography
    fi
    
    # Check Vercel authentication
    if ! vercel whoami &> /dev/null; then
        log_error "Not authenticated with Vercel. Run 'vercel login' first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

validate_secrets() {
    log_info "Validating secrets for deployment..."
    
    # Run pre-deployment check
    cd "$PROJECT_ROOT"
    
    if python3 -c "
from config.dev_workflow import DevWorkflowManager
import sys

try:
    dwm = DevWorkflowManager()
    results = dwm.pre_deployment_check('$DEPLOY_ENVIRONMENT')
    
    if results['passed']:
        print('✅ Secret validation passed')
        
        if results['warnings']:
            print('⚠️  Warnings:')
            for warning in results['warnings']:
                print(f'   - {warning}')
        
        sys.exit(0)
    else:
        print('❌ Secret validation failed')
        for error in results['errors']:
            print(f'   - {error}')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Validation error: {e}')
    sys.exit(1)
"; then
        log_success "Secret validation passed"
    else
        log_error "Secret validation failed"
        exit 1
    fi
}

deploy_secrets_to_vercel() {
    log_info "Deploying secrets to Vercel..."
    
    cd "$PROJECT_ROOT"
    
    # Deploy secrets using our secret manager
    python3 -c "
from config.vercel_secrets import VercelSecretManager
from config.secret_manager import Environment
import sys

try:
    vsm = VercelSecretManager()
    
    # Map deployment environment to local environment
    env_mapping = {
        'development': Environment.DEVELOPMENT,
        'staging': Environment.STAGING, 
        'production': Environment.PRODUCTION
    }
    
    local_env = env_mapping.get('$DEPLOY_ENVIRONMENT', Environment.PRODUCTION)
    
    # Deploy secrets
    results = vsm.deploy_secrets_to_vercel(local_env, '$VERCEL_TARGET_ENV')
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f'📊 Deployment Results: {success_count}/{total_count} secrets deployed')
    
    failed_secrets = [name for name, success in results.items() if not success]
    if failed_secrets:
        print(f'❌ Failed to deploy: {failed_secrets}')
        sys.exit(1)
    else:
        print('✅ All secrets deployed successfully')
        
except Exception as e:
    print(f'❌ Secret deployment failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Secrets deployed to Vercel"
    else
        log_error "Failed to deploy secrets"
        exit 1
    fi
}

build_application() {
    log_info "Building application..."
    
    cd "$PROJECT_ROOT"
    
    # Install dependencies
    if [ -f "package.json" ]; then
        npm install
    fi
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
    fi
    
    # Run build
    if [ -f "package.json" ] && grep -q "build" package.json; then
        npm run build
        log_success "Application built successfully"
    else
        log_info "No build script found, skipping build step"
    fi
}

deploy_to_vercel() {
    log_info "Deploying to Vercel..."
    
    cd "$PROJECT_ROOT"
    
    # Determine deployment command based on environment
    local deploy_cmd="vercel"
    
    if [ "$VERCEL_TARGET_ENV" = "production" ]; then
        deploy_cmd="vercel --prod"
    elif [ "$VERCEL_TARGET_ENV" = "preview" ]; then
        deploy_cmd="vercel"
    else
        deploy_cmd="vercel --target development"
    fi
    
    # Deploy
    if $deploy_cmd; then
        log_success "Deployed to Vercel successfully"
    else
        log_error "Vercel deployment failed"
        exit 1
    fi
}

post_deployment_verification() {
    log_info "Running post-deployment verification..."
    
    cd "$PROJECT_ROOT"
    
    # Generate deployment report
    python3 -c "
from config.vercel_secrets import VercelSecretManager
import sys

try:
    vsm = VercelSecretManager()
    report = vsm.generate_deployment_report()
    
    # Save report
    with open('deployment_report_$(date +%Y%m%d_%H%M%S).md', 'w') as f:
        f.write(report)
    
    print('📄 Deployment report generated')
    print(report)
    
except Exception as e:
    print(f'❌ Failed to generate report: {e}')
    sys.exit(1)
"
    
    log_success "Post-deployment verification completed"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove temporary files if any
    find "$PROJECT_ROOT" -name "*.tmp" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name ".env.temp*" -delete 2>/dev/null || true
    
    log_success "Cleanup completed"
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment    Deployment environment (development|staging|production)"
    echo "  -t, --target         Vercel target environment (development|preview|production)"
    echo "  -s, --secrets-only   Deploy secrets only, skip application deployment"
    echo "  -v, --validate-only  Validate secrets only, skip deployment"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Deploy to production"
    echo "  $0 -e staging -t preview             # Deploy staging to Vercel preview"
    echo "  $0 --secrets-only                    # Deploy secrets only"
    echo "  $0 --validate-only                   # Validate secrets only"
}

# Main script
main() {
    # Default values
    DEPLOY_ENVIRONMENT="$DEFAULT_ENVIRONMENT"
    VERCEL_TARGET_ENV="production"
    SECRETS_ONLY=false
    VALIDATE_ONLY=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                DEPLOY_ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--target)
                VERCEL_TARGET_ENV="$2"
                shift 2
                ;;
            -s|--secrets-only)
                SECRETS_ONLY=true
                shift
                ;;
            -v|--validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate arguments
    if [[ ! "$DEPLOY_ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        log_error "Invalid environment: $DEPLOY_ENVIRONMENT"
        exit 1
    fi
    
    if [[ ! "$VERCEL_TARGET_ENV" =~ ^(development|preview|production)$ ]]; then
        log_error "Invalid Vercel target: $VERCEL_TARGET_ENV"
        exit 1
    fi
    
    # Start deployment process
    print_banner
    
    # Step 1: Prerequisites
    check_prerequisites
    
    # Step 2: Validate secrets
    validate_secrets
    
    if [ "$VALIDATE_ONLY" = true ]; then
        log_success "Validation completed successfully"
        exit 0
    fi
    
    # Step 3: Deploy secrets
    deploy_secrets_to_vercel
    
    if [ "$SECRETS_ONLY" = true ]; then
        log_success "Secrets deployment completed successfully"
        exit 0
    fi
    
    # Step 4: Build application
    build_application
    
    # Step 5: Deploy to Vercel
    deploy_to_vercel
    
    # Step 6: Post-deployment verification
    post_deployment_verification
    
    # Step 7: Cleanup
    cleanup
    
    echo ""
    log_success "🎉 Deployment completed successfully!"
    echo "🔗 Check your deployment at: https://vercel.com/dashboard"
}

# Trap for cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"
