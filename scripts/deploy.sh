#!/bin/bash

# MCP Server Deployment Script for Vercel
# Handles both local development and production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
ENV_FILE="$PROJECT_ROOT/.env"
VERCEL_JSON="$PROJECT_ROOT/vercel.json"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [ ! -f "$VERCEL_JSON" ]; then
        error "vercel.json not found. Run this script from the project root."
    fi
    
    # Check if Vercel CLI is installed
    if ! command -v vercel &> /dev/null; then
        warning "Vercel CLI not found. Installing..."
        npm install -g vercel || error "Failed to install Vercel CLI"
    fi
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not found"
    fi
    
    success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    local env_type=${1:-"development"}
    
    log "Setting up $env_type environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        log "Creating .env file..."
        cat > "$ENV_FILE" << EOF
# MCP Server Environment Configuration
MCP_ENVIRONMENT=$env_type
MCP_JWT_SECRET=$(openssl rand -base64 32)
MCP_REQUIRE_AUTH=true
MCP_RATE_LIMIT_GLOBAL=1000
MCP_STRICT_VALIDATION=true
MCP_AUDIT_LOGGING=true

# Optional API Keys (uncomment and add your keys)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Database (optional)
# REDIS_URL=redis://localhost:6379
# DATABASE_URL=postgresql://...
EOF
        success "Created .env file with default configuration"
        warning "Please edit .env file and add your API keys"
    else
        log ".env file already exists"
    fi
    
    # Set environment-specific variables
    if [ "$env_type" = "development" ]; then
        sed -i.bak 's/MCP_REQUIRE_AUTH=true/MCP_REQUIRE_AUTH=false/' "$ENV_FILE"
        sed -i.bak 's/MCP_STRICT_VALIDATION=true/MCP_STRICT_VALIDATION=false/' "$ENV_FILE"
        rm -f "$ENV_FILE.bak"
        log "Configured for development (auth disabled, validation relaxed)"
    fi
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Install Python dependencies
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt || error "Failed to install Python dependencies"
    else
        warning "requirements.txt not found, creating basic one..."
        cat > requirements.txt << EOF
aiohttp>=3.8.0
websockets>=11.0.0
cryptography>=3.4.8
python-jose[cryptography]>=3.3.0
redis>=4.3.0
watchdog>=2.1.9
EOF
        pip3 install -r requirements.txt
    fi
    
    success "Dependencies installed"
}

# Run local development server
dev_server() {
    log "Starting local development server..."
    
    cd "$PROJECT_ROOT"
    setup_environment "development"
    
    # Check if port is available
    local port=${1:-8080}
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        warning "Port $port is already in use"
        read -p "Kill existing process and continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            lsof -ti:$port | xargs kill -9
        else
            error "Cannot start server on port $port"
        fi
    fi
    
    # Start development server
    log "Starting MCP development server on port $port..."
    python3 -m scripts.dev start --port $port --no-auth
}

# Test local server
test_local() {
    local url=${1:-"http://localhost:8080"}
    
    log "Testing local server at $url..."
    
    # Wait for server to be ready
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url/api/mcp" > /dev/null 2>&1; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    
    if [ $attempt -eq $max_attempts ]; then
        error "Server did not start within 30 seconds"
    fi
    
    # Run tests
    log "Running API tests..."
    
    # Test status endpoint
    local status_response=$(curl -s "$url/api/mcp")
    if echo "$status_response" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
        success "Status endpoint working"
    else
        error "Status endpoint failed"
    fi
    
    # Test tools list
    local tools_response=$(curl -s -X POST "$url/api/mcp" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"tools/list","id":"test"}')
    
    if echo "$tools_response" | jq -e '.result.tools' > /dev/null 2>&1; then
        local tool_count=$(echo "$tools_response" | jq '.result.tools | length')
        success "Tools endpoint working ($tool_count tools available)"
    else
        error "Tools endpoint failed"
    fi
    
    success "Local server tests passed"
}

# Deploy to Vercel
deploy_vercel() {
    local env_type=${1:-"preview"}
    
    log "Deploying to Vercel ($env_type)..."
    
    cd "$PROJECT_ROOT"
    
    # Ensure we're logged in to Vercel
    if ! vercel whoami > /dev/null 2>&1; then
        log "Logging in to Vercel..."
        vercel login || error "Failed to login to Vercel"
    fi
    
    # Set environment variables
    log "Setting up Vercel environment variables..."
    
    # Read .env file and set variables
    if [ -f "$ENV_FILE" ]; then
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            if [[ $key =~ ^[[:space:]]*# ]] || [[ -z "$key" ]]; then
                continue
            fi
            
            # Remove any quotes and whitespace
            key=$(echo "$key" | tr -d ' ')
            value=$(echo "$value" | sed 's/^"//' | sed 's/"$//')
            
            if [ ! -z "$value" ]; then
                log "Setting environment variable: $key"
                if [ "$env_type" = "production" ]; then
                    vercel env add "$key" production <<< "$value" > /dev/null 2>&1 || true
                else
                    vercel env add "$key" preview <<< "$value" > /dev/null 2>&1 || true
                fi
            fi
        done < "$ENV_FILE"
    fi
    
    # Deploy
    log "Starting deployment..."
    
    if [ "$env_type" = "production" ]; then
        vercel --prod || error "Production deployment failed"
    else
        vercel || error "Preview deployment failed"
    fi
    
    success "Deployment completed successfully!"
    
    # Get deployment URL
    local deployment_url=$(vercel ls 2>/dev/null | head -n 2 | tail -n 1 | awk '{print $2}')
    if [ ! -z "$deployment_url" ]; then
        success "Deployment URL: https://$deployment_url"
        log "Testing deployment..."
        
        # Wait a moment for deployment to be ready
        sleep 5
        
        # Test deployment
        if curl -s "https://$deployment_url/api/mcp" | jq -e '.status' > /dev/null 2>&1; then
            success "Deployment is healthy and responding"
        else
            warning "Deployment may still be initializing"
        fi
    fi
}

# Generate deployment info
generate_info() {
    log "Generating deployment information..."
    
    cat << EOF

🚀 MCP Server Deployment Information
=====================================

Local Development:
  • Start: ./scripts/deploy.sh dev
  • URL: http://localhost:8080
  • Docs: http://localhost:8080 (web interface)
  • API: http://localhost:8080/api/mcp

Vercel Deployment:
  • Preview: ./scripts/deploy.sh deploy
  • Production: ./scripts/deploy.sh deploy production
  • Environment: Edit .env file

Testing:
  • Local: ./scripts/deploy.sh test
  • API: curl -X POST [URL]/api/mcp -d '{"jsonrpc":"2.0","method":"tools/list","id":"1"}'

Configuration:
  • Main config: vercel.json
  • Environment: .env
  • API handler: api/mcp.py
  • Web interface: api/index.py

Quick Start:
  1. ./scripts/deploy.sh setup
  2. ./scripts/deploy.sh dev
  3. Open http://localhost:8080
  4. Test tools in web interface
  5. ./scripts/deploy.sh deploy (when ready)

EOF
}

# Main command handling
case "${1:-help}" in
    "setup"|"init")
        log "Setting up MCP Server deployment..."
        check_prerequisites
        setup_environment "development"
        install_dependencies
        success "Setup complete! Run './scripts/deploy.sh dev' to start development server"
        ;;
    
    "dev"|"development")
        check_prerequisites
        dev_server "${2:-8080}"
        ;;
    
    "test")
        test_local "${2:-http://localhost:8080}"
        ;;
    
    "deploy")
        env_type="${2:-preview}"
        if [ "$env_type" != "preview" ] && [ "$env_type" != "production" ]; then
            error "Deploy environment must be 'preview' or 'production'"
        fi
        check_prerequisites
        setup_environment "production"
        deploy_vercel "$env_type"
        ;;
    
    "info")
        generate_info
        ;;
    
    "help"|*)
        cat << EOF
MCP Server Deployment Script

Usage: $0 <command> [options]

Commands:
  setup          Initialize project and install dependencies
  dev [port]     Start local development server (default port: 8080)
  test [url]     Test local or remote server
  deploy [env]   Deploy to Vercel (env: preview|production, default: preview)
  info           Show deployment information
  help           Show this help message

Examples:
  $0 setup                    # Initial setup
  $0 dev                      # Start dev server on port 8080
  $0 dev 3000                 # Start dev server on port 3000
  $0 test                     # Test local server
  $0 deploy                   # Deploy preview to Vercel
  $0 deploy production        # Deploy to production

Quick Start:
  $0 setup && $0 dev

EOF
        ;;
esac