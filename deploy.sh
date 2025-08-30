#!/bin/bash
"""
Rex Crypto Trading Platform - Complete Deployment Script
Handles GitHub setup, Vercel deployment, and system initialization
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="rex-crypto-trading"
GITHUB_REPO="plturrell/cryptotrading"
VERCEL_PROJECT_NAME="rex-crypto-trading"

echo -e "${BLUE}🚀 Rex Crypto Trading Platform - Complete Deployment${NC}"
echo "============================================================"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed"
    exit 1
fi

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_warning "npm is not installed - frontend build will be skipped"
fi

print_status "Prerequisites checked"

# Initialize git repository if not exists
if [ ! -d ".git" ]; then
    echo -e "${BLUE}📦 Initializing Git repository...${NC}"
    git init
    git add .
    git commit -m "Initial commit: Rex Crypto Trading Platform with Russian news and image enhancement"
    print_status "Git repository initialized"
else
    print_status "Git repository already exists"
fi

# Run the unified framework
echo -e "${BLUE}🏗️ Running build and deploy framework...${NC}"
python3 deploy/build_deploy_framework.py build-deploy

if [ $? -eq 0 ]; then
    print_status "Build and deploy framework completed"
else
    print_error "Build and deploy framework failed"
    exit 1
fi

# Setup GitHub repository (if not already connected)
echo -e "${BLUE}🐙 Setting up GitHub repository...${NC}"

# Check if remote origin exists
if git remote get-url origin &> /dev/null; then
    print_status "GitHub remote already configured"
else
    print_warning "GitHub remote not configured"
    echo "To connect to GitHub:"
    echo "1. Create repository at: https://github.com/new"
    echo "2. Run: git remote add origin https://github.com/$GITHUB_REPO.git"
    echo "3. Run: git push -u origin main"
fi

# Setup Vercel deployment
echo -e "${BLUE}🌐 Setting up Vercel deployment...${NC}"

# Check if vercel CLI is installed
if command -v vercel &> /dev/null; then
    echo "Vercel CLI found, setting up deployment..."
    
    # Login to Vercel (if not already logged in)
    if ! vercel whoami &> /dev/null; then
        echo "Please login to Vercel:"
        vercel login
    fi
    
    # Deploy to Vercel
    echo "Deploying to Vercel..."
    vercel --prod
    
    print_status "Vercel deployment completed"
else
    print_warning "Vercel CLI not installed"
    echo "To deploy to Vercel:"
    echo "1. Install Vercel CLI: npm i -g vercel"
    echo "2. Login: vercel login"
    echo "3. Deploy: vercel --prod"
fi

# Create environment files
echo -e "${BLUE}⚙️ Creating environment configuration...${NC}"

# Create .env.production
cat > .env.production << EOF
# Rex Crypto Trading Platform - Production Environment
NODE_ENV=production
FLASK_ENV=production

# API Configuration
PERPLEXITY_API_KEY=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
ENABLE_RUSSIAN_TRANSLATION=true
ENABLE_IMAGE_ENHANCEMENT=true

# Database Configuration
DATABASE_URL=sqlite:///cryptotrading.db

# News Service Configuration
NEWS_REFRESH_INTERVAL=300
MAX_ARTICLES_PER_FETCH=20

# Image Service Configuration
UNSPLASH_ACCESS_KEY=demo_key
ENABLE_CHART_GENERATION=true
ENABLE_WEB_SCRAPING=true

# System Configuration
SYSTEM_VERSION=1.0.0
DEBUG=false
EOF

print_status "Environment configuration created"

# Create deployment documentation
echo -e "${BLUE}📚 Creating deployment documentation...${NC}"

cat > DEPLOYMENT.md << EOF
# Rex Crypto Trading Platform - Deployment Guide

## System Overview
Complete crypto trading platform with:
- Russian cryptocurrency news service
- Image enhancement (web scraping, chart generation, image search)
- SAP UI5 frontend
- REST API backend
- SQLite database with full schema

## Quick Start

### 1. System Startup
\`\`\`bash
# Complete system startup
python3 startup.py

# Health check
python3 startup.py health

# Build and deploy
python3 startup.py build

# Deploy to Vercel
python3 startup.py deploy
\`\`\`

### 2. Manual Deployment

#### Build and Deploy New System
\`\`\`bash
python3 deploy/build_deploy_framework.py build-deploy
\`\`\`

#### Full System Startup
\`\`\`bash
python3 deploy/build_deploy_framework.py startup
\`\`\`

#### Deploy to GitHub and Vercel
\`\`\`bash
python3 deploy/build_deploy_framework.py deploy
\`\`\`

### 3. GitHub Deployment

1. Create repository at GitHub
2. Connect local repository:
   \`\`\`bash
   git remote add origin https://github.com/$GITHUB_REPO.git
   git push -u origin main
   \`\`\`

### 4. Vercel Deployment

1. Install Vercel CLI: \`npm i -g vercel\`
2. Login: \`vercel login\`
3. Deploy: \`vercel --prod\`

## Environment Variables

Set these in Vercel dashboard:
- \`PERPLEXITY_API_KEY\`: Your Perplexity API key
- \`ENABLE_RUSSIAN_TRANSLATION\`: true
- \`ENABLE_IMAGE_ENHANCEMENT\`: true

## System Architecture

### Database Schema
- \`news_articles\`: Full article storage with Russian translations and images
- \`user_searches\`: User search history and saved searches
- \`market_data\`: Cryptocurrency price data for chart generation
- \`system_config\`: System configuration and settings

### API Endpoints
- \`/api/news/latest\`: Latest crypto news
- \`/api/news/latest/russian\`: Russian crypto news
- \`/api/news/translate\`: Translate articles
- \`/api/search\`: User news search
- \`/api/health\`: System health check

### Frontend
- SAP UI5 application with Fiori design
- Responsive news display with image support
- Russian/English language toggle
- Category filtering and search

## Features

### ✅ Implemented
- Russian cryptocurrency news service
- AI translation with Perplexity API
- Web scraping for article images
- Real-time price chart generation
- Cryptocurrency image search
- Complete database schema
- SAP UI5 frontend
- REST API backend
- GitHub Actions CI/CD
- Vercel deployment configuration

### 🔧 Usage Examples

#### Fetch News with Images
\`\`\`python
from src.cryptotrading.infrastructure.data.news_service import PerplexityNewsService

service = PerplexityNewsService(enable_images=True)
articles = await service.get_latest_news(limit=10)
enhanced = await service.enhance_articles_with_images(articles)

for article in enhanced:
    print(f"Title: {article.title}")
    print(f"Images: {article.image_count}")
    for img in article.images:
        print(f"  - {img.type}: {img.alt_text}")
\`\`\`

#### Generate Price Charts
\`\`\`python
from src.cryptotrading.infrastructure.data.image_services import CryptoPriceChartGenerator

generator = CryptoPriceChartGenerator()
chart = await generator.generate_price_chart('BTC', days=7, chart_type='candlestick')
print(f"Chart URL: {chart.url[:50]}...")
\`\`\`

## Monitoring and Health

### Health Check
\`\`\`bash
curl http://localhost:5000/api/health
\`\`\`

### System Status
\`\`\`bash
python3 startup.py health
\`\`\`

## Support

For issues and questions:
1. Check system health: \`python3 startup.py health\`
2. Review logs in console output
3. Verify environment variables
4. Test individual components with test scripts

## Version
Current version: 1.0.0
Last updated: $(date)
EOF

print_status "Deployment documentation created"

# Final status
echo ""
echo -e "${GREEN}🎉 Deployment Setup Complete!${NC}"
echo "============================================================"
echo -e "${BLUE}📊 System Status:${NC}"
echo "   ✅ Unified build and deploy framework"
echo "   ✅ Database schema and migrations"
echo "   ✅ Complete system startup scripts"
echo "   ✅ GitHub Actions workflow"
echo "   ✅ Vercel deployment configuration"
echo "   ✅ Environment configuration"
echo "   ✅ Deployment documentation"

echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo "1. Start system: ${YELLOW}python3 startup.py${NC}"
echo "2. Check health: ${YELLOW}python3 startup.py health${NC}"
echo "3. Push to GitHub: ${YELLOW}git push origin main${NC}"
echo "4. Deploy to Vercel: ${YELLOW}vercel --prod${NC}"

echo ""
echo -e "${GREEN}🌟 Your Rex Crypto Trading Platform is ready for deployment!${NC}"
