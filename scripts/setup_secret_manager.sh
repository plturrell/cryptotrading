#!/bin/bash
# 🔐 Secret Manager Quick Setup Script

echo "🔐 Crypto Trading Secret Manager Setup"
echo "====================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

# Install dependencies
echo "📦 Installing secret manager dependencies..."
pip install cryptography click

# Check if .env exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "📝 Creating .env from .env.example..."
        cp .env.example .env
        echo "⚠️  Please edit .env with your actual API keys"
    else
        echo "❌ No .env.example found"
        exit 1
    fi
fi

# Initialize secret manager
echo "🔧 Initializing secret manager..."
python scripts/secret_manager_cli.py status

# Sync from .env if it has real values
echo "🔄 Syncing secrets from .env..."
python scripts/secret_manager_cli.py sync

# Validate setup
echo "✅ Validating setup..."
python scripts/secret_manager_cli.py validate

# Show status
echo "📊 Secret Manager Status:"
python scripts/secret_manager_cli.py list

echo ""
echo "🎉 Secret Manager Setup Complete!"
echo ""
echo "🔧 Next Steps:"
echo "1. Edit .env with your actual API keys"
echo "2. Run: python scripts/secret_manager_cli.py sync"
echo "3. Deploy: python deployment/vercel_secrets_setup.py deploy"
echo ""
echo "📚 Full documentation: docs/SECRET_MANAGEMENT.md"
