#!/bin/bash
# SAP UI5 Build & Deploy Script for рекс.com Crypto Trading Platform

set -e

echo "🚀 Starting SAP UI5 Build & Deploy Process..."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
npm run clean

# Run linting
echo "🔍 Running ESLint..."
npm run lint

# Build for production
echo "📦 Building for production..."
ui5 build --dest dist --include-task=generateVersionInfo --clean-dest

# Generate build info
echo "📋 Generating build information..."
cat > dist/build-info.json << EOF
{
  "buildTime": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "version": "$(node -p "require('./package.json').version")",
  "commit": "$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
  "branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "environment": "production"
}
EOF

# Create deployment package
echo "📦 Creating deployment package..."
cd dist
zip -r ../launchpad-$(date +%Y%m%d-%H%M%S).zip . -x "*.map" "test/*" "localService/*"
cd ..

# Optional: Deploy to server (uncomment and configure as needed)
# echo "🚀 Deploying to server..."
# scp launchpad-*.zip user@server:/path/to/deployment/
# ssh user@server "cd /path/to/deployment && unzip -o launchpad-*.zip"

echo "✅ Build & Deploy completed successfully!"
echo "📁 Build output: ./dist/"
echo "📦 Deployment package: ./launchpad-*.zip"

# Optional: Start local preview
read -p "🌐 Start local preview? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🌐 Starting local preview server..."
    ui5 serve --config ui5.yaml --port 8080 --h2
fi
