# 🔐 Secret Management Guide

Complete guide for managing API keys and secrets in local development and deployment to containers and Vercel.

## 🎯 **Quick Start**

### 1. **Install Dependencies**
```bash
pip install cryptography click
```

### 2. **Setup Secret Manager**
```bash
# Sync existing secrets from .env
python scripts/secret_manager_cli.py sync

# Check status
python scripts/secret_manager_cli.py status
```

### 3. **Store New Secrets**
```bash
# Store API keys
python scripts/secret_manager_cli.py store GROK4_API_KEY "your-actual-key" --category ai
python scripts/secret_manager_cli.py store BINANCE_API_KEY "your-api-key" --category trading
```

---

## 🔐 **Local Development**

### **Secret Manager CLI**

#### **Store Secrets**
```bash
# Store with automatic categorization
python scripts/secret_manager_cli.py store GROK4_API_KEY "xai-abc123"

# Store with specific category
python scripts/secret_manager_cli.py store DATABASE_URL "postgresql://..." --category database
```

#### **Retrieve Secrets**
```bash
# Get secret (masked)
python scripts/secret_manager_cli.py get GROK4_API_KEY

# Show actual value
python scripts/secret_manager_cli.py get GROK4_API_KEY --show

# Search in specific category
python scripts/secret_manager_cli.py get DATABASE_URL --category database
```

#### **List Secrets**
```bash
# List all secrets
python scripts/secret_manager_cli.py list

# List specific category
python scripts/secret_manager_cli.py list --category ai
```

#### **Generate .env Files**
```bash
# Generate .env for development
python scripts/secret_manager_cli.py generate-env

# Generate for production
python scripts/secret_manager_cli.py generate-env --environment production --output .env.production
```

#### **Validate Secrets**
```bash
# Validate all secrets
python scripts/secret_manager_cli.py validate

# Validate specific required keys
python scripts/secret_manager_cli.py validate --required GROK4_API_KEY --required DATABASE_URL
```

### **Python API Usage**
```python
from config.secret_manager import SecretManager

# Initialize
sm = SecretManager()

# Store secrets
sm.store_secret('API_KEY', 'secret-value', 'ai')

# Retrieve secrets
api_key = sm.get_secret('API_KEY')

# Generate .env file
sm.generate_env_file('.env.example', '.env', 'development')
```

---

## 🐳 **Container Deployment**

### **Docker with Secrets**

#### **Method 1: Environment File Mount**
```bash
# Generate .env for container
python scripts/secret_manager_cli.py generate-env --environment production --output .env.container

# Run with mounted env file
docker run --env-file .env.container your-app
```

#### **Method 2: Docker Secrets (Recommended)**
```bash
# Export for Docker secrets
python scripts/secret_manager_cli.py export-container --format docker

# Create Docker secret
echo "GROK4_API_KEY=your-key" | docker secret create app-secrets -

# Run with secrets
docker service create --secret app-secrets your-app
```

#### **Method 3: Docker Compose**
```yaml
# docker-compose.secrets.yml
services:
  app:
    secrets:
      - app-secrets
    environment:
      - DATABASE_URL_FILE=/run/secrets/app-secrets

secrets:
  app-secrets:
    file: ./secrets/production.env
```

### **Kubernetes Deployment**
```bash
# Export Kubernetes manifest
python scripts/secret_manager_cli.py export-container --format k8s --output k8s-secrets.yaml

# Apply to cluster
kubectl apply -f k8s-secrets.yaml
```

---

## ▲ **Vercel Deployment**

### **Quick Deploy**
```bash
# Setup and deploy in one command
python deployment/vercel_secrets_setup.py deploy --environment production
```

### **Manual Setup**

#### **1. Setup Environment Variables**
```bash
# Setup all secrets
python deployment/vercel_secrets_setup.py setup --environment production

# Setup specific categories
python deployment/vercel_secrets_setup.py setup --environment production --categories ai trading
```

#### **2. Generate Setup Script**
```bash
# Generate automated setup script
python deployment/vercel_secrets_setup.py script --output deploy_vercel.sh

# Run the script
./deploy_vercel.sh
```

#### **3. Manual CLI Commands**
```bash
# Export Vercel commands
python scripts/secret_manager_cli.py export-vercel

# Run each command manually
vercel env add GROK4_API_KEY production <<< "your-key"
vercel env add DATABASE_URL production <<< "postgresql://..."
```

#### **4. Validate Deployment**
```bash
# Validate environment variables
python deployment/vercel_secrets_setup.py validate --environment production
```

### **Vercel Configuration**

#### **vercel.json Configuration**
```json
{
  "env": {
    "GROK4_API_KEY": "@grok4_api_key",
    "DATABASE_URL": "@database_url",
    "REDIS_URL": "@redis_url"
  },
  "build": {
    "env": {
      "GROK4_API_KEY": "@grok4_api_key",
      "DATABASE_URL": "@database_url"
    }
  }
}
```

---

## 🔒 **Security Best Practices**

### **File Permissions**
```bash
# Secure secret files
chmod 600 .env
chmod 600 config/.encryption_key
chmod 600 config/secrets.encrypted

# Secure scripts
chmod 755 scripts/secret_manager_cli.py
chmod 755 deployment/vercel_secrets_setup.py
```

### **Gitignore Configuration**
```gitignore
# 🔐 Secret files
.env
.env.local
.env.production
config/.encryption_key
config/secrets.encrypted
secrets/
vercel_setup.sh
deploy_*.sh

# 🐳 Container secrets
docker/secrets/
k8s-secrets.yaml
```

### **Environment Separation**
```bash
# Development
python scripts/secret_manager_cli.py generate-env --environment development

# Production  
python scripts/secret_manager_cli.py generate-env --environment production --output .env.production
```

---

## 🛠️ **Advanced Features**

### **Key Rotation**
```bash
# Rotate encryption key
python scripts/secret_manager_cli.py rotate-key

# With new master key
python scripts/secret_manager_cli.py rotate-key --new-key "new-master-password"
```

### **Category Management**
```bash
# Store in specific categories
python scripts/secret_manager_cli.py store GROK4_API_KEY "key" --category ai
python scripts/secret_manager_cli.py store BINANCE_API_KEY "key" --category trading

# Export specific categories
python scripts/secret_manager_cli.py export-vercel --categories ai trading
python scripts/secret_manager_cli.py export-container --categories database security
```

### **Backup & Restore**
```bash
# Backup encrypted secrets
cp config/secrets.encrypted config/secrets.backup

# Backup encryption key (store securely!)
cp config/.encryption_key ~/.secrets/cryptotrading.key
```

---

## 🚀 **Deployment Workflows**

### **Complete Local → Vercel Workflow**
```bash
# 1. Store all secrets locally
python scripts/secret_manager_cli.py sync  # From .env

# 2. Validate locally
python scripts/secret_manager_cli.py validate --required GROK4_API_KEY

# 3. Deploy to Vercel
python deployment/vercel_secrets_setup.py deploy --environment production

# 4. Validate deployment
python deployment/vercel_secrets_setup.py validate --environment production
```

### **Complete Local → Container Workflow**
```bash
# 1. Export for containers
python scripts/secret_manager_cli.py export-container --format docker

# 2. Build with secrets
docker build -f docker/secrets.dockerfile -t cryptotrading .

# 3. Run with secrets
docker run --env-file .env.production cryptotrading
```

---

## 🧪 **Testing & Validation**

### **Test Secret Manager**
```bash
# Test basic functionality
python config/secret_manager.py

# Validate specific deployment
python scripts/secret_manager_cli.py validate --required GROK4_API_KEY --required DATABASE_URL
```

### **Test Deployments**
```bash
# Test Vercel environment
python deployment/vercel_secrets_setup.py validate --environment production

# Test container secrets
docker run --env-file .env.container cryptotrading python -c "import os; print('✅' if os.getenv('GROK4_API_KEY') else '❌')"
```

---

## 🆘 **Troubleshooting**

### **Common Issues**

#### **"Encryption key not found"**
```bash
# Regenerate key
rm config/.encryption_key
python scripts/secret_manager_cli.py status
```

#### **"Vercel CLI authentication failed"**
```bash
# Re-authenticate
vercel logout
vercel login
```

#### **"Secrets not found in Vercel"**
```bash
# Check current Vercel environment
vercel env ls production

# Re-setup environment
python deployment/vercel_secrets_setup.py setup --environment production
```

#### **"Container secrets not loading"**
```bash
# Check container environment
docker run your-app env | grep API_KEY

# Debug secret initialization
docker run your-app cat /app/init-secrets.sh
```

---

## 📊 **Secret Categories**

The system automatically categorizes secrets:

- **🤖 ai**: AI service keys (Grok4, OpenAI, Perplexity)
- **💹 trading**: Exchange APIs (Binance, Coinbase)
- **🗄️ database**: Database connections (PostgreSQL, Redis)
- **🔒 security**: Authentication keys (JWT, encryption)
- **📊 monitoring**: Observability services (Sentry, OTEL)
- **🔧 general**: Uncategorized secrets

---

## 🎯 **Next Steps**

1. **✅ Setup**: Run the quick start commands
2. **🔐 Store**: Add your API keys using the CLI
3. **🚀 Deploy**: Use the deployment workflows
4. **✅ Validate**: Check that everything works
5. **🔄 Maintain**: Rotate keys regularly

**Your secrets are now secure and deployment-ready!** 🎉
