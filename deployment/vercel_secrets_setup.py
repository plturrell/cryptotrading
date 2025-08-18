#!/usr/bin/env python3
"""
▲ Vercel Secret Management Integration
Automated setup and management of environment variables for Vercel deployment
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.secret_manager import SecretManager

class VercelSecretsManager:
    """
    ▲ Vercel-specific secret management
    
    Handles:
    - Environment variable setup via CLI
    - vercel.json configuration
    - Deployment validation
    - Environment-specific configurations
    """
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.secret_manager = SecretManager()
        self.vercel_json_path = self.project_path / "vercel.json"
    
    def setup_vercel_env(self, 
                        environment: str = "production",
                        categories: List[str] = None) -> bool:
        """
        🚀 Setup Vercel environment variables
        
        Args:
            environment: Vercel environment (development, preview, production)
            categories: Secret categories to include
        """
        try:
            print(f"🚀 Setting up Vercel environment: {environment}")
            
            # Check Vercel CLI
            if not self._check_vercel_cli():
                return False
            
            # Export secrets
            export_data = self.secret_manager.export_for_vercel(categories)
            if not export_data:
                print("❌ No secrets to export")
                return False
            
            success_count = 0
            total_secrets = len(export_data['secrets'])
            
            print(f"📊 Setting {total_secrets} environment variables...")
            
            # Set each environment variable
            for key, value in export_data['secrets'].items():
                if self._set_vercel_env_var(key, value, environment):
                    success_count += 1
                    print(f"✅ {key}")
                else:
                    print(f"❌ {key}")
            
            print(f"\n🎯 Result: {success_count}/{total_secrets} secrets configured")
            
            # Update vercel.json if needed
            self._update_vercel_json(list(export_data['secrets'].keys()))
            
            return success_count == total_secrets
            
        except Exception as e:
            print(f"❌ Failed to setup Vercel environment: {e}")
            return False
    
    def deploy_with_secrets(self, 
                           environment: str = "production",
                           categories: List[str] = None) -> bool:
        """
        🚀 Deploy to Vercel with secrets validation
        """
        try:
            print(f"🚀 Deploying to Vercel ({environment})...")
            
            # Setup environment variables first
            if not self.setup_vercel_env(environment, categories):
                print("❌ Failed to setup environment variables")
                return False
            
            # Deploy
            deploy_cmd = ["vercel", "--prod" if environment == "production" else ""]
            result = subprocess.run(deploy_cmd, capture_output=True, text=True, cwd=self.project_path)
            
            if result.returncode == 0:
                print("✅ Deployment successful!")
                print(f"🌐 URL: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ Deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Deployment error: {e}")
            return False
    
    def validate_vercel_env(self, environment: str = "production") -> Dict[str, any]:
        """
        ✅ Validate Vercel environment variables
        """
        try:
            print(f"🔍 Validating Vercel environment: {environment}")
            
            # Get current Vercel environment variables
            result = subprocess.run(
                ["vercel", "env", "ls", environment],
                capture_output=True, text=True, cwd=self.project_path
            )
            
            if result.returncode != 0:
                return {"valid": False, "error": f"Failed to list Vercel env vars: {result.stderr}"}
            
            # Parse Vercel env vars
            vercel_vars = set()
            for line in result.stdout.split('\n'):
                if '│' in line and not line.strip().startswith('│ Name'):
                    parts = line.split('│')
                    if len(parts) > 1:
                        var_name = parts[1].strip()
                        if var_name and var_name != "Name":
                            vercel_vars.add(var_name)
            
            # Compare with local secrets
            local_secrets = self.secret_manager.list_secrets()
            local_vars = set()
            for category, keys in local_secrets.items():
                local_vars.update(keys)
            
            # Analysis
            missing_in_vercel = local_vars - vercel_vars
            extra_in_vercel = vercel_vars - local_vars
            
            validation_result = {
                "valid": len(missing_in_vercel) == 0,
                "local_secrets": len(local_vars),
                "vercel_secrets": len(vercel_vars),
                "missing_in_vercel": list(missing_in_vercel),
                "extra_in_vercel": list(extra_in_vercel),
                "synchronized": list(local_vars & vercel_vars)
            }
            
            return validation_result
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def generate_deployment_script(self, 
                                  output_file: str = "deploy_to_vercel.sh") -> bool:
        """
        📜 Generate complete deployment script
        """
        try:
            script_content = f"""#!/bin/bash
# 🚀 Complete Vercel Deployment Script with Secrets
# Generated by VercelSecretsManager

set -e  # Exit on any error

echo "🔐 Crypto Trading System - Vercel Deployment"
echo "=============================================="

# 🔍 Pre-deployment checks
echo "🔍 Pre-deployment validation..."

# Check Vercel CLI
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Install it:"
    echo "npm i -g vercel"
    exit 1
fi

# Check secret manager
if ! python3 scripts/secret_manager_cli.py status; then
    echo "❌ Secret manager not properly configured"
    exit 1
fi

# 🔑 Authentication
echo "🔑 Vercel authentication..."
if ! vercel whoami &> /dev/null; then
    echo "🔐 Please login to Vercel:"
    vercel login
fi

# 📊 Environment selection
ENVIRONMENT="${{1:-production}}"
echo "🎯 Target environment: $ENVIRONMENT"

# 🔐 Setup environment variables
echo "🔐 Setting up environment variables..."
python3 scripts/secret_manager_cli.py export-vercel --generate-script --output vercel_env_setup.sh

if [ -f "vercel_env_setup.sh" ]; then
    echo "🚀 Executing environment setup..."
    chmod +x vercel_env_setup.sh
    ./vercel_env_setup.sh
    rm vercel_env_setup.sh
else
    echo "❌ Failed to generate environment setup script"
    exit 1
fi

# ✅ Validate environment
echo "✅ Validating Vercel environment..."
python3 -c "
import sys
sys.path.append('.')
from deployment.vercel_secrets_setup import VercelSecretsManager
vsm = VercelSecretsManager()
result = vsm.validate_vercel_env('$ENVIRONMENT')
if not result['valid']:
    print(f'❌ Validation failed: {{result}}')
    sys.exit(1)
print('✅ Environment validation passed')
"

# 🚀 Deploy
echo "🚀 Deploying to Vercel..."
if [ "$ENVIRONMENT" = "production" ]; then
    vercel --prod
else
    vercel
fi

# 🎉 Success
echo "🎉 Deployment complete!"
echo "🔍 Check status: vercel ls"
echo "📊 View logs: vercel logs"
"""

            with open(output_file, 'w') as f:
                f.write(script_content)
            
            os.chmod(output_file, 0o755)
            print(f"✅ Generated deployment script: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate deployment script: {e}")
            return False
    
    def _check_vercel_cli(self) -> bool:
        """Check if Vercel CLI is installed"""
        try:
            result = subprocess.run(["vercel", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Vercel CLI: {result.stdout.strip()}")
                return True
            else:
                print("❌ Vercel CLI not found. Install with: npm i -g vercel")
                return False
        except FileNotFoundError:
            print("❌ Vercel CLI not found. Install with: npm i -g vercel")
            return False
    
    def _set_vercel_env_var(self, key: str, value: str, environment: str) -> bool:
        """Set a single Vercel environment variable"""
        try:
            # Use echo to pipe value to avoid shell escaping issues
            cmd = f'echo "{value}" | vercel env add {key} {environment}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.project_path)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Failed to set {key}: {e}")
            return False
    
    def _update_vercel_json(self, env_vars: List[str]) -> bool:
        """Update vercel.json with environment variable references"""
        try:
            # Load existing vercel.json or create new
            if self.vercel_json_path.exists():
                with open(self.vercel_json_path, 'r') as f:
                    vercel_config = json.load(f)
            else:
                vercel_config = {}
            
            # Add environment variables
            if 'env' not in vercel_config:
                vercel_config['env'] = {}
            
            for var in env_vars:
                vercel_config['env'][var] = f"@{var.lower()}"
            
            # Add build configuration
            if 'build' not in vercel_config:
                vercel_config['build'] = {}
            
            vercel_config['build']['env'] = {var: f"@{var.lower()}" for var in env_vars}
            
            # Save vercel.json
            with open(self.vercel_json_path, 'w') as f:
                json.dump(vercel_config, f, indent=2)
            
            print(f"✅ Updated {self.vercel_json_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update vercel.json: {e}")
            return False

def main():
    """🚀 Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="▲ Vercel Secret Management")
    parser.add_argument('action', choices=['setup', 'deploy', 'validate', 'script'])
    parser.add_argument('--environment', '-e', default='production', 
                       choices=['development', 'preview', 'production'])
    parser.add_argument('--categories', '-c', nargs='+', help='Secret categories to include')
    parser.add_argument('--output', '-o', default='deploy_to_vercel.sh', help='Script output file')
    
    args = parser.parse_args()
    
    vsm = VercelSecretsManager()
    
    if args.action == 'setup':
        success = vsm.setup_vercel_env(args.environment, args.categories)
        sys.exit(0 if success else 1)
        
    elif args.action == 'deploy':
        success = vsm.deploy_with_secrets(args.environment, args.categories)
        sys.exit(0 if success else 1)
        
    elif args.action == 'validate':
        result = vsm.validate_vercel_env(args.environment)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result['valid'] else 1)
        
    elif args.action == 'script':
        success = vsm.generate_deployment_script(args.output)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
