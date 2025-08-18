"""
Vercel-specific secret management integration
Handles deployment, environment variables, and Vercel CLI operations
"""

import os
import json
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from .secret_manager import SecretManager, Environment, SecretType


class VercelSecretManager:
    """
    Vercel-specific secret management integration
    
    Features:
    - Vercel CLI integration
    - Environment variable management
    - Deployment secret injection
    - Preview deployment handling
    """

    def __init__(self, project_path: str = ".", vercel_config_path: Optional[str] = None):
        self.project_path = Path(project_path)
        self.vercel_config_path = vercel_config_path or self.project_path / "vercel.json"
        
        # Initialize secret manager
        self.secret_manager = SecretManager(config_dir=str(self.project_path / "config"))
        
        # Setup logging
        self.logger = logging.getLogger("VercelSecretManager")
        
        # Load Vercel project info
        self.project_info = self._load_vercel_project_info()

    def _load_vercel_project_info(self) -> Dict[str, str]:
        """Load Vercel project information"""
        project_json_path = self.project_path / ".vercel" / "project.json"
        
        if project_json_path.exists():
            try:
                with open(project_json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load Vercel project info: {e}")
        
        return {}

    def _run_vercel_command(self, command: List[str], capture_output: bool = True) -> Tuple[bool, str]:
        """Run Vercel CLI command"""
        try:
            # Check if vercel CLI is available
            result = subprocess.run(
                ["which", "vercel"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, "Vercel CLI not found. Please install: npm i -g vercel"
            
            # Run the actual command
            cmd = ["vercel"] + command
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=capture_output,
                text=True
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
                
        except Exception as e:
            return False, str(e)

    def deploy_secrets_to_vercel(self, 
                                environment: Environment,
                                target_vercel_env: str = "production") -> Dict[str, bool]:
        """
        Deploy secrets to Vercel environment variables
        
        Args:
            environment: Local environment to deploy from
            target_vercel_env: Vercel environment (production, preview, development)
        """
        results = {}
        secrets = self.secret_manager.get_secrets_for_environment(environment)
        
        self.logger.info(f"Deploying {len(secrets)} secrets to Vercel {target_vercel_env}")
        
        for name, value in secrets.items():
            success, output = self._run_vercel_command([
                "env", "add", name,
                "--environment", target_vercel_env,
                "--value", value
            ])
            
            results[name] = success
            
            if success:
                self.logger.info(f"✅ Successfully deployed secret: {name}")
            else:
                self.logger.error(f"❌ Failed to deploy secret {name}: {output}")
        
        return results

    def pull_secrets_from_vercel(self, 
                                vercel_env: str = "production",
                                local_env: Environment = Environment.PRODUCTION) -> Dict[str, bool]:
        """Pull secrets from Vercel to local secret manager"""
        # Get environment variables from Vercel
        success, output = self._run_vercel_command([
            "env", "ls", "--environment", vercel_env, "--json"
        ])
        
        if not success:
            self.logger.error(f"Failed to list Vercel secrets: {output}")
            return {}
        
        try:
            env_vars = json.loads(output)
            results = {}
            
            for env_var in env_vars:
                name = env_var.get("key")
                # Note: Vercel doesn't return values for security, only metadata
                # This is mainly for syncing metadata and structure
                
                if name:
                    # Create placeholder entry in local secret manager
                    secret_type = self.secret_manager._infer_secret_type(name)
                    
                    # We can't get the actual value, so we create a placeholder
                    success = self.secret_manager.store_secret(
                        name=name,
                        value="PLACEHOLDER_FROM_VERCEL",
                        secret_type=secret_type,
                        description=f"Synced from Vercel {vercel_env} environment"
                    )
                    
                    results[name] = success
            
            return results
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Vercel env vars: {e}")
            return {}

    def sync_environments(self, 
                         source_env: Environment,
                         target_vercel_envs: List[str] = None) -> Dict[str, Dict[str, bool]]:
        """Sync secrets from local environment to multiple Vercel environments"""
        if target_vercel_envs is None:
            target_vercel_envs = ["development", "preview", "production"]
        
        results = {}
        
        for vercel_env in target_vercel_envs:
            self.logger.info(f"Syncing to Vercel {vercel_env} environment...")
            results[vercel_env] = self.deploy_secrets_to_vercel(source_env, vercel_env)
        
        return results

    def create_vercel_deployment_hook(self) -> str:
        """Create a deployment hook script for secret management"""
        hook_script = """#!/bin/bash
# Vercel Deployment Hook - Secret Management
# This script runs before deployment to ensure secrets are properly configured

set -e

echo "🔐 Starting secret management deployment hook..."

# Check if secrets are properly configured
python3 -c "
from config.vercel_secrets import VercelSecretManager
import sys

try:
    vsm = VercelSecretManager()
    health = vsm.secret_manager.get_health_report()
    
    print(f'📊 Secret Health Report:')
    print(f'   Total secrets: {health[\"total_secrets\"]}')
    print(f'   Expired secrets: {health[\"expired_secrets\"]}')
    print(f'   Expiring soon: {health[\"expiring_soon\"]}')
    
    if health['expired_secrets'] > 0:
        print('⚠️  WARNING: Some secrets have expired!')
        sys.exit(1)
    
    if health['expiring_soon'] > 0:
        print(f'⚠️  WARNING: {health[\"expiring_soon\"]} secrets expiring soon!')
    
    print('✅ Secret validation passed')
    
except Exception as e:
    print(f'❌ Secret validation failed: {e}')
    sys.exit(1)
"

echo "✅ Secret management deployment hook completed"
"""
        
        hook_path = self.project_path / "deploy" / "secret-hook.sh"
        hook_path.parent.mkdir(exist_ok=True)
        
        with open(hook_path, 'w') as f:
            f.write(hook_script)
        
        # Make executable
        os.chmod(hook_path, 0o755)
        
        return str(hook_path)

    def setup_environment_specific_configs(self) -> Dict[str, str]:
        """Setup environment-specific Vercel configurations"""
        configs = {}
        
        # Development environment
        dev_config = {
            "env": {
                "VERCEL_ENV": "development",
                "NODE_ENV": "development",
                "FLASK_ENV": "development"
            },
            "builds": [
                {
                    "src": "app.py",
                    "use": "@vercel/python"
                }
            ]
        }
        
        # Production environment  
        prod_config = {
            "env": {
                "VERCEL_ENV": "production",
                "NODE_ENV": "production", 
                "FLASK_ENV": "production"
            },
            "builds": [
                {
                    "src": "app.py",
                    "use": "@vercel/python"
                }
            ],
            "functions": {
                "app.py": {
                    "memory": 1024,
                    "maxDuration": 30
                }
            }
        }
        
        # Write configurations
        dev_path = self.project_path / "vercel.development.json"
        prod_path = self.project_path / "vercel.production.json"
        
        with open(dev_path, 'w') as f:
            json.dump(dev_config, f, indent=2)
        
        with open(prod_path, 'w') as f:
            json.dump(prod_config, f, indent=2)
        
        configs["development"] = str(dev_path)
        configs["production"] = str(prod_path)
        
        return configs

    def validate_vercel_secrets(self, environment: str = "production") -> Dict[str, Any]:
        """Validate that all required secrets are configured in Vercel"""
        validation_results = {
            "missing_secrets": [],
            "configured_secrets": [],
            "validation_errors": []
        }
        
        # Get required secrets for environment
        env_enum = Environment(environment) if environment in [e.value for e in Environment] else Environment.PRODUCTION
        local_secrets = self.secret_manager.get_secrets_for_environment(env_enum)
        
        # Check each secret in Vercel
        for secret_name in local_secrets.keys():
            success, output = self._run_vercel_command([
                "env", "ls", "--environment", environment, "--json"
            ])
            
            if success:
                try:
                    vercel_vars = json.loads(output)
                    vercel_names = [var.get("key") for var in vercel_vars]
                    
                    if secret_name in vercel_names:
                        validation_results["configured_secrets"].append(secret_name)
                    else:
                        validation_results["missing_secrets"].append(secret_name)
                        
                except json.JSONDecodeError:
                    validation_results["validation_errors"].append(f"Failed to parse Vercel env vars for {secret_name}")
            else:
                validation_results["validation_errors"].append(f"Vercel CLI error: {output}")
        
        return validation_results

    def setup_preview_deployment_secrets(self) -> bool:
        """Setup safe secrets for Vercel preview deployments"""
        try:
            # Create safe preview secrets (non-production values)
            preview_secrets = {
                "GROK4_API_KEY": "preview-key-placeholder",
                "DATABASE_URL": "sqlite:///tmp/preview.db",
                "JWT_SECRET": "preview-jwt-secret-for-testing-only",
                "ENCRYPTION_KEY": "preview-encryption-key-safe",
                "USE_REAL_APIS": "false",
                "ENABLE_MONITORING": "false"
            }
            
            # Deploy preview secrets
            results = {}
            for name, value in preview_secrets.items():
                success, output = self._run_vercel_command([
                    "env", "add", name,
                    "--environment", "preview", 
                    "--value", value
                ])
                results[name] = success
                
            return all(results.values())
            
        except Exception as e:
            self.logger.error(f"Failed to setup preview secrets: {e}")
            return False

    def generate_deployment_report(self) -> str:
        """Generate a deployment readiness report"""
        report = []
        report.append("# 🚀 Vercel Deployment Secret Report")
        report.append(f"Generated at: {datetime.now().isoformat()}")
        report.append("")
        
        # Health check
        health = self.secret_manager.get_health_report()
        report.append("## 📊 Secret Health Summary")
        report.append(f"- Total secrets: {health['total_secrets']}")
        report.append(f"- Expired secrets: {health['expired_secrets']}")
        report.append(f"- Expiring soon: {health['expiring_soon']}")
        report.append("")
        
        # Environment validation
        for env in ["development", "preview", "production"]:
            report.append(f"## 🔍 {env.title()} Environment Validation")
            validation = self.validate_vercel_secrets(env)
            
            if validation["configured_secrets"]:
                report.append(f"✅ Configured secrets ({len(validation['configured_secrets'])}):")
                for secret in validation["configured_secrets"]:
                    report.append(f"   - {secret}")
            
            if validation["missing_secrets"]:
                report.append(f"❌ Missing secrets ({len(validation['missing_secrets'])}):")
                for secret in validation["missing_secrets"]:
                    report.append(f"   - {secret}")
            
            if validation["validation_errors"]:
                report.append("⚠️ Validation errors:")
                for error in validation["validation_errors"]:
                    report.append(f"   - {error}")
            
            report.append("")
        
        return "\n".join(report)


def setup_vercel_secrets_cli():
    """CLI interface for Vercel secret management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vercel Secret Management CLI")
    parser.add_argument("action", choices=[
        "deploy", "pull", "validate", "report", "setup-preview"
    ])
    parser.add_argument("--environment", "-e", default="production",
                       choices=["development", "preview", "production"])
    parser.add_argument("--local-env", "-l", default="production", 
                       choices=["development", "staging", "production"])
    
    args = parser.parse_args()
    
    vsm = VercelSecretManager()
    
    if args.action == "deploy":
        local_env = Environment(args.local_env)
        results = vsm.deploy_secrets_to_vercel(local_env, args.environment)
        
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"🚀 Deployment Results: {success_count}/{total_count} secrets deployed")
        
        for name, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {name}")
    
    elif args.action == "validate":
        validation = vsm.validate_vercel_secrets(args.environment)
        
        print(f"🔍 Validation Results for {args.environment}:")
        print(f"   ✅ Configured: {len(validation['configured_secrets'])}")
        print(f"   ❌ Missing: {len(validation['missing_secrets'])}")
        print(f"   ⚠️ Errors: {len(validation['validation_errors'])}")
        
        if validation["missing_secrets"]:
            print("\nMissing secrets:")
            for secret in validation["missing_secrets"]:
                print(f"   - {secret}")
    
    elif args.action == "report":
        report = vsm.generate_deployment_report()
        print(report)
        
        # Save to file
        report_path = vsm.project_path / "deployment_secret_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {report_path}")
    
    elif args.action == "setup-preview":
        success = vsm.setup_preview_deployment_secrets()
        if success:
            print("✅ Preview deployment secrets configured")
        else:
            print("❌ Failed to configure preview deployment secrets")


if __name__ == "__main__":
    setup_vercel_secrets_cli()
