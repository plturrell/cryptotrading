"""
Development Workflow Secret Management
Handles local development, team collaboration, and deployment preparation
"""

import os
import shutil
import json
from typing import Dict, List, Optional, Set
from pathlib import Path
import logging
from datetime import datetime
import subprocess

from .secret_manager import SecretManager, Environment, SecretType


class DevWorkflowManager:
    """
    Development workflow secret management
    
    Features:
    - Local development setup
    - Team secret sharing (without exposing values)
    - Pre-deployment validation
    - Environment synchronization
    - Secret template management
    """

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.secret_manager = SecretManager(config_dir=str(self.project_path / "config"))
        
        # Setup logging
        self.logger = logging.getLogger("DevWorkflowManager")
        
        # Paths
        self.env_templates_dir = self.project_path / "config" / "env_templates"
        self.env_templates_dir.mkdir(exist_ok=True)

    def initialize_development_environment(self, 
                                         developer_name: Optional[str] = None) -> Dict[str, bool]:
        """Initialize development environment for a new developer"""
        results = {}
        
        self.logger.info("🚀 Initializing development environment...")
        
        # 1. Create .env from template if it doesn't exist
        env_file = self.project_path / ".env"
        env_example = self.project_path / ".env.example"
        
        if not env_file.exists() and env_example.exists():
            shutil.copy(env_example, env_file)
            os.chmod(env_file, 0o600)
            results["env_file_created"] = True
            self.logger.info("✅ Created .env file from template")
        else:
            results["env_file_created"] = False
        
        # 2. Validate required development secrets
        validation_results = self.validate_development_secrets()
        results["validation"] = validation_results["is_valid"]
        
        # 3. Create development-safe secrets if missing
        if not validation_results["is_valid"]:
            created_secrets = self.create_development_secrets()
            results["dev_secrets_created"] = created_secrets
        
        # 4. Setup git hooks for secret protection
        git_hooks_setup = self.setup_git_hooks()
        results["git_hooks"] = git_hooks_setup
        
        # 5. Generate developer-specific documentation
        if developer_name:
            doc_path = self.generate_developer_documentation(developer_name)
            results["documentation"] = doc_path
        
        return results

    def validate_development_secrets(self) -> Dict[str, any]:
        """Validate that all required development secrets are present and valid"""
        validation_results = {
            "is_valid": True,
            "missing_secrets": [],
            "invalid_secrets": [],
            "recommendations": []
        }
        
        # Required development secrets
        required_secrets = {
            "GROK4_API_KEY": SecretType.API_KEY,
            "PERPLEXITY_API_KEY": SecretType.API_KEY,
            "DATABASE_URL": SecretType.DATABASE_URL,
            "JWT_SECRET": SecretType.JWT_SECRET,
            "ENCRYPTION_KEY": SecretType.ENCRYPTION_KEY
        }
        
        dev_secrets = self.secret_manager.get_secrets_for_environment(Environment.DEVELOPMENT)
        
        for secret_name, expected_type in required_secrets.items():
            if secret_name not in dev_secrets:
                validation_results["missing_secrets"].append(secret_name)
                validation_results["is_valid"] = False
            else:
                # Validate secret format
                secret_value = dev_secrets[secret_name]
                try:
                    self.secret_manager._validate_secret(secret_value, expected_type)
                except ValueError as e:
                    validation_results["invalid_secrets"].append({
                        "name": secret_name,
                        "error": str(e)
                    })
                    validation_results["is_valid"] = False
        
        # Add recommendations
        if validation_results["missing_secrets"]:
            validation_results["recommendations"].append(
                "Run 'python config/dev_workflow.py create-dev-secrets' to generate development secrets"
            )
        
        return validation_results

    def create_development_secrets(self) -> Dict[str, bool]:
        """Create safe development secrets"""
        dev_secrets = {
            "GROK4_API_KEY": "dev-grok4-key-placeholder-123456789",
            "PERPLEXITY_API_KEY": "dev-perplexity-key-placeholder-123456789", 
            "DATABASE_URL": "sqlite:///./data/development.db",
            "REDIS_URL": "redis://localhost:6379/0",
            "JWT_SECRET": "dev-jwt-secret-32-chars-minimum-safe-for-development",
            "ENCRYPTION_KEY": "dev-encryption-key-for-local-testing-only-123",
            "USE_REAL_APIS": "false",
            "ENABLE_CACHING": "true",
            "ENABLE_MONITORING": "false",
            "SENTRY_DSN": "https://dev-placeholder@sentry.io/dev",
            "VERCEL_ENV": "development",
            "ENVIRONMENT": "development"
        }
        
        results = {}
        
        for name, value in dev_secrets.items():
            # Infer secret type
            secret_type = self.secret_manager._infer_secret_type(name)
            
            success = self.secret_manager.store_secret(
                name=name,
                value=value,
                secret_type=secret_type,
                description=f"Development secret for {name}",
                tags=["development", "auto-generated"]
            )
            
            results[name] = success
        
        return results

    def setup_git_hooks(self) -> bool:
        """Setup git hooks to prevent secret commits"""
        try:
            hooks_dir = self.project_path / ".git" / "hooks"
            if not hooks_dir.exists():
                self.logger.warning("Git hooks directory not found")
                return False
            
            # Pre-commit hook to check for secrets
            pre_commit_hook = hooks_dir / "pre-commit"
            
            hook_content = """#!/bin/bash
# Pre-commit hook to prevent secret commits

echo "🔍 Checking for secrets in staged files..."

# Check for potential secrets in staged files
if git diff --cached --name-only | grep -E "\\.env$|\\.env\\." | grep -v "\\.env\\.example$"; then
    echo "❌ Error: .env files detected in staged changes!"
    echo "   Please ensure .env files are not committed to version control"
    echo "   Use .env.example for templates instead"
    exit 1
fi

# Check for potential API keys or secrets in code
if git diff --cached | grep -iE "(api_key|secret|password|token)\\s*=\\s*['\"][^'\"\\s]{16,}"; then
    echo "⚠️  Warning: Potential secrets detected in staged code!"
    echo "   Please review the following patterns:"
    git diff --cached | grep -iE "(api_key|secret|password|token)\\s*=\\s*['\"][^'\"\\s]{16,}" || true
    echo ""
    read -p "Continue with commit anyway? (y/N): " confirm
    if [[ $confirm != [yY] ]]; then
        echo "Commit cancelled"
        exit 1
    fi
fi

echo "✅ Secret check passed"
"""
            
            with open(pre_commit_hook, 'w') as f:
                f.write(hook_content)
            
            os.chmod(pre_commit_hook, 0o755)
            
            self.logger.info("✅ Git pre-commit hook installed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup git hooks: {e}")
            return False

    def generate_developer_documentation(self, developer_name: str) -> str:
        """Generate developer-specific secret management documentation"""
        doc_content = f"""# 🔐 Secret Management Guide for {developer_name}

## 📋 Quick Start

### 1. Initial Setup
```bash
# Initialize development environment
python config/dev_workflow.py init --developer {developer_name}

# Validate your setup
python config/dev_workflow.py validate
```

### 2. Working with Secrets

#### Loading Secrets in Code
```python
from config.secret_manager import load_secrets_from_env

# Load secrets for current environment
secrets = load_secrets_from_env()
api_key = secrets.get('GROK4_API_KEY')
```

#### Adding New Secrets
```python
from config.secret_manager import create_secret_manager, SecretType

sm = create_secret_manager()
sm.store_secret(
    name="NEW_API_KEY",
    value="your-secret-value",
    secret_type=SecretType.API_KEY,
    description="Description of the secret",
    rotation_days=90
)
```

### 3. Environment Management

#### Development Environment
- Uses safe placeholder values
- Points to local databases
- No real API calls by default

#### Production Environment  
- Real API keys and credentials
- Production databases
- Full monitoring enabled

### 4. Deployment Workflow

#### Before Deploying
```bash
# Validate all secrets
python config/vercel_secrets.py validate --environment production

# Generate deployment report
python config/vercel_secrets.py report

# Deploy secrets to Vercel
python config/vercel_secrets.py deploy --environment production
```

### 5. Security Best Practices

#### ✅ DO:
- Use the SecretManager for all sensitive data
- Regularly rotate API keys (every 90 days)
- Use different secrets for each environment
- Review expiring secrets weekly

#### ❌ DON'T:
- Commit .env files to git
- Share production secrets in Slack/email
- Use production secrets in development
- Hardcode secrets in application code

### 6. Emergency Procedures

#### If Secrets Are Compromised:
1. Immediately rotate affected secrets
2. Update Vercel environment variables
3. Redeploy affected environments
4. Review audit logs for access patterns

### 7. Team Collaboration

#### Adding a New Secret:
1. Add to local secret manager
2. Update .env.example template
3. Deploy to appropriate Vercel environments
4. Document in team chat

#### Sharing Secret Metadata (NOT VALUES):
```bash
# Export secret structure for team
python config/dev_workflow.py export-structure --output team-secrets.json
```

## 📞 Support

For issues with secret management:
1. Check the audit logs: `config/secrets/audit.log`
2. Run health report: `python config/dev_workflow.py health`
3. Review this documentation

---
Generated for: {developer_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        doc_path = self.project_path / f"SECRET_MANAGEMENT_GUIDE_{developer_name.upper()}.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)
        
        self.logger.info(f"📚 Developer documentation created: {doc_path}")
        return str(doc_path)

    def export_secret_structure(self, output_path: str) -> Dict[str, Any]:
        """Export secret structure (metadata only) for team sharing"""
        structure = {
            "environments": [e.value for e in Environment],
            "secret_types": [t.value for t in SecretType],
            "secrets": {}
        }
        
        for name, metadata in self.secret_manager.metadata.items():
            structure["secrets"][name] = {
                "type": metadata.secret_type.value,
                "environment": metadata.environment.value,
                "description": metadata.description,
                "tags": metadata.tags,
                "has_expiration": metadata.expires_at is not None,
                "requires_rotation": metadata.rotation_days is not None
            }
        
        with open(output_path, 'w') as f:
            json.dump(structure, f, indent=2)
        
        return structure

    def sync_env_files(self) -> Dict[str, bool]:
        """Synchronize .env files across environments"""
        results = {}
        
        # Generate .env files for each environment
        for env in Environment:
            try:
                env_file_path = self.secret_manager.export_to_env_file(
                    target_env=env,
                    output_path=f".env.{env.value}"
                )
                results[env.value] = True
                self.logger.info(f"✅ Generated {env_file_path}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to generate .env.{env.value}: {e}")
                results[env.value] = False
        
        return results

    def pre_deployment_check(self, target_environment: str = "production") -> Dict[str, Any]:
        """Run comprehensive pre-deployment secret validation"""
        check_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # 1. Validate local secrets
            env_enum = Environment(target_environment)
            local_secrets = self.secret_manager.get_secrets_for_environment(env_enum)
            
            if not local_secrets:
                check_results["errors"].append(f"No secrets found for {target_environment} environment")
                check_results["passed"] = False
            
            # 2. Check for expiring secrets
            expiring = self.secret_manager.check_expiring_secrets(days_ahead=30)
            if expiring:
                for secret in expiring:
                    check_results["warnings"].append(
                        f"Secret '{secret.name}' expires on {secret.expires_at}"
                    )
            
            # 3. Validate secret formats
            for name, value in local_secrets.items():
                if name in self.secret_manager.metadata:
                    metadata = self.secret_manager.metadata[name]
                    try:
                        self.secret_manager._validate_secret(value, metadata.secret_type)
                    except ValueError as e:
                        check_results["errors"].append(f"Invalid secret format for '{name}': {e}")
                        check_results["passed"] = False
            
            # 4. Check for development secrets in production
            if target_environment == "production":
                dev_indicators = ["dev-", "test-", "placeholder", "localhost", "127.0.0.1"]
                for name, value in local_secrets.items():
                    if any(indicator in value.lower() for indicator in dev_indicators):
                        check_results["errors"].append(
                            f"Development placeholder detected in production secret: {name}"
                        )
                        check_results["passed"] = False
            
            # 5. Check required secrets checklist
            required_for_production = [
                "GROK4_API_KEY", "DATABASE_URL", "JWT_SECRET", "ENCRYPTION_KEY"
            ]
            
            if target_environment == "production":
                for required in required_for_production:
                    if required not in local_secrets:
                        check_results["errors"].append(f"Required production secret missing: {required}")
                        check_results["passed"] = False
            
            # 6. Add recommendations
            if len(local_secrets) > 20:
                check_results["recommendations"].append("Consider organizing secrets into categories")
            
            if target_environment == "production":
                check_results["recommendations"].append("Ensure all API keys have rate limiting configured")
                check_results["recommendations"].append("Verify database connections are secure (SSL enabled)")
            
        except Exception as e:
            check_results["errors"].append(f"Pre-deployment check failed: {e}")
            check_results["passed"] = False
        
        return check_results

    def create_team_secret_template(self) -> str:
        """Create a template file for team collaboration"""
        template_data = {
            "project": "Crypto Trading Platform",
            "environments": {},
            "secret_requirements": {},
            "setup_instructions": {
                "1": "Copy this template to your local environment",
                "2": "Fill in actual secret values",
                "3": "Run initialization: python config/dev_workflow.py init",
                "4": "Validate setup: python config/dev_workflow.py validate"
            }
        }
        
        # Export structure for each environment
        for env in Environment:
            secrets = self.secret_manager.list_secrets(environment=env)
            template_data["environments"][env.value] = {
                "secrets": [
                    {
                        "name": s.name,
                        "type": s.secret_type.value,
                        "description": s.description,
                        "required": True,
                        "example": self.secret_manager._generate_placeholder(s.secret_type)
                    }
                    for s in secrets
                ]
            }
        
        # Secret requirements
        for secret_type in SecretType:
            template_data["secret_requirements"][secret_type.value] = {
                "description": self._get_secret_type_description(secret_type),
                "format_requirements": self._get_secret_format_requirements(secret_type)
            }
        
        template_path = self.env_templates_dir / "team_secrets_template.json"
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        return str(template_path)

    def _get_secret_type_description(self, secret_type: SecretType) -> str:
        """Get description for secret type"""
        descriptions = {
            SecretType.API_KEY: "API key for external service authentication",
            SecretType.DATABASE_URL: "Database connection string",
            SecretType.JWT_SECRET: "Secret for JWT token signing and verification",
            SecretType.ENCRYPTION_KEY: "Key for encrypting sensitive data at rest",
            SecretType.WEBHOOK_URL: "URL endpoint for webhook notifications",
            SecretType.OAUTH_SECRET: "OAuth client secret for authentication flows",
            SecretType.GENERIC: "Generic secret value"
        }
        return descriptions.get(secret_type, "Generic secret value")

    def _get_secret_format_requirements(self, secret_type: SecretType) -> List[str]:
        """Get format requirements for secret type"""
        requirements = {
            SecretType.API_KEY: ["Minimum 16 characters", "Alphanumeric with special characters"],
            SecretType.DATABASE_URL: ["Valid connection string format", "Include protocol (postgresql://, mysql://, etc.)"],
            SecretType.JWT_SECRET: ["Minimum 32 characters", "High entropy recommended"],
            SecretType.ENCRYPTION_KEY: ["Minimum 32 characters", "Base64 encoded recommended"],
            SecretType.WEBHOOK_URL: ["Valid HTTPS URL", "Accessible endpoint"],
            SecretType.OAUTH_SECRET: ["As provided by OAuth provider", "Keep confidential"],
            SecretType.GENERIC: ["As required by your application"]
        }
        return requirements.get(secret_type, ["As required by your application"])

    def generate_secret_rotation_schedule(self) -> Dict[str, Any]:
        """Generate a schedule for secret rotation"""
        schedule = {
            "immediate_action_required": [],
            "rotate_this_week": [],
            "rotate_this_month": [],
            "no_rotation_needed": []
        }
        
        now = datetime.now()
        
        for name, metadata in self.secret_manager.metadata.items():
            if not metadata.rotation_days:
                schedule["no_rotation_needed"].append(name)
                continue
            
            if metadata.expires_at:
                days_until_expiry = (metadata.expires_at - now).days
                
                if days_until_expiry < 0:
                    schedule["immediate_action_required"].append({
                        "name": name,
                        "expired_days_ago": abs(days_until_expiry)
                    })
                elif days_until_expiry <= 7:
                    schedule["rotate_this_week"].append({
                        "name": name,
                        "expires_in_days": days_until_expiry
                    })
                elif days_until_expiry <= 30:
                    schedule["rotate_this_month"].append({
                        "name": name,
                        "expires_in_days": days_until_expiry
                    })
        
        return schedule


def cli_interface():
    """Command line interface for development workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Development Workflow Secret Management")
    parser.add_argument("action", choices=[
        "init", "validate", "create-dev-secrets", "pre-check", 
        "sync-env", "team-template", "rotation-schedule", "health"
    ])
    parser.add_argument("--developer", "-d", help="Developer name for initialization")
    parser.add_argument("--environment", "-e", default="development",
                       choices=["development", "staging", "production"])
    
    args = parser.parse_args()
    
    dwm = DevWorkflowManager()
    
    if args.action == "init":
        results = dwm.initialize_development_environment(args.developer)
        print("🚀 Development Environment Initialization Results:")
        for key, result in results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {key}")
    
    elif args.action == "validate":
        results = dwm.validate_development_secrets()
        if results["is_valid"]:
            print("✅ Development environment validation passed")
        else:
            print("❌ Development environment validation failed")
            
            if results["missing_secrets"]:
                print(f"\nMissing secrets:")
                for secret in results["missing_secrets"]:
                    print(f"   - {secret}")
            
            if results["invalid_secrets"]:
                print(f"\nInvalid secrets:")
                for invalid in results["invalid_secrets"]:
                    print(f"   - {invalid['name']}: {invalid['error']}")
    
    elif args.action == "create-dev-secrets":
        results = dwm.create_development_secrets()
        success_count = sum(results.values())
        print(f"🔐 Created {success_count}/{len(results)} development secrets")
    
    elif args.action == "pre-check":
        results = dwm.pre_deployment_check(args.environment)
        if results["passed"]:
            print(f"✅ Pre-deployment check passed for {args.environment}")
        else:
            print(f"❌ Pre-deployment check failed for {args.environment}")
            
            for error in results["errors"]:
                print(f"   ❌ {error}")
            
            for warning in results["warnings"]:
                print(f"   ⚠️ {warning}")
    
    elif args.action == "team-template":
        template_path = dwm.create_team_secret_template()
        print(f"📋 Team secret template created: {template_path}")
    
    elif args.action == "rotation-schedule":
        schedule = dwm.generate_secret_rotation_schedule()
        print("🔄 Secret Rotation Schedule:")
        
        if schedule["immediate_action_required"]:
            print("\n🚨 IMMEDIATE ACTION REQUIRED:")
            for item in schedule["immediate_action_required"]:
                print(f"   - {item['name']} (expired {item['expired_days_ago']} days ago)")
        
        if schedule["rotate_this_week"]:
            print("\n📅 Rotate This Week:")
            for item in schedule["rotate_this_week"]:
                print(f"   - {item['name']} (expires in {item['expires_in_days']} days)")
    
    elif args.action == "health":
        health = dwm.secret_manager.get_health_report()
        print("📊 Secret Manager Health Report:")
        print(json.dumps(health, indent=2))


if __name__ == "__main__":
    cli_interface()
