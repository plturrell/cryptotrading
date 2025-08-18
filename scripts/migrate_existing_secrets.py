#!/usr/bin/env python3
"""
Migration script to import existing secrets into the new secret management system
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.secret_manager import SecretManager, Environment, SecretType
from config.dev_workflow import DevWorkflowManager


def setup_logging():
    """Setup logging for migration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def migrate_env_files():
    """Migrate existing .env files to secret manager"""
    project_path = Path.cwd()
    sm = SecretManager()
    
    migration_results = {
        "files_processed": [],
        "secrets_migrated": {},
        "errors": []
    }
    
    # Mapping of .env files to environments
    env_file_mapping = {
        ".env": Environment.DEVELOPMENT,
        ".env.production": Environment.PRODUCTION,
        ".env.staging": Environment.STAGING,
        ".env.local": Environment.DEVELOPMENT
    }
    
    print("🔄 Starting migration of existing .env files...")
    
    for env_file, environment in env_file_mapping.items():
        env_path = project_path / env_file
        
        if env_path.exists():
            print(f"\n📁 Processing {env_file} -> {environment.value} environment")
            
            try:
                # Read .env file
                with open(env_path, 'r') as f:
                    lines = f.readlines()
                
                secrets_in_file = {}
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value
                    if '=' not in line:
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')  # Remove quotes
                    
                    # Skip empty values or obvious placeholders
                    if not value or value in ['your-key-here', 'placeholder', 'TODO']:
                        continue
                    
                    secrets_in_file[key] = value
                
                # Import secrets
                for key, value in secrets_in_file.items():
                    secret_type = sm._infer_secret_type(key)
                    
                    # Set expiration for API keys
                    expires_in_days = None
                    rotation_days = None
                    
                    if secret_type == SecretType.API_KEY:
                        rotation_days = 90  # Rotate API keys every 90 days
                        expires_in_days = 90
                    elif secret_type == SecretType.JWT_SECRET:
                        rotation_days = 180  # Rotate JWT secrets every 6 months
                    
                    success = sm.store_secret(
                        name=key,
                        value=value,
                        secret_type=secret_type,
                        description=f"Migrated from {env_file}",
                        expires_in_days=expires_in_days,
                        rotation_days=rotation_days,
                        tags=["migrated", environment.value]
                    )
                    
                    if success:
                        print(f"   ✅ Migrated: {key}")
                        if key not in migration_results["secrets_migrated"]:
                            migration_results["secrets_migrated"][key] = []
                        migration_results["secrets_migrated"][key].append(environment.value)
                    else:
                        print(f"   ❌ Failed: {key}")
                        migration_results["errors"].append(f"Failed to migrate {key} from {env_file}")
                
                migration_results["files_processed"].append(env_file)
                
            except Exception as e:
                error_msg = f"Error processing {env_file}: {e}"
                migration_results["errors"].append(error_msg)
                print(f"   ❌ {error_msg}")
    
    return migration_results


def migrate_vercel_config():
    """Migrate existing Vercel configuration"""
    project_path = Path.cwd()
    vercel_json = project_path / "vercel.json"
    
    if not vercel_json.exists():
        print("⚠️  No vercel.json found to migrate")
        return {}
    
    print("\n📄 Migrating Vercel configuration...")
    
    try:
        import json
        with open(vercel_json, 'r') as f:
            config = json.load(f)
        
        sm = SecretManager()
        migrated = {}
        
        # Migrate environment variables from vercel.json
        if "env" in config:
            for key, value in config["env"].items():
                # These are usually not secrets but configuration
                secret_type = SecretType.GENERIC
                
                success = sm.store_secret(
                    name=key,
                    value=str(value),
                    secret_type=secret_type,
                    description=f"Migrated from vercel.json env config",
                    tags=["vercel", "config", "migrated"]
                )
                
                migrated[key] = success
                status = "✅" if success else "❌"
                print(f"   {status} {key}")
        
        return migrated
        
    except Exception as e:
        print(f"❌ Error migrating vercel.json: {e}")
        return {}


def setup_production_secrets():
    """Setup production secrets with placeholder values"""
    print("\n🏭 Setting up production secret templates...")
    
    sm = SecretManager()
    
    # Production secrets that need real values
    production_secrets = {
        "GROK4_API_KEY": {
            "type": SecretType.API_KEY,
            "description": "Grok4 AI API key for production",
            "placeholder": "REPLACE_WITH_REAL_GROK4_KEY"
        },
        "PERPLEXITY_API_KEY": {
            "type": SecretType.API_KEY,
            "description": "Perplexity AI API key for production",
            "placeholder": "REPLACE_WITH_REAL_PERPLEXITY_KEY"
        },
        "DATABASE_URL": {
            "type": SecretType.DATABASE_URL,
            "description": "Production database connection string",
            "placeholder": "postgresql://user:pass@prod-db.host:5432/cryptotrading"
        },
        "JWT_SECRET": {
            "type": SecretType.JWT_SECRET,
            "description": "Production JWT signing secret",
            "placeholder": "GENERATE_SECURE_JWT_SECRET_32_CHARS_MINIMUM"
        },
        "BINANCE_API_KEY": {
            "type": SecretType.API_KEY,
            "description": "Binance API key for live trading",
            "placeholder": "REPLACE_WITH_REAL_BINANCE_API_KEY"
        },
        "BINANCE_API_SECRET": {
            "type": SecretType.API_KEY,
            "description": "Binance API secret for live trading",
            "placeholder": "REPLACE_WITH_REAL_BINANCE_API_SECRET"
        }
    }
    
    results = {}
    
    for name, config in production_secrets.items():
        success = sm.store_secret(
            name=name,
            value=config["placeholder"],
            secret_type=config["type"],
            description=config["description"],
            rotation_days=90 if config["type"] == SecretType.API_KEY else None,
            tags=["production", "template", "requires-real-value"]
        )
        
        results[name] = success
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    return results


def generate_migration_report(migration_results: Dict) -> str:
    """Generate migration report"""
    report = []
    report.append("# 🔄 Secret Migration Report")
    report.append(f"Migration completed at: {datetime.now().isoformat()}")
    report.append("")
    
    # Files processed
    if migration_results.get("files_processed"):
        report.append("## 📁 Files Processed")
        for file_name in migration_results["files_processed"]:
            report.append(f"- {file_name}")
        report.append("")
    
    # Secrets migrated
    if migration_results.get("secrets_migrated"):
        report.append("## 🔐 Secrets Migrated")
        for secret_name, environments in migration_results["secrets_migrated"].items():
            env_list = ", ".join(environments)
            report.append(f"- **{secret_name}** → {env_list}")
        report.append("")
    
    # Errors
    if migration_results.get("errors"):
        report.append("## ❌ Errors Encountered")
        for error in migration_results["errors"]:
            report.append(f"- {error}")
        report.append("")
    
    # Next steps
    report.append("## 🎯 Next Steps")
    report.append("1. Review migrated secrets for accuracy")
    report.append("2. Replace placeholder values with real production secrets")
    report.append("3. Deploy secrets to Vercel environments")
    report.append("4. Test deployment with new secret system")
    report.append("5. Update application code to use new config system")
    
    return "\n".join(report)


def main():
    """Main migration process"""
    setup_logging()
    
    print("🔄 CRYPTO TRADING PLATFORM - SECRET MIGRATION")
    print("=" * 50)
    
    # 1. Migrate existing .env files
    migration_results = migrate_env_files()
    
    # 2. Migrate Vercel configuration
    vercel_results = migrate_vercel_config()
    migration_results["vercel_config"] = vercel_results
    
    # 3. Setup production secret templates
    prod_results = setup_production_secrets()
    migration_results["production_templates"] = prod_results
    
    # 4. Generate and save migration report
    report = generate_migration_report(migration_results)
    
    report_path = Path.cwd() / f"SECRET_MIGRATION_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Migration report saved: {report_path}")
    print("\n" + "=" * 50)
    print("🎉 Migration completed!")
    
    # 5. Show summary
    total_secrets = len(migration_results.get("secrets_migrated", {}))
    total_errors = len(migration_results.get("errors", []))
    
    print(f"📊 Summary:")
    print(f"   ✅ Secrets migrated: {total_secrets}")
    print(f"   ❌ Errors: {total_errors}")
    
    if total_errors == 0:
        print("\n🚀 Ready for next steps:")
        print("   1. Review migrated secrets")
        print("   2. Add real production values")
        print("   3. Deploy to Vercel")
    else:
        print("\n⚠️  Please resolve errors before proceeding")


if __name__ == "__main__":
    main()
