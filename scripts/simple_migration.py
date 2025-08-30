#!/usr/bin/env python3
"""
Simple migration script to test the existing secret manager
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_existing_secret_manager():
    """Test the existing secret manager functionality"""
    try:
        print("🔍 Testing existing secret manager...")

        # Import the existing secret manager
        from config.secret_manager import setup_secret_manager, validate_deployment_secrets

        print("✅ Secret manager imported successfully")

        # Initialize
        sm = setup_secret_manager()
        print(f"✅ Secret manager initialized")

        # Test storing a secret
        success = sm.store_secret("TEST_KEY", "test_value_123", "testing")
        print(f"✅ Store secret test: {'Success' if success else 'Failed'}")

        # Test retrieving a secret
        value = sm.get_secret("TEST_KEY")
        print(f"✅ Retrieve secret test: {'Success' if value == 'test_value_123' else 'Failed'}")

        # List secrets
        secrets = sm.list_secrets()
        print(f"✅ List secrets: {secrets}")

        # Validate
        validation = validate_deployment_secrets()
        print(f"✅ Validation: {validation}")

        return True

    except Exception as e:
        print(f"❌ Error testing secret manager: {e}")
        return False


def migrate_existing_env():
    """Migrate existing .env files using the current secret manager"""
    try:
        print("\n🔄 Starting simple migration...")

        from config.secret_manager import setup_secret_manager

        sm = setup_secret_manager()

        # Sync from .env file
        print("📄 Syncing from .env file...")
        success = sm.sync_from_env(".env.example")  # Start with example

        if success:
            print("✅ Migration from .env.example completed")
        else:
            print("⚠️ No .env.example found or sync failed")

        # Try production file
        prod_env = Path(".env.production")
        if prod_env.exists():
            print("📄 Syncing from .env.production...")
            success = sm.sync_from_env(".env.production")
            if success:
                print("✅ Migration from .env.production completed")

        # List what we have
        print("\n📋 Current secrets:")
        secrets = sm.list_secrets()
        for category, keys in secrets.items():
            print(f"  📁 {category}: {len(keys)} secrets")
            for key in keys:
                print(f"    - {key}")

        return True

    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False


def generate_env_files():
    """Generate .env files for different environments"""
    try:
        print("\n📝 Generating .env files...")

        from config.secret_manager import setup_secret_manager

        sm = setup_secret_manager()

        # Generate development .env
        success = sm.generate_env_file(
            template_file=".env.example", output_file=".env.development", environment="development"
        )

        if success:
            print("✅ Generated .env.development")
        else:
            print("⚠️ Failed to generate .env.development")

        # Generate for Vercel
        print("\n▲ Preparing for Vercel...")
        vercel_export = sm.export_for_vercel()

        if vercel_export:
            print(f"✅ Vercel export ready: {len(vercel_export.get('secrets', {}))} secrets")

            # Save CLI commands to file
            if "cli_commands" in vercel_export:
                with open("vercel_setup.sh", "w") as f:
                    f.write("#!/bin/bash\n")
                    f.write("# Vercel Environment Variables Setup\n\n")
                    for cmd in vercel_export["cli_commands"]:
                        f.write(f"{cmd}\n")
                os.chmod("vercel_setup.sh", 0o755)
                print("✅ Created vercel_setup.sh script")

        return True

    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False


def main():
    """Main migration process"""
    print("🔐 CRYPTO TRADING - SECRET MANAGER SETUP")
    print("=" * 50)

    # Test basic functionality
    if not test_existing_secret_manager():
        print("❌ Basic functionality test failed")
        return

    # Migrate existing secrets
    if not migrate_existing_env():
        print("❌ Migration failed")
        return

    # Generate environment files
    if not generate_env_files():
        print("❌ Environment file generation failed")
        return

    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Review generated .env.development file")
    print("2. Update secrets with real values for production")
    print("3. Run ./vercel_setup.sh to configure Vercel")
    print("4. Test deployment with scripts/deploy_with_secrets.sh")


if __name__ == "__main__":
    main()
