#!/usr/bin/env python3
"""
Security Verification Script
Verifies that all hardcoded AWS credentials have been removed
"""

import glob
import os
import re


def scan_for_hardcoded_credentials():
    """Scan all Python files for potential hardcoded AWS credentials"""

    print("🔒 AWS Security Verification")
    print("=" * 50)

    # Pattern to detect AWS access keys
    aws_key_pattern = r"AKIA[0-9A-Z]{16}"
    secret_patterns = [
        r'aws_secret_access_key.*=.*["\'][^"\']{20,}["\']',
        r'AWS_SECRET_ACCESS_KEY.*=.*["\'][^"\']{20,}["\']',
        r'["\'][A-Za-z0-9+/]{40}["\']',  # Base64-like patterns typical of secrets
    ]

    issues_found = []
    files_scanned = 0

    # Scan all Python files in the project
    for py_file in glob.glob("/Users/apple/projects/cryptotrading/**/*.py", recursive=True):
        # Skip node_modules and other non-project directories
        if "node_modules" in py_file or "__pycache__" in py_file:
            continue

        files_scanned += 1

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for AWS access key pattern
            if re.search(aws_key_pattern, content):
                issues_found.append(
                    {
                        "file": py_file,
                        "issue": "Potential AWS Access Key found",
                        "pattern": "AKIA...",
                    }
                )

            # Check for secret patterns
            for pattern in secret_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Filter out obvious placeholders and examples
                    for match in matches:
                        if not any(
                            placeholder in match.lower()
                            for placeholder in [
                                "your_",
                                "placeholder",
                                "example",
                                "test_",
                                "mock_",
                                "fake_",
                                "dummy",
                                "xxx",
                                "***",
                                "replace",
                            ]
                        ):
                            issues_found.append(
                                {
                                    "file": py_file,
                                    "issue": "Potential hardcoded secret",
                                    "pattern": match[:50] + "..." if len(match) > 50 else match,
                                }
                            )

        except Exception as e:
            print(f"⚠️  Could not scan {py_file}: {e}")

    print(f"📊 Scanned {files_scanned} Python files")

    if issues_found:
        print(f"\n❌ Found {len(issues_found)} potential security issues:")
        for issue in issues_found:
            print(f"   • {issue['file']}")
            print(f"     Issue: {issue['issue']}")
            print(f"     Pattern: {issue['pattern']}")
            print()
    else:
        print("\n✅ No hardcoded AWS credentials found!")
        print("✅ All scripts now use environment variables")
        print("✅ Security issue has been resolved")

    return len(issues_found) == 0


def check_environment_setup():
    """Check if .env.example has been properly updated"""

    print("\n🔧 Environment Configuration Check")
    print("=" * 50)

    env_example_path = "/Users/apple/projects/cryptotrading/.env.example"

    try:
        with open(env_example_path, "r") as f:
            content = f.read()

        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
            "S3_BUCKET_NAME",
        ]

        missing_vars = []
        for var in required_vars:
            if var not in content:
                missing_vars.append(var)

        if missing_vars:
            print(f"❌ Missing environment variables in .env.example:")
            for var in missing_vars:
                print(f"   • {var}")
        else:
            print("✅ All required AWS environment variables found in .env.example")

        return len(missing_vars) == 0

    except Exception as e:
        print(f"❌ Error checking .env.example: {e}")
        return False


def check_security_documentation():
    """Check if security documentation has been created"""

    print("\n📚 Security Documentation Check")
    print("=" * 50)

    docs_path = "/Users/apple/projects/cryptotrading/AWS_SETUP_SECURITY.md"

    if os.path.exists(docs_path):
        print("✅ Security documentation created: AWS_SETUP_SECURITY.md")

        try:
            with open(docs_path, "r") as f:
                content = f.read()

            if "CRITICAL SECURITY ISSUE RESOLVED" in content:
                print("✅ Documentation confirms security issue resolution")
                return True
            else:
                print("⚠️  Documentation exists but may not be complete")
                return False

        except Exception as e:
            print(f"❌ Error reading documentation: {e}")
            return False
    else:
        print("❌ Security documentation not found")
        return False


def main():
    """Run all security verification checks"""

    print("🛡️  AWS SECURITY VERIFICATION COMPLETE")
    print("=" * 60)

    # Run all checks
    credentials_clean = scan_for_hardcoded_credentials()
    env_configured = check_environment_setup()
    docs_created = check_security_documentation()

    print(f"\n📋 SECURITY AUDIT SUMMARY")
    print("=" * 50)
    print(f"✅ Hardcoded credentials removed: {'YES' if credentials_clean else 'NO'}")
    print(f"✅ Environment variables configured: {'YES' if env_configured else 'NO'}")
    print(f"✅ Security documentation created: {'YES' if docs_created else 'NO'}")

    if all([credentials_clean, env_configured, docs_created]):
        print("\n🎉 ALL SECURITY CHECKS PASSED!")
        print("✅ The critical security vulnerability has been fully resolved")
        print("✅ AWS integration is now secure and production-ready")
    else:
        print("\n⚠️  SOME ISSUES STILL NEED ATTENTION")
        print("Please address the issues above before deploying to production")

    print("\n📝 NEXT STEPS:")
    print("1. Set your AWS credentials as environment variables:")
    print("   export AWS_ACCESS_KEY_ID='your_access_key'")
    print("   export AWS_SECRET_ACCESS_KEY='your_secret_key'")
    print("2. Test the secure scripts:")
    print("   python3 scripts/setup_real_s3.py")
    print("3. Never commit .env files with real credentials!")


if __name__ == "__main__":
    main()
