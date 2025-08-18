#!/usr/bin/env python3
"""
🔐 Secret Manager for Crypto Trading System
Handles secure storage and retrieval of API keys and secrets for local development and deployment
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib

logger = logging.getLogger(__name__)

class SecretManager:
    """
    🔐 Secure secret management for local development and deployment
    
    Features:
    - Encrypted local storage
    - Environment variable management
    - Container/Vercel deployment support
    - Key rotation and validation
    """
    
    def __init__(self, 
                 config_dir: str = "config",
                 env_file: str = ".env",
                 master_key: Optional[str] = None):
        self.config_dir = Path(config_dir)
        self.env_file = Path(env_file)
        self.secrets_file = self.config_dir / "secrets.encrypted"
        self.key_file = self.config_dir / ".encryption_key"
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize encryption
        self.cipher = self._init_encryption(master_key)
        
    def _init_encryption(self, master_key: Optional[str] = None) -> Fernet:
        """Initialize encryption with master key or generate new one"""
        try:
            if master_key:
                # Use provided master key
                key = self._derive_key_from_password(master_key)
            elif self.key_file.exists():
                # Load existing key
                with open(self.key_file, 'rb') as f:
                    key = f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                # Secure the key file
                os.chmod(self.key_file, 0o600)
                logger.info(f"🔑 Generated new encryption key: {self.key_file}")
                
            return Fernet(key)
        except Exception as e:
            logger.error(f"❌ Failed to initialize encryption: {e}")
            raise
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password"""
        password_bytes = password.encode()
        salt = b'cryptotrading_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password_bytes))
    
    def store_secret(self, key: str, value: str, category: str = "general") -> bool:
        """
        🔒 Store a secret securely
        
        Args:
            key: Secret identifier (e.g., 'GROK4_API_KEY')
            value: Secret value
            category: Organization category (e.g., 'ai', 'trading', 'database')
        """
        try:
            # Load existing secrets
            secrets = self._load_secrets()
            
            # Organize by category
            if category not in secrets:
                secrets[category] = {}
            
            # Encrypt and store
            encrypted_value = self.cipher.encrypt(value.encode()).decode()
            secrets[category][key] = {
                'value': encrypted_value,
                'updated': self._get_timestamp()
            }
            
            # Save secrets
            self._save_secrets(secrets)
            logger.info(f"🔒 Stored secret: {key} in category: {category}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store secret {key}: {e}")
            return False
    
    def get_secret(self, key: str, category: str = None) -> Optional[str]:
        """
        🔓 Retrieve a secret
        
        Args:
            key: Secret identifier
            category: Optional category to search in
        """
        try:
            secrets = self._load_secrets()
            
            if category:
                # Search in specific category
                if category in secrets and key in secrets[category]:
                    encrypted_value = secrets[category][key]['value']
                    return self.cipher.decrypt(encrypted_value.encode()).decode()
            else:
                # Search all categories
                for cat_name, cat_secrets in secrets.items():
                    if key in cat_secrets:
                        encrypted_value = cat_secrets[key]['value']
                        return self.cipher.decrypt(encrypted_value.encode()).decode()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve secret {key}: {e}")
            return None
    
    def list_secrets(self, category: str = None) -> Dict[str, List[str]]:
        """📋 List all secret keys (not values) organized by category"""
        try:
            secrets = self._load_secrets()
            
            if category:
                return {category: list(secrets.get(category, {}).keys())}
            else:
                return {cat: list(keys.keys()) for cat, keys in secrets.items()}
                
        except Exception as e:
            logger.error(f"❌ Failed to list secrets: {e}")
            return {}
    
    def delete_secret(self, key: str, category: str = None) -> bool:
        """🗑️ Delete a secret"""
        try:
            secrets = self._load_secrets()
            
            if category:
                if category in secrets and key in secrets[category]:
                    del secrets[category][key]
                    self._save_secrets(secrets)
                    logger.info(f"🗑️ Deleted secret: {key} from category: {category}")
                    return True
            else:
                # Search and delete from any category
                for cat_name, cat_secrets in secrets.items():
                    if key in cat_secrets:
                        del cat_secrets[key]
                        self._save_secrets(secrets)
                        logger.info(f"🗑️ Deleted secret: {key} from category: {cat_name}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to delete secret {key}: {e}")
            return False
    
    def generate_env_file(self, 
                         template_file: str = ".env.example",
                         output_file: str = ".env",
                         environment: str = "development") -> bool:
        """
        📝 Generate .env file from secrets and template
        
        Args:
            template_file: Template file to use as base
            output_file: Output environment file
            environment: Target environment (development, production, etc.)
        """
        try:
            template_path = Path(template_file)
            if not template_path.exists():
                logger.error(f"❌ Template file not found: {template_file}")
                return False
            
            # Read template
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Load secrets
            secrets = self._load_secrets()
            
            # Generate .env content
            env_content = f"# Generated by SecretManager for {environment} environment\n"
            env_content += f"# Generated at: {self._get_timestamp()}\n\n"
            
            # Process template line by line
            for line in template_content.split('\n'):
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    env_content += line + '\n'
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, template_value = line.split('=', 1)
                    key = key.strip()
                    
                    # Try to find secret value
                    secret_value = self.get_secret(key)
                    
                    if secret_value:
                        env_content += f"{key}={secret_value}\n"
                    else:
                        # Keep template value if no secret found
                        env_content += line + '\n'
                else:
                    env_content += line + '\n'
            
            # Write .env file
            with open(output_file, 'w') as f:
                f.write(env_content)
            
            # Secure the file
            os.chmod(output_file, 0o600)
            logger.info(f"📝 Generated {output_file} for {environment}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate env file: {e}")
            return False
    
    def export_for_container(self, 
                            categories: List[str] = None,
                            format: str = "docker") -> Dict[str, Any]:
        """
        🐳 Export secrets for container deployment
        
        Args:
            categories: Categories to export (default: all)
            format: Export format ('docker', 'k8s', 'compose')
        """
        try:
            secrets = self._load_secrets()
            
            # Filter categories
            if categories:
                filtered_secrets = {cat: secrets[cat] for cat in categories if cat in secrets}
            else:
                filtered_secrets = secrets
            
            # Flatten secrets
            flat_secrets = {}
            for category, cat_secrets in filtered_secrets.items():
                for key, secret_data in cat_secrets.items():
                    decrypted_value = self.cipher.decrypt(secret_data['value'].encode()).decode()
                    flat_secrets[key] = decrypted_value
            
            # Format for different deployment targets
            if format == "docker":
                return {
                    "env_vars": flat_secrets,
                    "dockerfile_env": [f"ENV {key}=${{{key}}}" for key in flat_secrets.keys()],
                    "docker_run": [f"-e {key}" for key in flat_secrets.keys()]
                }
            elif format == "k8s":
                return {
                    "configmap": flat_secrets,
                    "secret_manifest": self._generate_k8s_secret(flat_secrets)
                }
            elif format == "compose":
                return {
                    "environment": flat_secrets,
                    "env_file": ".env"
                }
            else:
                return {"secrets": flat_secrets}
                
        except Exception as e:
            logger.error(f"❌ Failed to export secrets: {e}")
            return {}
    
    def export_for_vercel(self, categories: List[str] = None) -> Dict[str, Any]:
        """
        ▲ Export secrets for Vercel deployment
        
        Returns both CLI commands and web interface data
        """
        try:
            secrets = self._load_secrets()
            
            # Filter categories
            if categories:
                filtered_secrets = {cat: secrets[cat] for cat in categories if cat in secrets}
            else:
                filtered_secrets = secrets
            
            # Flatten secrets
            flat_secrets = {}
            for category, cat_secrets in filtered_secrets.items():
                for key, secret_data in cat_secrets.items():
                    decrypted_value = self.cipher.decrypt(secret_data['value'].encode()).decode()
                    flat_secrets[key] = decrypted_value
            
            # Generate Vercel CLI commands
            cli_commands = []
            for key, value in flat_secrets.items():
                # Escape special characters for shell
                escaped_value = value.replace('"', '\\"').replace('$', '\\$')
                cli_commands.append(f'vercel env add {key} production <<< "{escaped_value}"')
            
            return {
                "secrets": flat_secrets,
                "cli_commands": cli_commands,
                "vercel_json_env": list(flat_secrets.keys()),
                "setup_script": self._generate_vercel_setup_script(cli_commands)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to export for Vercel: {e}")
            return {}
    
    def sync_from_env(self, env_file: str = ".env") -> bool:
        """
        🔄 Sync secrets from .env file to encrypted storage
        """
        try:
            env_path = Path(env_file)
            if not env_path.exists():
                logger.error(f"❌ Environment file not found: {env_file}")
                return False
            
            with open(env_path, 'r') as f:
                content = f.read()
            
            synced_count = 0
            for line in content.split('\n'):
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith('#') or not line or '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                # Skip template values
                if value.startswith('your-') or value.startswith('placeholder'):
                    continue
                
                # Determine category based on key name
                category = self._categorize_key(key)
                
                # Store secret
                if self.store_secret(key, value, category):
                    synced_count += 1
            
            logger.info(f"🔄 Synced {synced_count} secrets from {env_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to sync from env file: {e}")
            return False
    
    def validate_secrets(self, required_keys: List[str] = None) -> Dict[str, Any]:
        """
        ✅ Validate that required secrets are present and valid
        """
        try:
            validation_result = {
                "valid": True,
                "missing": [],
                "empty": [],
                "categories": {},
                "total_secrets": 0
            }
            
            secrets = self._load_secrets()
            
            # Count total secrets
            for category, cat_secrets in secrets.items():
                validation_result["categories"][category] = len(cat_secrets)
                validation_result["total_secrets"] += len(cat_secrets)
            
            # Check required keys
            if required_keys:
                for key in required_keys:
                    value = self.get_secret(key)
                    if value is None:
                        validation_result["missing"].append(key)
                        validation_result["valid"] = False
                    elif not value.strip():
                        validation_result["empty"].append(key)
                        validation_result["valid"] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Failed to validate secrets: {e}")
            return {"valid": False, "error": str(e)}
    
    def rotate_encryption_key(self, new_master_key: Optional[str] = None) -> bool:
        """
        🔄 Rotate encryption key (re-encrypt all secrets)
        """
        try:
            # Load current secrets with old key
            old_secrets = self._load_secrets()
            
            # Generate new key
            if new_master_key:
                new_key = self._derive_key_from_password(new_master_key)
            else:
                new_key = Fernet.generate_key()
            
            # Create new cipher
            new_cipher = Fernet(new_key)
            
            # Re-encrypt all secrets
            for category, cat_secrets in old_secrets.items():
                for key, secret_data in cat_secrets.items():
                    # Decrypt with old key
                    decrypted_value = self.cipher.decrypt(secret_data['value'].encode()).decode()
                    # Encrypt with new key
                    new_encrypted = new_cipher.encrypt(decrypted_value.encode()).decode()
                    secret_data['value'] = new_encrypted
                    secret_data['rotated'] = self._get_timestamp()
            
            # Save with new key
            self.cipher = new_cipher
            with open(self.key_file, 'wb') as f:
                f.write(new_key)
            os.chmod(self.key_file, 0o600)
            
            self._save_secrets(old_secrets)
            logger.info("🔄 Successfully rotated encryption key")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to rotate encryption key: {e}")
            return False
    
    def _load_secrets(self) -> Dict[str, Dict[str, Any]]:
        """Load and decrypt secrets from storage"""
        if not self.secrets_file.exists():
            return {}
        
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            if not encrypted_data:
                return {}
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
            
        except Exception as e:
            logger.error(f"❌ Failed to load secrets: {e}")
            return {}
    
    def _save_secrets(self, secrets: Dict[str, Dict[str, Any]]) -> bool:
        """Encrypt and save secrets to storage"""
        try:
            # Convert to JSON and encrypt
            json_data = json.dumps(secrets, indent=2)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            
            # Write to file
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Secure the file
            os.chmod(self.secrets_file, 0o600)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save secrets: {e}")
            return False
    
    def _categorize_key(self, key: str) -> str:
        """Automatically categorize a secret key"""
        key_lower = key.lower()
        
        if any(term in key_lower for term in ['grok', 'openai', 'perplexity', 'ai']):
            return 'ai'
        elif any(term in key_lower for term in ['binance', 'coinbase', 'trading']):
            return 'trading'
        elif any(term in key_lower for term in ['database', 'postgres', 'redis', 'db']):
            return 'database'
        elif any(term in key_lower for term in ['jwt', 'secret', 'encryption', 'auth']):
            return 'security'
        elif any(term in key_lower for term in ['sentry', 'otel', 'monitoring']):
            return 'monitoring'
        else:
            return 'general'
    
    def _generate_k8s_secret(self, secrets: Dict[str, str]) -> str:
        """Generate Kubernetes secret manifest"""
        encoded_secrets = {}
        for key, value in secrets.items():
            encoded_secrets[key] = base64.b64encode(value.encode()).decode()
        
        manifest = f"""apiVersion: v1
kind: Secret
metadata:
  name: cryptotrading-secrets
  namespace: default
type: Opaque
data:
"""
        for key, encoded_value in encoded_secrets.items():
            manifest += f"  {key}: {encoded_value}\n"
        
        return manifest
    
    def _generate_vercel_setup_script(self, cli_commands: List[str]) -> str:
        """Generate shell script for Vercel environment setup"""
        script = f"""#!/bin/bash
# 🔐 Vercel Environment Variables Setup Script
# Generated by SecretManager at {self._get_timestamp()}

echo "🚀 Setting up Vercel environment variables..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Install it first:"
    echo "npm i -g vercel"
    exit 1
fi

# Login to Vercel if needed
echo "🔑 Ensuring Vercel authentication..."
vercel whoami || vercel login

# Set environment variables
"""
        for cmd in cli_commands:
            script += f"{cmd}\n"
        
        script += """
echo "✅ Vercel environment variables configured!"
echo "🚀 Deploy with: vercel --prod"
"""
        return script
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

# 🛠️ Utility Functions

def setup_secret_manager(master_key: Optional[str] = None) -> SecretManager:
    """Initialize secret manager with optional master key"""
    return SecretManager(master_key=master_key)

def quick_setup_from_env() -> bool:
    """Quick setup: sync all secrets from .env file"""
    sm = setup_secret_manager()
    return sm.sync_from_env()

def validate_deployment_secrets() -> Dict[str, Any]:
    """Validate secrets for deployment readiness"""
    sm = setup_secret_manager()
    
    required_keys = [
        'GROK4_API_KEY',
        'DATABASE_URL',
        'REDIS_URL',
        'ENCRYPTION_KEY',
        'JWT_SECRET'
    ]
    
    return sm.validate_secrets(required_keys)

if __name__ == "__main__":
    # 🧪 Test the secret manager
    sm = setup_secret_manager()
    
    # Sync from .env
    print("🔄 Syncing secrets from .env...")
    sm.sync_from_env()
    
    # List secrets
    print("📋 Available secrets:")
    secrets_list = sm.list_secrets()
    for category, keys in secrets_list.items():
        print(f"  {category}: {keys}")
    
    # Validate
    print("✅ Validation:")
    validation = sm.validate_secrets()
    print(f"  Valid: {validation['valid']}")
    print(f"  Total secrets: {validation['total_secrets']}")
    
    print("🎉 Secret manager ready!")
