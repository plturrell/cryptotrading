#!/usr/bin/env python3
"""
🔐 Secret Manager CLI
Command-line interface for managing secrets in local development and deployment
"""

import click
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.secret_manager import SecretManager

@click.group()
@click.option('--master-key', envvar='MASTER_KEY', help='Master encryption key')
@click.pass_context
def cli(ctx, master_key):
    """🔐 Crypto Trading Secret Manager CLI"""
    ctx.ensure_object(dict)
    ctx.obj['secret_manager'] = SecretManager(master_key=master_key)

@cli.command()
@click.argument('key')
@click.argument('value')
@click.option('--category', '-c', default='general', help='Secret category')
@click.pass_context
def store(ctx, key, value, category):
    """🔒 Store a secret"""
    sm = ctx.obj['secret_manager']
    
    if sm.store_secret(key, value, category):
        click.echo(f"✅ Stored secret: {key} in category: {category}")
    else:
        click.echo(f"❌ Failed to store secret: {key}")
        sys.exit(1)

@cli.command()
@click.argument('key')
@click.option('--category', '-c', help='Secret category to search in')
@click.option('--show', '-s', is_flag=True, help='Show the secret value')
@click.pass_context
def get(ctx, key, category, show):
    """🔓 Retrieve a secret"""
    sm = ctx.obj['secret_manager']
    
    value = sm.get_secret(key, category)
    if value:
        if show:
            click.echo(f"🔓 {key}: {value}")
        else:
            click.echo(f"🔓 {key}: {'*' * min(len(value), 10)}")
    else:
        click.echo(f"❌ Secret not found: {key}")
        sys.exit(1)

@cli.command()
@click.option('--category', '-c', help='List secrets in specific category')
@click.pass_context
def list(ctx, category):
    """📋 List all secrets"""
    sm = ctx.obj['secret_manager']
    
    secrets = sm.list_secrets(category)
    
    if not secrets:
        click.echo("📭 No secrets found")
        return
    
    for cat, keys in secrets.items():
        click.echo(f"\n📁 Category: {cat}")
        for key in keys:
            click.echo(f"  🔑 {key}")

@cli.command()
@click.argument('key')
@click.option('--category', '-c', help='Category to delete from')
@click.confirmation_option(prompt='Are you sure you want to delete this secret?')
@click.pass_context
def delete(ctx, key, category):
    """🗑️ Delete a secret"""
    sm = ctx.obj['secret_manager']
    
    if sm.delete_secret(key, category):
        click.echo(f"✅ Deleted secret: {key}")
    else:
        click.echo(f"❌ Failed to delete secret: {key}")
        sys.exit(1)

@cli.command()
@click.option('--template', '-t', default='.env.example', help='Template file')
@click.option('--output', '-o', default='.env', help='Output file')
@click.option('--environment', '-e', default='development', help='Target environment')
@click.pass_context
def generate_env(ctx, template, output, environment):
    """📝 Generate .env file from secrets"""
    sm = ctx.obj['secret_manager']
    
    if sm.generate_env_file(template, output, environment):
        click.echo(f"✅ Generated {output} for {environment}")
    else:
        click.echo(f"❌ Failed to generate {output}")
        sys.exit(1)

@cli.command()
@click.option('--env-file', '-f', default='.env', help='Environment file to sync from')
@click.pass_context
def sync(ctx, env_file):
    """🔄 Sync secrets from .env file"""
    sm = ctx.obj['secret_manager']
    
    if sm.sync_from_env(env_file):
        click.echo(f"✅ Synced secrets from {env_file}")
    else:
        click.echo(f"❌ Failed to sync from {env_file}")
        sys.exit(1)

@cli.command()
@click.option('--categories', '-c', multiple=True, help='Categories to export')
@click.option('--format', '-f', default='docker', 
              type=click.Choice(['docker', 'k8s', 'compose']), help='Export format')
@click.option('--output', '-o', help='Output file for manifest')
@click.pass_context
def export_container(ctx, categories, format, output):
    """🐳 Export secrets for container deployment"""
    sm = ctx.obj['secret_manager']
    
    export_data = sm.export_for_container(list(categories) if categories else None, format)
    
    if not export_data:
        click.echo("❌ Failed to export secrets")
        sys.exit(1)
    
    click.echo(f"🐳 Container export ({format}):")
    
    if format == 'docker':
        click.echo("\n📄 Dockerfile ENV statements:")
        for env_line in export_data['dockerfile_env']:
            click.echo(f"  {env_line}")
        
        click.echo("\n🏃 Docker run arguments:")
        docker_args = ' '.join(export_data['docker_run'])
        click.echo(f"  docker run {docker_args} your-image")
        
    elif format == 'k8s':
        click.echo("\n📄 Kubernetes Secret Manifest:")
        manifest = export_data['secret_manifest']
        if output:
            with open(output, 'w') as f:
                f.write(manifest)
            click.echo(f"✅ Saved to {output}")
        else:
            click.echo(manifest)
    
    elif format == 'compose':
        click.echo("\n📄 Docker Compose environment:")
        for key in export_data['environment'].keys():
            click.echo(f"  - {key}")
        click.echo("\n💡 Add to docker-compose.yml:")
        click.echo("  environment:")
        for key in export_data['environment'].keys():
            click.echo(f"    - {key}")

@cli.command()
@click.option('--categories', '-c', multiple=True, help='Categories to export')
@click.option('--generate-script', '-s', is_flag=True, help='Generate setup script')
@click.option('--output', '-o', default='vercel_setup.sh', help='Setup script output file')
@click.pass_context
def export_vercel(ctx, categories, generate_script, output):
    """▲ Export secrets for Vercel deployment"""
    sm = ctx.obj['secret_manager']
    
    export_data = sm.export_for_vercel(list(categories) if categories else None)
    
    if not export_data:
        click.echo("❌ Failed to export for Vercel")
        sys.exit(1)
    
    click.echo("▲ Vercel deployment secrets:")
    click.echo(f"📊 Total secrets: {len(export_data['secrets'])}")
    
    click.echo("\n🔧 CLI Commands:")
    for cmd in export_data['cli_commands']:
        click.echo(f"  {cmd}")
    
    click.echo("\n📋 vercel.json environment variables:")
    for key in export_data['vercel_json_env']:
        click.echo(f"  \"{key}\"")
    
    if generate_script:
        with open(output, 'w') as f:
            f.write(export_data['setup_script'])
        os.chmod(output, 0o755)
        click.echo(f"\n✅ Setup script saved to: {output}")
        click.echo(f"🚀 Run: ./{output}")

@cli.command()
@click.option('--required', '-r', multiple=True, help='Required secret keys')
@click.pass_context
def validate(ctx, required):
    """✅ Validate secrets"""
    sm = ctx.obj['secret_manager']
    
    result = sm.validate_secrets(list(required) if required else None)
    
    if result.get('error'):
        click.echo(f"❌ Validation error: {result['error']}")
        sys.exit(1)
    
    click.echo("🔍 Secret Validation Report:")
    click.echo(f"✅ Valid: {result['valid']}")
    click.echo(f"📊 Total secrets: {result['total_secrets']}")
    
    if result['categories']:
        click.echo("\n📁 Categories:")
        for category, count in result['categories'].items():
            click.echo(f"  {category}: {count} secrets")
    
    if result['missing']:
        click.echo(f"\n❌ Missing required secrets:")
        for key in result['missing']:
            click.echo(f"  - {key}")
    
    if result['empty']:
        click.echo(f"\n⚠️  Empty secrets:")
        for key in result['empty']:
            click.echo(f"  - {key}")
    
    if not result['valid']:
        sys.exit(1)

@cli.command()
@click.option('--new-key', help='New master key (leave empty for auto-generated)')
@click.confirmation_option(prompt='This will re-encrypt all secrets. Continue?')
@click.pass_context
def rotate_key(ctx, new_key):
    """🔄 Rotate encryption key"""
    sm = ctx.obj['secret_manager']
    
    if sm.rotate_encryption_key(new_key):
        click.echo("✅ Successfully rotated encryption key")
    else:
        click.echo("❌ Failed to rotate encryption key")
        sys.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    """📊 Show secret manager status"""
    sm = ctx.obj['secret_manager']
    
    # Get basic info
    secrets = sm.list_secrets()
    validation = sm.validate_secrets()
    
    click.echo("🔐 Secret Manager Status:")
    click.echo(f"📁 Config directory: {sm.config_dir}")
    click.echo(f"🔑 Encryption key: {'✅ Present' if sm.key_file.exists() else '❌ Missing'}")
    click.echo(f"💾 Secrets file: {'✅ Present' if sm.secrets_file.exists() else '❌ Missing'}")
    click.echo(f"📊 Total secrets: {validation['total_secrets']}")
    
    if secrets:
        click.echo("\n📁 Categories:")
        for category, keys in secrets.items():
            click.echo(f"  {category}: {len(keys)} secrets")

if __name__ == '__main__':
    cli()
