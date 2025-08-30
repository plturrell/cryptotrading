#!/usr/bin/env python3
"""
AWS Data Exchange CLI Tool
Interactive command-line interface for AWS Data Exchange data loading
"""

import sys
import os
import argparse
import json
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.append('.')

try:
    from src.cryptotrading.infrastructure.aws.data_exchange_service import AWSDataExchangeService
    AWS_AVAILABLE = True
except ImportError as e:
    print(f"AWS Data Exchange service not available: {e}")
    AWS_AVAILABLE = False

def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{title}")
    print("-" * len(title))

def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size"""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def check_aws_setup() -> bool:
    """Check AWS credentials and setup"""
    print_section("AWS Setup Check")
    
    # Check environment variables
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    bucket = os.getenv('AWS_DATA_EXCHANGE_BUCKET')
    
    print(f"AWS_ACCESS_KEY_ID: {'✓ Set' if aws_key else '✗ Not set'}")
    print(f"AWS_SECRET_ACCESS_KEY: {'✓ Set' if aws_secret else '✗ Not set'}")
    print(f"AWS_DEFAULT_REGION: {aws_region}")
    print(f"AWS_DATA_EXCHANGE_BUCKET: {bucket if bucket else '✗ Not set'}")
    
    if not aws_key or not aws_secret:
        print("\n⚠️  Missing AWS credentials!")
        print("Set environment variables:")
        print("  export AWS_ACCESS_KEY_ID=your_access_key")
        print("  export AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("  export AWS_DATA_EXCHANGE_BUCKET=your_bucket_name")
        return False
    
    # Test AWS connection
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"\n✓ AWS Connection successful")
        print(f"  Account: {identity.get('Account')}")
        print(f"  User: {identity.get('Arn', '').split('/')[-1]}")
        return True
    except Exception as e:
        print(f"\n✗ AWS Connection failed: {e}")
        return False

def discover_datasets(service: AWSDataExchangeService, dataset_type: str = "all") -> List[Dict]:
    """Discover available datasets"""
    print_section(f"Discovering {dataset_type.title()} Datasets")
    
    try:
        if dataset_type == "crypto":
            datasets = service.get_available_crypto_datasets()
        elif dataset_type == "economic":
            datasets = service.get_available_economic_datasets()
        else:
            # Get all financial datasets
            all_datasets = service.discover_financial_datasets()
            datasets = [{
                'dataset_id': ds.dataset_id,
                'name': ds.name,
                'description': ds.description,
                'provider': ds.provider,
                'last_updated': ds.last_updated.isoformat()
            } for ds in all_datasets]
        
        print(f"Found {len(datasets)} {dataset_type} datasets:\n")
        
        for i, dataset in enumerate(datasets, 1):
            print(f"{i:2d}. {dataset['name']}")
            print(f"     Provider: {dataset['provider']}")
            print(f"     ID: {dataset['dataset_id']}")
            if dataset.get('description'):
                desc = dataset['description'][:100] + "..." if len(dataset['description']) > 100 else dataset['description']
                print(f"     Description: {desc}")
            print()
        
        return datasets
        
    except Exception as e:
        print(f"Error discovering datasets: {e}")
        return []

def select_dataset(datasets: List[Dict]) -> Dict:
    """Interactive dataset selection"""
    if not datasets:
        print("No datasets available")
        return None
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets)}): ").strip()
            if choice.lower() in ['q', 'quit', 'exit']:
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                return datasets[idx]
            else:
                print(f"Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

def list_dataset_assets(service: AWSDataExchangeService, dataset_id: str):
    """List assets in a dataset"""
    print_section("Dataset Assets")
    
    try:
        assets = service.get_dataset_assets(dataset_id)
        
        if not assets:
            print("No assets found in this dataset")
            return []
        
        print(f"Found {len(assets)} assets:\n")
        
        for i, asset in enumerate(assets, 1):
            print(f"{i:2d}. {asset.name}")
            print(f"     Asset ID: {asset.asset_id}")
            print(f"     Format: {asset.file_format}")
            print(f"     Size: {format_size(asset.size_bytes)}")
            print(f"     Created: {asset.created_at}")
            print()
        
        return assets
        
    except Exception as e:
        print(f"Error listing assets: {e}")
        return []

def select_asset(assets: List) -> Any:
    """Interactive asset selection"""
    if not assets:
        return None
    
    while True:
        try:
            choice = input(f"\nSelect asset (1-{len(assets)}): ").strip()
            if choice.lower() in ['q', 'quit', 'exit']:
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(assets):
                return assets[idx]
            else:
                print(f"Please enter a number between 1 and {len(assets)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

def load_data_to_database(service: AWSDataExchangeService, dataset_id: str, asset_id: str, table_name: str = None):
    """Load dataset to database"""
    print_section("Loading Data to Database")
    
    if not table_name:
        table_name = f"aws_data_{dataset_id[:10]}_{asset_id[:10]}"
        # Sanitize table name
        table_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in table_name.lower())
    
    print(f"Dataset ID: {dataset_id}")
    print(f"Asset ID: {asset_id}")
    print(f"Target table: {table_name}")
    
    confirm = input("\nProceed with data loading? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Data loading cancelled")
        return
    
    try:
        print("\nStarting data loading process...")
        print("This may take several minutes for large datasets...")
        
        result = service.load_dataset_to_database(dataset_id, asset_id, table_name)
        
        if result['status'] == 'success':
            print("\n✅ Data loading completed successfully!")
            print(f"   Records loaded: {result['records_loaded']:,}")
            print(f"   Table name: {result['table_name']}")
            print(f"   Data shape: {result['data_shape']} (rows × columns)")
            print(f"   Columns: {', '.join(result['columns'][:5])}{'...' if len(result['columns']) > 5 else ''}")
        else:
            print(f"\n❌ Data loading failed: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Error during data loading: {e}")

def interactive_mode(service: AWSDataExchangeService):
    """Interactive mode for data discovery and loading"""
    print_header("AWS Data Exchange - Interactive Mode")
    
    while True:
        print("\nSelect an option:")
        print("1. Discover All Financial Datasets")
        print("2. Discover Crypto Datasets")
        print("3. Discover Economic Datasets")
        print("4. Load Dataset to Database")
        print("5. Check AWS Setup")
        print("q. Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'q' or choice == 'quit':
            break
        elif choice == '1':
            datasets = discover_datasets(service, "all")
        elif choice == '2':
            datasets = discover_datasets(service, "crypto")
        elif choice == '3':
            datasets = discover_datasets(service, "economic")
        elif choice == '4':
            # Full workflow: discover -> select -> load
            datasets = discover_datasets(service, "all")
            if datasets:
                selected_dataset = select_dataset(datasets)
                if selected_dataset:
                    assets = list_dataset_assets(service, selected_dataset['dataset_id'])
                    if assets:
                        selected_asset = select_asset(assets)
                        if selected_asset:
                            load_data_to_database(
                                service, 
                                selected_dataset['dataset_id'],
                                selected_asset.asset_id
                            )
        elif choice == '5':
            check_aws_setup()
        else:
            print("Invalid choice. Please try again.")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="AWS Data Exchange CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check-setup                    # Check AWS configuration
  %(prog)s --discover crypto               # Discover crypto datasets
  %(prog)s --discover economic             # Discover economic datasets
  %(prog)s --interactive                   # Interactive mode
  %(prog)s --load DATASET_ID ASSET_ID      # Load specific dataset/asset
        """
    )
    
    parser.add_argument('--check-setup', action='store_true', 
                       help='Check AWS credentials and setup')
    parser.add_argument('--discover', choices=['all', 'crypto', 'economic'],
                       help='Discover available datasets')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--load', nargs=2, metavar=('DATASET_ID', 'ASSET_ID'),
                       help='Load specific dataset and asset to database')
    parser.add_argument('--table-name', 
                       help='Custom table name for database loading')
    parser.add_argument('--json', action='store_true',
                       help='Output in JSON format')
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    print_header("AWS Data Exchange CLI Tool")
    
    # Check if AWS Data Exchange is available
    if not AWS_AVAILABLE:
        print("❌ AWS Data Exchange service not available")
        print("   Install required dependencies: pip install boto3")
        sys.exit(1)
    
    # Check setup first
    if args.check_setup or not check_aws_setup():
        if args.check_setup:
            sys.exit(0)
        else:
            print("\n⚠️  Please fix AWS setup before proceeding")
            sys.exit(1)
    
    # Initialize service
    try:
        service = AWSDataExchangeService()
        print("✅ AWS Data Exchange service initialized")
    except Exception as e:
        print(f"❌ Failed to initialize AWS Data Exchange service: {e}")
        sys.exit(1)
    
    # Handle different modes
    try:
        if args.interactive:
            interactive_mode(service)
        elif args.discover:
            datasets = discover_datasets(service, args.discover)
            if args.json:
                print(json.dumps(datasets, indent=2, default=str))
        elif args.load:
            dataset_id, asset_id = args.load
            load_data_to_database(service, dataset_id, asset_id, args.table_name)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()