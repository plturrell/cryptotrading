"""
AWS Data Exchange API Integration
REST API endpoints for AWS Data Exchange data loading
"""

from flask import Blueprint, jsonify, request
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cryptotrading.infrastructure.mcp.aws_data_exchange_mcp_agent import get_mcp_agent

logger = logging.getLogger(__name__)

# Create blueprint
aws_data_exchange_bp = Blueprint('aws_data_exchange', __name__, url_prefix='/api/odata/v4/AWSDataExchange')

# Initialize MCP agent
try:
    mcp_agent = get_mcp_agent()
    service_available = True
    logger.info("AWS Data Exchange MCP Agent initialized for API")
except Exception as e:
    logger.error(f"AWS Data Exchange MCP Agent not available: {e}")
    mcp_agent = None
    service_available = False

@aws_data_exchange_bp.route('/getAvailableDatasets', methods=['GET'])
async def get_available_datasets():
    """Get all available financial datasets from AWS Data Exchange with recommendations"""
    try:
        if not service_available:
            return jsonify({
                'error': 'AWS Data Exchange MCP Agent not available. Check AWS credentials and permissions.'
            }), 503
        
        # Get query parameters
        dataset_type = request.args.get('type', 'all')  # all, crypto, economic
        keywords = request.args.getlist('keywords')  # Support multiple keywords
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        # Use the enhanced discovery with recommendations
        result = await mcp_agent.execute_tool('discover_datasets_with_recommendations', {
            'dataset_type': dataset_type,
            'keywords': keywords,
            'force_refresh': force_refresh
        })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting available datasets: {e}")
        return jsonify({'error': str(e)}), 500

@aws_data_exchange_bp.route('/getDatasetAssets', methods=['POST'])
def get_dataset_assets():
    """Get assets (files) for a specific dataset"""
    try:
        if not service_available:
            return jsonify({'error': 'AWS Data Exchange service not available'}), 503
            
        data = request.json
        dataset_id = data.get('dataset_id')
        
        if not dataset_id:
            return jsonify({'error': 'dataset_id is required'}), 400
        
        assets = aws_dx_service.get_dataset_assets(dataset_id)
        
        asset_data = [{
            'asset_id': asset.asset_id,
            'name': asset.name,
            'file_format': asset.file_format,
            'size_bytes': asset.size_bytes,
            'size_mb': round(asset.size_bytes / (1024 * 1024), 2),
            'created_at': asset.created_at.isoformat()
        } for asset in assets]
        
        return jsonify({
            'status': 'success',
            'dataset_id': dataset_id,
            'asset_count': len(asset_data),
            'assets': asset_data
        })
        
    except Exception as e:
        logger.error(f"Error getting dataset assets: {e}")
        return jsonify({'error': str(e)}), 500

@aws_data_exchange_bp.route('/loadDatasetToDatabase', methods=['POST'])
def load_dataset_to_database():
    """Load a specific dataset/asset to the database"""
    try:
        if not service_available:
            return jsonify({'error': 'AWS Data Exchange service not available'}), 503
            
        data = request.json
        dataset_id = data.get('dataset_id')
        asset_id = data.get('asset_id') 
        table_name = data.get('table_name', f'aws_data_{dataset_id}_{asset_id}'[:50])
        
        if not dataset_id or not asset_id:
            return jsonify({'error': 'dataset_id and asset_id are required'}), 400
        
        # Sanitize table name
        table_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in table_name.lower())
        
        logger.info(f"Loading AWS Data Exchange dataset {dataset_id}, asset {asset_id} to table {table_name}")
        
        # Load data to database
        result = aws_dx_service.load_dataset_to_database(dataset_id, asset_id, table_name)
        
        if result['status'] == 'success':
            return jsonify({
                'status': 'completed',
                'dataset_id': dataset_id,
                'asset_id': asset_id,
                'table_name': table_name,
                'records_loaded': result['records_loaded'],
                'data_shape': result['data_shape'],
                'columns': result['columns'],
                'loaded_at': datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                'status': 'failed',
                'error': result['error']
            }), 500
            
    except Exception as e:
        logger.error(f"Error loading dataset to database: {e}")
        return jsonify({'error': str(e)}), 500

@aws_data_exchange_bp.route('/getJobStatus', methods=['POST'])
def get_job_status():
    """Get status of AWS Data Exchange export job"""
    try:
        if not service_available:
            return jsonify({'error': 'AWS Data Exchange service not available'}), 503
            
        data = request.json
        job_id = data.get('job_id')
        
        if not job_id:
            return jsonify({'error': 'job_id is required'}), 400
        
        status = aws_dx_service.get_job_status(job_id)
        
        return jsonify({
            'status': 'success',
            'job_status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({'error': str(e)}), 500

@aws_data_exchange_bp.route('/getServiceStatus', methods=['GET'])
def get_service_status():
    """Get AWS Data Exchange service status and configuration"""
    try:
        import boto3
        
        # Check AWS credentials
        try:
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            aws_available = True
            aws_account = identity.get('Account')
            aws_user_arn = identity.get('Arn')
        except Exception as aws_error:
            aws_available = False
            aws_account = None
            aws_user_arn = None
            aws_error_msg = str(aws_error)
        
        # Check required environment variables
        env_vars = {
            'AWS_ACCESS_KEY_ID': bool(os.getenv('AWS_ACCESS_KEY_ID')),
            'AWS_SECRET_ACCESS_KEY': bool(os.getenv('AWS_SECRET_ACCESS_KEY')),
            'AWS_DATA_EXCHANGE_BUCKET': os.getenv('AWS_DATA_EXCHANGE_BUCKET', 'not_set')
        }
        
        return jsonify({
            'service_available': service_available,
            'aws_credentials_valid': aws_available,
            'aws_account': aws_account,
            'aws_user_arn': aws_user_arn,
            'environment_variables': env_vars,
            'required_permissions': [
                'AWSDataExchangeFullAccess',
                'AmazonS3FullAccess (for temporary storage)'
            ],
            'setup_instructions': {
                'step_1': 'Set AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)',
                'step_2': 'Create S3 bucket for temporary data processing',
                'step_3': 'Set AWS_DATA_EXCHANGE_BUCKET environment variable',
                'step_4': 'Ensure AWS user has AWSDataExchangeFullAccess policy'
            },
            'error': aws_error_msg if not aws_available else None
        })
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        return jsonify({'error': str(e)}), 500

@aws_data_exchange_bp.route('/discoverCryptoData', methods=['GET'])
def discover_crypto_data():
    """Discover available cryptocurrency datasets"""
    try:
        if not service_available:
            return jsonify({'error': 'AWS Data Exchange service not available'}), 503
        
        crypto_datasets = aws_dx_service.get_available_crypto_datasets()
        
        return jsonify({
            'status': 'success',
            'crypto_datasets': crypto_datasets,
            'dataset_count': len(crypto_datasets),
            'discovered_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error discovering crypto data: {e}")
        return jsonify({'error': str(e)}), 500

@aws_data_exchange_bp.route('/discoverEconomicData', methods=['GET'])
def discover_economic_data():
    """Discover available economic datasets"""
    try:
        if not service_available:
            return jsonify({'error': 'AWS Data Exchange service not available'}), 503
        
        econ_datasets = aws_dx_service.get_available_economic_datasets()
        
        return jsonify({
            'status': 'success', 
            'economic_datasets': econ_datasets,
            'dataset_count': len(econ_datasets),
            'discovered_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error discovering economic data: {e}")
        return jsonify({'error': str(e)}), 500

@aws_data_exchange_bp.route('/processDatasetPipeline', methods=['POST'])
async def process_dataset_pipeline():
    """Complete pipeline: export, monitor, and process dataset"""
    try:
        if not service_available:
            return jsonify({'error': 'AWS Data Exchange MCP Agent not available'}), 503
            
        data = request.json
        dataset_id = data.get('dataset_id')
        asset_id = data.get('asset_id')
        auto_process = data.get('auto_process', True)
        timeout_minutes = data.get('timeout_minutes', 30)
        
        if not dataset_id or not asset_id:
            return jsonify({'error': 'dataset_id and asset_id are required'}), 400
        
        logger.info(f"Starting complete pipeline for dataset {dataset_id}, asset {asset_id}")
        
        # Execute complete pipeline via agent
        result = await mcp_agent.execute_tool('create_and_monitor_export_pipeline', {
            'dataset_id': dataset_id,
            'asset_id': asset_id,
            'auto_process': auto_process,
            'timeout_minutes': timeout_minutes
        })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in dataset pipeline: {e}")
        return jsonify({'error': str(e)}), 500

@aws_data_exchange_bp.route('/getAgentStatus', methods=['GET'])
async def get_agent_status():
    """Get comprehensive AWS Data Exchange agent status"""
    try:
        if not service_available:
            return jsonify({'error': 'AWS Data Exchange MCP Agent not available'}), 503
        
        result = await mcp_agent.execute_tool('get_comprehensive_agent_status', {})
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        return jsonify({'error': str(e)}), 500