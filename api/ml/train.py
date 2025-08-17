"""
Vercel API route for model training
Allows triggering model training and checking status
"""

import os
import sys
import json
import hashlib
import pickle
from datetime import datetime
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cryptotrading.core.ml.model_server import ModelRegistry
from cryptotrading.core.ml.models import CryptoPricePredictor
from cryptotrading.core.ml.model_storage import VercelBlobModelStorage

# Global training status tracker
training_status = {}


async def train_model_async(model_id: str, request_id: str):
    """Train a model asynchronously"""
    try:
        # Update status
        training_status[request_id] = {
            'status': 'training',
            'progress': 10,
            'message': 'Fetching training data...'
        }
        
        # Initialize components
        predictor = CryptoPricePredictor()
        registry = ModelRegistry()
        
        # Train the model with real data
        training_status[request_id]['progress'] = 30
        training_status[request_id]['message'] = 'Training model...'
        
        metrics = predictor.train(model_id)
        
        # Get the trained model
        model = predictor.get_model(model_id)
        
        if not model:
            raise ValueError(f"Training failed for {model_id}")
        
        # Serialize model
        training_status[request_id]['progress'] = 70
        training_status[request_id]['message'] = 'Saving model...'
        
        model_data = pickle.dumps(model)
        
        # Prepare metadata
        metadata = {
            'model_id': model_id,
            'metrics': metrics,
            'features': predictor.feature_columns,
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(predictor.training_data) if hasattr(predictor, 'training_data') else 0,
            'model_type': 'ensemble',
            'status': 'active'
        }
        
        # Register model
        training_status[request_id]['progress'] = 90
        training_status[request_id]['message'] = 'Registering model...'
        
        model_version = await registry.register_model(model_id, model_data, metadata)
        
        # Update status
        training_status[request_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Model trained successfully',
            'model_version': model_version.version,
            'metrics': metrics,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        training_status[request_id] = {
            'status': 'failed',
            'progress': 0,
            'message': f'Training failed: {str(e)}',
            'error': str(e),
            'failed_at': datetime.now().isoformat()
        }


def handler(request):
    """
    Vercel serverless function handler for model training
    POST /api/ml/train - Start training
    GET /api/ml/train/status?request_id=xxx - Check status
    """
    try:
        if request.method == 'POST':
            # Start training
            body = json.loads(request.body)
            model_id = body.get('model_id', body.get('symbol', 'BTC'))
            
            # Generate request ID
            request_id = hashlib.md5(
                f"{model_id}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            
            # Initialize status
            training_status[request_id] = {
                'status': 'queued',
                'progress': 0,
                'message': 'Training request queued',
                'model_id': model_id,
                'created_at': datetime.now().isoformat()
            }
            
            # Start async training
            asyncio.create_task(train_model_async(model_id, request_id))
            
            return {
                'statusCode': 202,
                'headers': {
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'request_id': request_id,
                    'model_id': model_id,
                    'status': 'queued',
                    'message': 'Training started',
                    'check_status_url': f'/api/ml/train/status?request_id={request_id}'
                })
            }
        
        elif request.method == 'GET':
            # Check training status
            request_id = request.args.get('request_id')
            
            if not request_id:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'request_id required'})
                }
            
            status = training_status.get(request_id)
            
            if not status:
                return {
                    'statusCode': 404,
                    'body': json.dumps({'error': 'Training request not found'})
                }
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json'
                },
                'body': json.dumps(status)
            }
        
        else:
            return {
                'statusCode': 405,
                'body': json.dumps({'error': 'Method not allowed'})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }


# For testing locally
if __name__ == "__main__":
    import asyncio
    
    class MockRequest:
        def __init__(self, method='POST', body=None):
            self.method = method
            self.body = body or json.dumps({'model_id': 'BTC'})
            self.args = {}
    
    # Test training start
    print("Starting training...")
    result = handler(MockRequest())
    print(json.dumps(result, indent=2))
    
    # Extract request_id
    response_body = json.loads(result['body'])
    request_id = response_body['request_id']
    
    # Wait a bit
    print("\nWaiting for training to progress...")
    asyncio.run(asyncio.sleep(2))
    
    # Check status
    print("\nChecking status...")
    status_request = MockRequest(method='GET')
    status_request.args = {'request_id': request_id}
    status_result = handler(status_request)
    print(json.dumps(status_result, indent=2))