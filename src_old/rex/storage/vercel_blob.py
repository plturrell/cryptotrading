"""
Vercel Blob storage integration for rex.com
Store trading data, analysis results, and model artifacts
"""

import requests
import json
from typing import Dict, Optional, Any, List, Union
from datetime import datetime
import os
import base64
from pathlib import Path

class VercelBlobClient:
    def __init__(self, token: Optional[str] = None):
        """Initialize Vercel Blob client"""
        self.token = token or os.getenv('BLOB_READ_WRITE_TOKEN')
        if not self.token:
            raise ValueError("Vercel Blob token required. Set BLOB_READ_WRITE_TOKEN env var.")
        
        self.base_url = "https://blob.vercel-storage.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        })
    
    def put(self, pathname: str, data: Union[str, bytes, Dict], 
            access: str = 'public', content_type: Optional[str] = None) -> Dict[str, str]:
        """
        Upload data to Vercel Blob storage
        
        Args:
            pathname: Path for the blob (e.g., 'trading/signals/btc_2024.json')
            data: Data to upload (string, bytes, or dict that will be JSON serialized)
            access: 'public' or 'private'
            content_type: MIME type (auto-detected if not provided)
        
        Returns:
            Dict with 'url' and other metadata
        """
        try:
            # Prepare data
            if isinstance(data, dict):
                upload_data = json.dumps(data, indent=2)
                content_type = content_type or 'application/json'
            elif isinstance(data, bytes):
                upload_data = base64.b64encode(data).decode('utf-8')
                content_type = content_type or 'application/octet-stream'
            else:
                upload_data = str(data)
                content_type = content_type or 'text/plain'
            
            # Prepare request
            payload = {
                'pathname': pathname,
                'data': upload_data,
                'access': access,
                'contentType': content_type
            }
            
            response = self.session.post(
                f"{self.base_url}/api/put",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                'url': result.get('url'),
                'pathname': pathname,
                'contentType': content_type,
                'size': len(upload_data),
                'uploadedAt': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Vercel Blob upload error: {e}")
            return {'error': str(e)}
    
    def put_json(self, pathname: str, data: Dict, access: str = 'public') -> Dict[str, str]:
        """Convenience method for uploading JSON data"""
        return self.put(pathname, data, access, 'application/json')
    
    def list(self, prefix: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List blobs in storage
        
        Args:
            prefix: Filter by path prefix (e.g., 'trading/signals/')
            limit: Maximum number of results
        
        Returns:
            List of blob metadata
        """
        try:
            params = {'limit': limit}
            if prefix:
                params['prefix'] = prefix
            
            response = self.session.get(
                f"{self.base_url}/api/list",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('blobs', [])
            
        except Exception as e:
            print(f"Vercel Blob list error: {e}")
            return []
    
    def delete(self, urls: Union[str, List[str]]) -> bool:
        """
        Delete one or more blobs
        
        Args:
            urls: Single URL or list of URLs to delete
        
        Returns:
            True if successful
        """
        try:
            if isinstance(urls, str):
                urls = [urls]
            
            payload = {'urls': urls}
            
            response = self.session.post(
                f"{self.base_url}/api/delete",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            print(f"Vercel Blob delete error: {e}")
            return False
    
    def store_trading_signal(self, symbol: str, signal_data: Dict) -> Dict[str, str]:
        """Store trading signal in blob storage"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pathname = f"signals/{symbol}/{timestamp}.json"
        
        # Add metadata
        signal_data['stored_at'] = datetime.now().isoformat()
        signal_data['symbol'] = symbol
        
        return self.put_json(pathname, signal_data)
    
    def store_market_analysis(self, analysis_data: Dict) -> Dict[str, str]:
        """Store market analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pathname = f"analysis/market/{timestamp}.json"
        
        return self.put_json(pathname, analysis_data)
    
    def store_model_checkpoint(self, model_name: str, model_data: bytes) -> Dict[str, str]:
        """Store ML model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pathname = f"models/{model_name}/{timestamp}.pkl"
        
        return self.put(pathname, model_data, 'private', 'application/octet-stream')
    
    def store_portfolio_snapshot(self, user_id: str, portfolio_data: Dict) -> Dict[str, str]:
        """Store portfolio snapshot"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pathname = f"portfolios/{user_id}/{timestamp}.json"
        
        return self.put_json(pathname, portfolio_data, 'private')
    
    def store_backtest_results(self, strategy_name: str, results: Dict) -> Dict[str, str]:
        """Store backtest results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pathname = f"backtests/{strategy_name}/{timestamp}.json"
        
        return self.put_json(pathname, results)
    
    def get_latest_signals(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest trading signals for a symbol"""
        prefix = f"signals/{symbol}/"
        blobs = self.list(prefix, limit)
        
        # Sort by creation time (newest first)
        blobs.sort(key=lambda x: x.get('uploadedAt', ''), reverse=True)
        
        return blobs[:limit]
    
    def get_analysis_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get market analysis history"""
        prefix = "analysis/market/"
        blobs = self.list(prefix, limit=1000)
        
        # Filter by date
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_blobs = []
        
        for blob in blobs:
            try:
                # Extract timestamp from pathname
                filename = blob['pathname'].split('/')[-1].replace('.json', '')
                blob_time = datetime.strptime(filename, '%Y%m%d_%H%M%S').timestamp()
                
                if blob_time >= cutoff:
                    recent_blobs.append(blob)
            except (ValueError, KeyError, IndexError) as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not parse timestamp from blob {blob.get('pathname', 'unknown')}: {e}")
                continue
        
        return recent_blobs
    
    def cleanup_old_data(self, prefix: str, days_to_keep: int = 30) -> int:
        """Clean up old data from blob storage"""
        blobs = self.list(prefix, limit=1000)
        cutoff = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        urls_to_delete = []
        
        for blob in blobs:
            try:
                # Check age
                uploaded_at = blob.get('uploadedAt', '')
                if uploaded_at:
                    blob_time = datetime.fromisoformat(uploaded_at.replace('Z', '+00:00')).timestamp()
                    if blob_time < cutoff:
                        urls_to_delete.append(blob['url'])
            except (ValueError, KeyError) as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not parse upload date from blob {blob.get('pathname', 'unknown')}: {e}")
                continue
        
        if urls_to_delete:
            success = self.delete(urls_to_delete)
            return len(urls_to_delete) if success else 0
        
        return 0


# Convenience functions for direct usage
def put_blob(pathname: str, data: Union[str, bytes, Dict], 
             access: str = 'public', token: Optional[str] = None) -> Dict[str, str]:
    """
    Quick function to upload to Vercel Blob
    
    Example:
        url = put_blob('articles/analysis.txt', 'BTC is bullish!', access='public')
    """
    client = VercelBlobClient(token)
    return client.put(pathname, data, access)


def put_json_blob(pathname: str, data: Dict, 
                  access: str = 'public', token: Optional[str] = None) -> Dict[str, str]:
    """
    Quick function to upload JSON to Vercel Blob
    
    Example:
        url = put_json_blob('signals/btc.json', {'signal': 'BUY', 'confidence': 85})
    """
    client = VercelBlobClient(token)
    return client.put_json(pathname, data, access)