#!/usr/bin/env python3
"""
Cryptocurrency Data Manager for S3 Storage
Handles upload, download, and organization of crypto trading data
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

from .s3_storage_service import S3StorageService

logger = logging.getLogger(__name__)

class CryptoDataManager:
    """
    High-level data manager for cryptocurrency trading data in S3
    """
    
    def __init__(self, s3_service: S3StorageService):
        """
        Initialize crypto data manager
        
        Args:
            s3_service: Configured S3StorageService instance
        """
        self.s3_service = s3_service
        self.bucket_name = s3_service.bucket_name
    
    # Market Data Operations
    
    def save_ohlcv_data(self, symbol: str, timeframe: str, 
                       ohlcv_data: List[Dict], timestamp: datetime = None) -> bool:
        """
        Save OHLCV (Open, High, Low, Close, Volume) data to S3
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            timeframe: Data timeframe (e.g., '1h', '1d')
            ohlcv_data: List of OHLCV dictionaries
            timestamp: Data timestamp (defaults to now)
            
        Returns:
            bool: True if successful
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        # Organize by symbol, timeframe, and date
        date_str = timestamp.strftime('%Y/%m/%d')
        s3_key = f"market-data/ohlcv/{symbol}/{timeframe}/{date_str}/{timestamp.isoformat()}.json"
        
        # Add comprehensive metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data_type': 'ohlcv',
            'timestamp': timestamp.isoformat(),
            'record_count': str(len(ohlcv_data))
        }
        
        # Save data
        data_payload = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': timestamp.isoformat(),
            'data': ohlcv_data
        }
        
        return self.s3_service.upload_data(
            data=json.dumps(data_payload, indent=2, default=str),
            s3_key=s3_key,
            metadata=metadata,
            content_type='application/json'
        )
    
    def save_orderbook_data(self, symbol: str, orderbook: Dict, 
                           timestamp: datetime = None) -> bool:
        """
        Save orderbook data to S3
        
        Args:
            symbol: Trading pair symbol
            orderbook: Orderbook dictionary with bids/asks
            timestamp: Data timestamp
            
        Returns:
            bool: True if successful
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        # Organize by symbol and hour for efficient querying
        date_str = timestamp.strftime('%Y/%m/%d')
        hour_str = timestamp.strftime('%H')
        minute_str = timestamp.strftime('%M')
        
        s3_key = f"market-data/orderbook/{symbol}/{date_str}/{hour_str}/{minute_str}_{timestamp.isoformat()}.json"
        
        metadata = {
            'symbol': symbol,
            'data_type': 'orderbook',
            'timestamp': timestamp.isoformat(),
            'bid_count': str(len(orderbook.get('bids', []))),
            'ask_count': str(len(orderbook.get('asks', [])))
        }
        
        data_payload = {
            'symbol': symbol,
            'timestamp': timestamp.isoformat(),
            'orderbook': orderbook
        }
        
        return self.s3_service.upload_data(
            data=json.dumps(data_payload, indent=2),
            s3_key=s3_key,
            metadata=metadata,
            content_type='application/json'
        )
    
    def save_trade_data(self, symbol: str, trades: List[Dict], 
                       timestamp: datetime = None) -> bool:
        """
        Save trade tick data to S3
        
        Args:
            symbol: Trading pair symbol
            trades: List of trade dictionaries
            timestamp: Data timestamp
            
        Returns:
            bool: True if successful
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        date_str = timestamp.strftime('%Y/%m/%d')
        hour_str = timestamp.strftime('%H')
        
        s3_key = f"market-data/trades/{symbol}/{date_str}/{hour_str}/{timestamp.isoformat()}.json"
        
        metadata = {
            'symbol': symbol,
            'data_type': 'trades',
            'timestamp': timestamp.isoformat(),
            'trade_count': str(len(trades))
        }
        
        data_payload = {
            'symbol': symbol,
            'timestamp': timestamp.isoformat(),
            'trades': trades
        }
        
        return self.s3_service.upload_data(
            data=json.dumps(data_payload, indent=2),
            s3_key=s3_key,
            metadata=metadata,
            content_type='application/json'
        )
    
    # User Data Operations
    
    def save_user_portfolio(self, user_id: str, portfolio: Dict, 
                           timestamp: datetime = None) -> bool:
        """
        Save user portfolio data to S3
        
        Args:
            user_id: User identifier
            portfolio: Portfolio dictionary
            timestamp: Data timestamp
            
        Returns:
            bool: True if successful
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        date_str = timestamp.strftime('%Y/%m/%d')
        s3_key = f"user-data/{user_id}/portfolio/{date_str}/{timestamp.isoformat()}.json"
        
        metadata = {
            'user_id': user_id,
            'data_type': 'portfolio',
            'timestamp': timestamp.isoformat(),
            'asset_count': str(len(portfolio.get('assets', [])))
        }
        
        data_payload = {
            'user_id': user_id,
            'timestamp': timestamp.isoformat(),
            'portfolio': portfolio
        }
        
        return self.s3_service.upload_data(
            data=json.dumps(data_payload, indent=2, default=str),
            s3_key=s3_key,
            metadata=metadata,
            content_type='application/json'
        )
    
    def save_user_trades(self, user_id: str, trades: List[Dict], 
                        timestamp: datetime = None) -> bool:
        """
        Save user trade history to S3
        
        Args:
            user_id: User identifier
            trades: List of user trades
            timestamp: Data timestamp
            
        Returns:
            bool: True if successful
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        date_str = timestamp.strftime('%Y/%m')
        s3_key = f"user-data/{user_id}/trades/{date_str}/{timestamp.isoformat()}.json"
        
        metadata = {
            'user_id': user_id,
            'data_type': 'trades',
            'timestamp': timestamp.isoformat(),
            'trade_count': str(len(trades))
        }
        
        data_payload = {
            'user_id': user_id,
            'timestamp': timestamp.isoformat(),
            'trades': trades
        }
        
        return self.s3_service.upload_data(
            data=json.dumps(data_payload, indent=2, default=str),
            s3_key=s3_key,
            metadata=metadata,
            content_type='application/json'
        )
    
    # Analytics Data Operations
    
    def save_analytics_report(self, report_type: str, analysis_data: Dict, 
                             timestamp: datetime = None) -> bool:
        """
        Save analytics report to S3
        
        Args:
            report_type: Type of analysis (e.g., 'sentiment', 'technical')
            analysis_data: Analysis results dictionary
            timestamp: Report timestamp
            
        Returns:
            bool: True if successful
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        date_str = timestamp.strftime('%Y/%m/%d')
        s3_key = f"analytics/{report_type}/{date_str}/{timestamp.isoformat()}.json"
        
        metadata = {
            'report_type': report_type,
            'data_type': 'analytics',
            'timestamp': timestamp.isoformat()
        }
        
        data_payload = {
            'report_type': report_type,
            'timestamp': timestamp.isoformat(),
            'analysis': analysis_data
        }
        
        return self.s3_service.upload_data(
            data=json.dumps(data_payload, indent=2, default=str),
            s3_key=s3_key,
            metadata=metadata,
            content_type='application/json'
        )
    
    # Backup Operations
    
    def backup_database_table(self, table_name: str, data: List[Dict], 
                             timestamp: datetime = None) -> bool:
        """
        Backup database table data to S3
        
        Args:
            table_name: Name of database table
            data: List of records to backup
            timestamp: Backup timestamp
            
        Returns:
            bool: True if successful
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        date_str = timestamp.strftime('%Y/%m/%d')
        s3_key = f"backups/database/{table_name}/{date_str}/{timestamp.isoformat()}.json"
        
        metadata = {
            'table_name': table_name,
            'data_type': 'database_backup',
            'timestamp': timestamp.isoformat(),
            'record_count': str(len(data))
        }
        
        backup_payload = {
            'table_name': table_name,
            'timestamp': timestamp.isoformat(),
            'records': data
        }
        
        return self.s3_service.upload_data(
            data=json.dumps(backup_payload, indent=2, default=str),
            s3_key=s3_key,
            metadata=metadata,
            content_type='application/json'
        )
    
    def upload_log_file(self, log_type: str, log_file_path: str, 
                       timestamp: datetime = None) -> bool:
        """
        Upload log file to S3
        
        Args:
            log_type: Type of log (e.g., 'application', 'trading', 'error')
            log_file_path: Path to log file
            timestamp: Log timestamp
            
        Returns:
            bool: True if successful
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        date_str = timestamp.strftime('%Y/%m/%d')
        log_filename = Path(log_file_path).name
        s3_key = f"logs/{log_type}/{date_str}/{timestamp.isoformat()}_{log_filename}"
        
        metadata = {
            'log_type': log_type,
            'data_type': 'log_file',
            'timestamp': timestamp.isoformat(),
            'filename': log_filename
        }
        
        return self.s3_service.upload_file(
            local_file_path=log_file_path,
            s3_key=s3_key,
            metadata=metadata,
            content_type='text/plain'
        )
    
    # Data Retrieval Operations
    
    def get_latest_market_data(self, symbol: str, data_type: str, 
                              days_back: int = 1) -> Optional[Dict]:
        """
        Retrieve latest market data for a symbol
        
        Args:
            symbol: Trading pair symbol
            data_type: Type of data (ohlcv, orderbook, trades)
            days_back: Number of days to search back
            
        Returns:
            Dict: Latest data or None if not found
        """
        try:
            # Search recent dates for data
            for i in range(days_back):
                search_date = datetime.utcnow() - timedelta(days=i)
                date_str = search_date.strftime('%Y/%m/%d')
                prefix = f"market-data/{data_type}/{symbol}/{date_str}/"
                
                objects = self.s3_service.list_objects(prefix=prefix, max_keys=1)
                if objects:
                    # Get the latest object
                    latest_key = max(objects, key=lambda x: x['last_modified'])['key']
                    data = self.s3_service.get_object_data(latest_key)
                    
                    if data:
                        return json.loads(data.decode('utf-8'))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest market data: {e}")
            return None
    
    def list_user_data_files(self, user_id: str, data_type: str = None) -> List[str]:
        """
        List available user data files in S3
        
        Args:
            user_id: User identifier
            data_type: Optional data type filter
            
        Returns:
            List of S3 keys for user data files
        """
        try:
            prefix = f"user-data/{user_id}/"
            if data_type:
                prefix += f"{data_type}/"
            
            objects = self.s3_service.list_objects(prefix=prefix)
            return [obj['key'] for obj in objects]
            
        except Exception as e:
            logger.error(f"Failed to list user data files: {e}")
            return []
    
    # Utility Methods
    
    def get_storage_stats(self) -> Dict:
        """
        Get storage usage statistics
        
        Returns:
            Dict: Storage statistics
        """
        try:
            stats = {
                'total_objects': 0,
                'total_size_bytes': 0,
                'data_types': {},
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Get all objects (limited sample for performance)
            objects = self.s3_service.list_objects(max_keys=10000)
            
            for obj in objects:
                stats['total_objects'] += 1
                stats['total_size_bytes'] += obj['size']
                
                # Extract data type from key
                key_parts = obj['key'].split('/')
                if len(key_parts) > 1:
                    data_type = key_parts[0]
                    if data_type not in stats['data_types']:
                        stats['data_types'][data_type] = {'count': 0, 'size_bytes': 0}
                    stats['data_types'][data_type]['count'] += 1
                    stats['data_types'][data_type]['size_bytes'] += obj['size']
            
            # Convert bytes to human readable format
            stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
            
            for data_type in stats['data_types']:
                size_bytes = stats['data_types'][data_type]['size_bytes']
                stats['data_types'][data_type]['size_mb'] = round(size_bytes / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}