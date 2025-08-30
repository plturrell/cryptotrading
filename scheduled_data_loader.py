#!/usr/bin/env python3
"""
Scheduled Data Loader for Cryptotrading Platform
Automated data loading that can be run via cron or scheduled tasks

Example cron entries:
    # Load crypto data every hour
    0 * * * * /usr/bin/python3 /path/to/scheduled_data_loader.py --crypto hourly

    # Load economic data daily at 9 AM
    0 9 * * * /usr/bin/python3 /path/to/scheduled_data_loader.py --economic daily

    # Load DEX data every 4 hours
    0 */4 * * * /usr/bin/python3 /path/to/scheduled_data_loader.py --dex 4h

    # Load all data daily at midnight
    0 0 * * * /usr/bin/python3 /path/to/scheduled_data_loader.py --all daily
"""

import argparse
import requests
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

# Configuration
DEFAULT_SERVER = os.getenv('DATA_LOADER_SERVER', 'http://localhost:5001')
API_BASE = "/api/odata/v4/DataLoadingService"

# Setup logging
log_dir = Path.home() / '.cryptotrading' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'data_loader_{datetime.now().strftime("%Y%m%d")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScheduledDataLoader:
    def __init__(self, server_url: str = DEFAULT_SERVER):
        self.server_url = server_url
        self.api_base = f"{server_url}{API_BASE}"
        
        # Default configurations for different schedules
        self.configs = {
            'crypto': {
                'hourly': {
                    'symbols': ['BTC', 'ETH', 'SOL', 'BNB', 'XRP'],
                    'days': 1,
                    'interval': '15m'
                },
                'daily': {
                    'symbols': ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC', 'DOT', 'AVAX'],
                    'days': 7,
                    'interval': '1h'
                },
                'weekly': {
                    'symbols': ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC', 'DOT', 'AVAX'],
                    'days': 30,
                    'interval': '1d'
                }
            },
            'economic': {
                'daily': {
                    'series': ['DGS10', 'WALCL', 'M2SL', 'EFFR', 'CPIAUCSL', 'UNRATE'],
                    'days': 30
                },
                'weekly': {
                    'series': ['DGS10', 'WALCL', 'M2SL', 'EFFR', 'CPIAUCSL', 'UNRATE', 'T10Y2Y', 'RRPONTSYD'],
                    'days': 180
                },
                'monthly': {
                    'series': ['DGS10', 'WALCL', 'M2SL', 'EFFR', 'CPIAUCSL', 'UNRATE', 'T10Y2Y', 'RRPONTSYD'],
                    'days': 365
                }
            },
            'dex': {
                '4h': {
                    'networks': ['ethereum', 'bsc', 'polygon'],
                    'pools': 10
                },
                'daily': {
                    'networks': ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism'],
                    'pools': 20
                },
                'weekly': {
                    'networks': ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism', 'avalanche', 'base'],
                    'pools': 50
                }
            }
        }
    
    def check_server(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"{self.api_base}/getDataSourceStatus", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Server check failed: {e}")
            return False
    
    def load_crypto_data(self, schedule: str = 'daily') -> Optional[str]:
        """Load cryptocurrency data based on schedule"""
        config = self.configs['crypto'].get(schedule, self.configs['crypto']['daily'])
        
        logger.info(f"Loading crypto data with {schedule} schedule")
        logger.info(f"Symbols: {config['symbols']}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['days'])
        
        data = {
            "symbols": config['symbols'],
            "startDate": start_date.isoformat() + "Z",
            "endDate": end_date.isoformat() + "Z",
            "interval": config['interval']
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/loadYahooFinanceData",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get('jobId')
                logger.info(f"Crypto data loading started - Job ID: {job_id}")
                return job_id
            else:
                logger.error(f"Failed to start crypto data loading: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception loading crypto data: {e}")
            return None
    
    def load_economic_data(self, schedule: str = 'daily') -> Optional[str]:
        """Load economic data based on schedule"""
        config = self.configs['economic'].get(schedule, self.configs['economic']['daily'])
        
        logger.info(f"Loading economic data with {schedule} schedule")
        logger.info(f"Series: {config['series']}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['days'])
        
        data = {
            "series": config['series'],
            "startDate": start_date.isoformat() + "Z",
            "endDate": end_date.isoformat() + "Z"
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/loadFREDData",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get('jobId')
                logger.info(f"Economic data loading started - Job ID: {job_id}")
                return job_id
            else:
                logger.error(f"Failed to start economic data loading: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception loading economic data: {e}")
            return None
    
    def load_dex_data(self, schedule: str = 'daily') -> Optional[str]:
        """Load DEX data based on schedule"""
        config = self.configs['dex'].get(schedule, self.configs['dex']['daily'])
        
        logger.info(f"Loading DEX data with {schedule} schedule")
        logger.info(f"Networks: {config['networks']}")
        
        data = {
            "networks": config['networks'],
            "poolCount": config['pools'],
            "includeVolume": True,
            "includeLiquidity": True
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/loadGeckoTerminalData",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get('jobId')
                logger.info(f"DEX data loading started - Job ID: {job_id}")
                return job_id
            else:
                logger.error(f"Failed to start DEX data loading: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Exception loading DEX data: {e}")
            return None
    
    def load_all_data(self, schedule: str = 'daily') -> List[str]:
        """Load data from all sources"""
        logger.info(f"Loading all data sources with {schedule} schedule")
        
        # Determine appropriate schedules for each data type
        crypto_schedule = 'daily' if schedule in ['daily', 'hourly'] else schedule
        economic_schedule = 'daily' if schedule in ['daily', 'hourly'] else schedule
        dex_schedule = 'daily' if schedule in ['daily', '4h'] else schedule
        
        job_ids = []
        
        # Load crypto data
        crypto_job = self.load_crypto_data(crypto_schedule)
        if crypto_job:
            job_ids.append(crypto_job)
        
        # Load economic data
        economic_job = self.load_economic_data(economic_schedule)
        if economic_job:
            job_ids.append(economic_job)
        
        # Load DEX data
        dex_job = self.load_dex_data(dex_schedule)
        if dex_job:
            job_ids.append(dex_job)
        
        logger.info(f"All data loading started - {len(job_ids)} jobs created")
        return job_ids
    
    def monitor_jobs(self, job_ids: List[str], timeout: int = 300) -> bool:
        """Monitor jobs until completion or timeout"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                response = requests.get(f"{self.api_base}/getActiveJobs", timeout=10)
                
                if response.status_code == 200:
                    jobs = response.json()
                    
                    # Filter to our jobs
                    our_jobs = [job for job in jobs if job['jobId'] in job_ids]
                    
                    if not our_jobs:
                        logger.info("All jobs completed")
                        return True
                    
                    # Log status
                    for job in our_jobs:
                        logger.info(f"Job {job['jobId'][:8]}... - {job['source']}: {job['progress']}% ({job['status']})")
                    
                    # Check if any failed
                    failed_jobs = [job for job in our_jobs if job['status'] == 'failed']
                    if failed_jobs:
                        logger.error(f"{len(failed_jobs)} jobs failed")
                        return False
                    
                    import time
                    time.sleep(10)
                else:
                    logger.error(f"Failed to get job status: {response.status_code}")
                    return False
                    
            except Exception as e:
                logger.error(f"Exception monitoring jobs: {e}")
                return False
        
        logger.warning(f"Job monitoring timed out after {timeout} seconds")
        return False
    
    def send_notification(self, success: bool, job_count: int, message: str = ""):
        """Send notification about job completion (implement as needed)"""
        status = "SUCCESS" if success else "FAILURE"
        logger.info(f"Data loading {status} - {job_count} jobs - {message}")
        
        # Here you could add:
        # - Email notifications
        # - Slack/Discord webhooks
        # - Database status updates
        # - Monitoring system alerts

def main():
    parser = argparse.ArgumentParser(
        description='Scheduled Data Loader for Cryptotrading Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--server', default=DEFAULT_SERVER,
                       help=f'Server URL (default: {DEFAULT_SERVER})')
    
    # Data source options
    parser.add_argument('--crypto', choices=['hourly', 'daily', 'weekly'],
                       help='Load cryptocurrency data')
    parser.add_argument('--economic', choices=['daily', 'weekly', 'monthly'],
                       help='Load economic data')
    parser.add_argument('--dex', choices=['4h', 'daily', 'weekly'],
                       help='Load DEX data')
    parser.add_argument('--all', choices=['hourly', 'daily', 'weekly'],
                       help='Load all data sources')
    
    # Monitoring options
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor jobs until completion')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Job monitoring timeout in seconds (default: 300)')
    
    # Notification options
    parser.add_argument('--notify', action='store_true',
                       help='Send notifications on completion')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = ScheduledDataLoader(args.server)
    
    # Check server
    if not loader.check_server():
        logger.error(f"Cannot connect to server at {args.server}")
        sys.exit(1)
    
    logger.info(f"Connected to server at {args.server}")
    
    # Collect job IDs
    job_ids = []
    
    # Execute requested loads
    if args.all:
        job_ids = loader.load_all_data(args.all)
    else:
        if args.crypto:
            job_id = loader.load_crypto_data(args.crypto)
            if job_id:
                job_ids.append(job_id)
        
        if args.economic:
            job_id = loader.load_economic_data(args.economic)
            if job_id:
                job_ids.append(job_id)
        
        if args.dex:
            job_id = loader.load_dex_data(args.dex)
            if job_id:
                job_ids.append(job_id)
    
    if not job_ids:
        logger.warning("No data loading jobs were started")
        parser.print_help()
        sys.exit(1)
    
    # Monitor if requested
    success = True
    if args.monitor:
        success = loader.monitor_jobs(job_ids, args.timeout)
    
    # Send notifications if requested
    if args.notify:
        loader.send_notification(success, len(job_ids))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()