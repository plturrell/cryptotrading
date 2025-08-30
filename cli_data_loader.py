#!/usr/bin/env python3
"""
CLI Data Loader for Cryptotrading Platform
Trigger data loads from command line for Yahoo Finance, FRED, and GeckoTerminal

Usage:
    python cli_data_loader.py yahoo --symbols BTC ETH SOL --days 30
    python cli_data_loader.py fred --series DGS10 WALCL M2SL --days 365
    python cli_data_loader.py gecko --networks ethereum polygon --pools 20
    python cli_data_loader.py all --symbols BTC ETH --series DGS10 --networks ethereum
    python cli_data_loader.py status
    python cli_data_loader.py jobs
    python cli_data_loader.py cancel <job_id>
"""

import argparse
import requests
import json
import sys
from datetime import datetime, timedelta
from typing import List, Optional
from tabulate import tabulate
import time

# Configuration
DEFAULT_SERVER = "http://localhost:5001"
API_BASE = "/api/odata/v4/DataLoadingService"

class DataLoaderCLI:
    def __init__(self, server_url: str = DEFAULT_SERVER):
        self.server_url = server_url
        self.api_base = f"{server_url}{API_BASE}"
        
    def check_server(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"{self.api_base}/getDataSourceStatus", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def load_yahoo_data(self, symbols: List[str], days: int = 30, interval: str = "1d"):
        """Load Yahoo Finance data"""
        print(f"\n📈 Loading Yahoo Finance data...")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Period: Last {days} days")
        print(f"   Interval: {interval}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = {
            "symbols": symbols,
            "startDate": start_date.isoformat() + "Z",
            "endDate": end_date.isoformat() + "Z",
            "interval": interval
        }
        
        response = requests.post(
            f"{self.api_base}/loadYahooFinanceData",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Yahoo data loading started")
            print(f"   Job ID: {result.get('jobId')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Records queued: {result.get('recordsQueued', len(symbols))}")
            return result.get('jobId')
        else:
            print(f"❌ Failed to start Yahoo data loading: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    
    def load_fred_data(self, series: List[str], days: int = 365):
        """Load FRED economic data"""
        print(f"\n📊 Loading FRED economic data...")
        print(f"   Series: {', '.join(series)}")
        print(f"   Period: Last {days} days")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = {
            "series": series,
            "startDate": start_date.isoformat() + "Z",
            "endDate": end_date.isoformat() + "Z"
        }
        
        response = requests.post(
            f"{self.api_base}/loadFREDData",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ FRED data loading started")
            print(f"   Job ID: {result.get('jobId')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Records queued: {result.get('recordsQueued', len(series))}")
            return result.get('jobId')
        else:
            print(f"❌ Failed to start FRED data loading: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    
    def load_gecko_data(self, networks: List[str], pool_count: int = 20, 
                        include_volume: bool = True, include_liquidity: bool = True):
        """Load GeckoTerminal DEX data"""
        print(f"\n🦎 Loading GeckoTerminal DEX data...")
        print(f"   Networks: {', '.join(networks)}")
        print(f"   Pools per network: {pool_count}")
        print(f"   Include volume: {include_volume}")
        print(f"   Include liquidity: {include_liquidity}")
        
        data = {
            "networks": networks,
            "poolCount": pool_count,
            "includeVolume": include_volume,
            "includeLiquidity": include_liquidity
        }
        
        response = requests.post(
            f"{self.api_base}/loadGeckoTerminalData",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ GeckoTerminal data loading started")
            print(f"   Job ID: {result.get('jobId')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Records queued: {result.get('recordsQueued', len(networks) * pool_count)}")
            return result.get('jobId')
        else:
            print(f"❌ Failed to start GeckoTerminal data loading: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    
    def load_all_data(self, symbols: Optional[List[str]] = None, 
                     series: Optional[List[str]] = None,
                     networks: Optional[List[str]] = None,
                     days: int = 30):
        """Load data from all sources"""
        print(f"\n🚀 Loading data from all sources...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Default values if not provided
        if not symbols:
            symbols = ["BTC", "ETH", "SOL"]
        if not series:
            series = ["DGS10", "WALCL", "M2SL"]
        if not networks:
            networks = ["ethereum", "polygon", "bsc"]
        
        print(f"   Crypto symbols: {', '.join(symbols)}")
        print(f"   FRED series: {', '.join(series)}")
        print(f"   DEX networks: {', '.join(networks)}")
        print(f"   Period: Last {days} days")
        
        data = {
            "cryptoSymbols": symbols,
            "fredSeries": series,
            "dexNetworks": networks,
            "startDate": start_date.isoformat() + "Z",
            "endDate": end_date.isoformat() + "Z"
        }
        
        response = requests.post(
            f"{self.api_base}/loadAllMarketData",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ All data sources loading started")
            print(f"   Total jobs: {result.get('totalJobs')}")
            print(f"   Job IDs: {', '.join([jid[:8] + '...' for jid in result.get('jobIds', [])])}")
            return result.get('jobIds', [])
        else:
            print(f"❌ Failed to start bulk data loading: {response.status_code}")
            print(f"   Error: {response.text}")
            return []
    
    def get_data_source_status(self):
        """Get status of all data sources"""
        print(f"\n📊 Data Source Status")
        print("=" * 80)
        
        response = requests.get(f"{self.api_base}/getDataSourceStatus")
        
        if response.status_code == 200:
            sources = response.json()
            
            table_data = []
            for source in sources:
                table_data.append([
                    source['source'],
                    "✅" if source['isAvailable'] else "❌",
                    source['apiStatus'],
                    f"{source['recordCount']:,}",
                    source['rateLimit'],
                    source['lastSync'][:19] if source['lastSync'] else "Never"
                ])
            
            headers = ["Source", "Available", "API Status", "Records", "Rate Limit", "Last Sync"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            return True
        else:
            print(f"❌ Failed to get data source status: {response.status_code}")
            return False
    
    def get_active_jobs(self, watch: bool = False):
        """Get all active loading jobs"""
        if watch:
            print(f"\n📋 Monitoring Active Jobs (Press Ctrl+C to stop)")
            print("=" * 80)
            
            try:
                while True:
                    response = requests.get(f"{self.api_base}/getActiveJobs")
                    
                    if response.status_code == 200:
                        jobs = response.json()
                        
                        # Clear screen for update
                        print("\033[2J\033[H")  # Clear screen and move cursor to top
                        print(f"📋 Active Jobs - {datetime.now().strftime('%H:%M:%S')}")
                        print("=" * 80)
                        
                        if jobs:
                            table_data = []
                            for job in jobs:
                                progress_bar = self._create_progress_bar(job['progress'])
                                table_data.append([
                                    job['jobId'][:8] + '...',
                                    job['source'],
                                    job['status'],
                                    progress_bar,
                                    job['startTime'][:19] if job['startTime'] else ""
                                ])
                            
                            headers = ["Job ID", "Source", "Status", "Progress", "Start Time"]
                            print(tabulate(table_data, headers=headers, tablefmt="grid"))
                        else:
                            print("No active jobs")
                        
                        # Check if all jobs are complete
                        if all(job['status'] in ['completed', 'failed', 'cancelled'] for job in jobs):
                            print("\n✅ All jobs completed!")
                            break
                        
                        time.sleep(2)
                    else:
                        print(f"❌ Failed to get active jobs: {response.status_code}")
                        break
                        
            except KeyboardInterrupt:
                print("\n\nStopped monitoring")
        else:
            print(f"\n📋 Active Jobs")
            print("=" * 80)
            
            response = requests.get(f"{self.api_base}/getActiveJobs")
            
            if response.status_code == 200:
                jobs = response.json()
                
                if jobs:
                    table_data = []
                    for job in jobs:
                        table_data.append([
                            job['jobId'][:8] + '...',
                            job['source'],
                            job['status'],
                            f"{job['progress']}%",
                            job['startTime'][:19] if job['startTime'] else ""
                        ])
                    
                    headers = ["Job ID", "Source", "Status", "Progress", "Start Time"]
                    print(tabulate(table_data, headers=headers, tablefmt="grid"))
                else:
                    print("No active jobs")
                return True
            else:
                print(f"❌ Failed to get active jobs: {response.status_code}")
                return False
    
    def cancel_job(self, job_id: str):
        """Cancel a loading job"""
        print(f"\n🛑 Cancelling job {job_id}...")
        
        data = {"jobId": job_id}
        
        response = requests.post(
            f"{self.api_base}/cancelLoadingJob",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Job cancelled successfully")
            print(f"   Job ID: {result.get('jobId')}")
            print(f"   Status: {result.get('status')}")
            return True
        else:
            print(f"❌ Failed to cancel job: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def _create_progress_bar(self, progress: int, width: int = 20) -> str:
        """Create a text progress bar"""
        filled = int(width * progress / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"{bar} {progress}%"

def main():
    parser = argparse.ArgumentParser(
        description='CLI Data Loader for Cryptotrading Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s yahoo --symbols BTC ETH SOL --days 30
  %(prog)s fred --series DGS10 WALCL M2SL --days 365
  %(prog)s gecko --networks ethereum polygon --pools 20
  %(prog)s all --symbols BTC ETH --series DGS10 --networks ethereum
  %(prog)s status
  %(prog)s jobs --watch
  %(prog)s cancel <job_id>
        """
    )
    
    parser.add_argument('--server', default=DEFAULT_SERVER, 
                       help=f'Server URL (default: {DEFAULT_SERVER})')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Yahoo Finance command
    yahoo_parser = subparsers.add_parser('yahoo', help='Load Yahoo Finance data')
    yahoo_parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'SOL'],
                             help='Crypto symbols to load (default: BTC ETH SOL)')
    yahoo_parser.add_argument('--days', type=int, default=30,
                             help='Number of days of history (default: 30)')
    yahoo_parser.add_argument('--interval', default='1d',
                             choices=['1m', '5m', '15m', '1h', '1d', '1wk'],
                             help='Data interval (default: 1d)')
    
    # FRED command
    fred_parser = subparsers.add_parser('fred', help='Load FRED economic data')
    fred_parser.add_argument('--series', nargs='+', default=['DGS10', 'WALCL', 'M2SL'],
                            help='FRED series to load (default: DGS10 WALCL M2SL)')
    fred_parser.add_argument('--days', type=int, default=365,
                            help='Number of days of history (default: 365)')
    
    # GeckoTerminal command
    gecko_parser = subparsers.add_parser('gecko', help='Load GeckoTerminal DEX data')
    gecko_parser.add_argument('--networks', nargs='+', 
                             default=['ethereum', 'polygon', 'bsc'],
                             help='Networks to load (default: ethereum polygon bsc)')
    gecko_parser.add_argument('--pools', type=int, default=20,
                             help='Number of pools per network (default: 20)')
    gecko_parser.add_argument('--no-volume', action='store_true',
                             help='Exclude volume data')
    gecko_parser.add_argument('--no-liquidity', action='store_true',
                             help='Exclude liquidity data')
    
    # All sources command
    all_parser = subparsers.add_parser('all', help='Load data from all sources')
    all_parser.add_argument('--symbols', nargs='+', 
                           help='Crypto symbols (default: BTC ETH SOL)')
    all_parser.add_argument('--series', nargs='+',
                           help='FRED series (default: DGS10 WALCL M2SL)')
    all_parser.add_argument('--networks', nargs='+',
                           help='DEX networks (default: ethereum polygon bsc)')
    all_parser.add_argument('--days', type=int, default=30,
                           help='Number of days of history (default: 30)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show data source status')
    
    # Jobs command
    jobs_parser = subparsers.add_parser('jobs', help='Show active jobs')
    jobs_parser.add_argument('--watch', action='store_true',
                            help='Watch jobs until completion')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel a job')
    cancel_parser.add_argument('job_id', help='Job ID to cancel')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = DataLoaderCLI(args.server)
    
    # Check server connection
    if not cli.check_server():
        print(f"❌ Cannot connect to server at {args.server}")
        print("   Please ensure the Flask server is running:")
        print("   python app.py")
        sys.exit(1)
    
    # Execute command
    if args.command == 'yahoo':
        job_id = cli.load_yahoo_data(args.symbols, args.days, args.interval)
        if job_id and input("\nMonitor job progress? (y/n): ").lower() == 'y':
            cli.get_active_jobs(watch=True)
            
    elif args.command == 'fred':
        job_id = cli.load_fred_data(args.series, args.days)
        if job_id and input("\nMonitor job progress? (y/n): ").lower() == 'y':
            cli.get_active_jobs(watch=True)
            
    elif args.command == 'gecko':
        job_id = cli.load_gecko_data(
            args.networks, 
            args.pools,
            not args.no_volume,
            not args.no_liquidity
        )
        if job_id and input("\nMonitor job progress? (y/n): ").lower() == 'y':
            cli.get_active_jobs(watch=True)
            
    elif args.command == 'all':
        job_ids = cli.load_all_data(args.symbols, args.series, args.networks, args.days)
        if job_ids and input("\nMonitor job progress? (y/n): ").lower() == 'y':
            cli.get_active_jobs(watch=True)
            
    elif args.command == 'status':
        cli.get_data_source_status()
        
    elif args.command == 'jobs':
        cli.get_active_jobs(watch=args.watch)
        
    elif args.command == 'cancel':
        cli.cancel_job(args.job_id)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()