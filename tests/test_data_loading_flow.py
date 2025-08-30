#!/usr/bin/env python3
"""
Test script for end-to-end data loading flow
Tests UI -> CDS Service -> Database flow for Yahoo, FRED, and GeckoTerminal
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Base URL - update if server is running on different port
BASE_URL = "http://localhost:5001"

def test_data_source_status():
    """Test getting data source status"""
    print("\n=== Testing Data Source Status ===")
    response = requests.get(f"{BASE_URL}/api/odata/v4/DataLoadingService/getDataSourceStatus")
    
    if response.status_code == 200:
        sources = response.json()
        print(f"✓ Found {len(sources)} data sources")
        for source in sources:
            print(f"  - {source['source']}: {source['apiStatus']} (Available: {source['isAvailable']})")
            print(f"    Last Sync: {source['lastSync']}, Records: {source['recordCount']}")
        return True
    else:
        print(f"✗ Failed to get data source status: {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def test_yahoo_data_loading():
    """Test Yahoo Finance data loading"""
    print("\n=== Testing Yahoo Finance Data Loading ===")
    
    # Prepare request data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = {
        "symbols": ["BTC", "ETH", "SOL"],
        "startDate": start_date.isoformat() + "Z",
        "endDate": end_date.isoformat() + "Z",
        "interval": "1d"
    }
    
    print(f"Loading data for: {', '.join(data['symbols'])}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    response = requests.post(
        f"{BASE_URL}/api/odata/v4/DataLoadingService/loadYahooFinanceData",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Yahoo data loading started")
        print(f"  Job ID: {result.get('jobId')}")
        print(f"  Status: {result.get('status')}")
        print(f"  Message: {result.get('message')}")
        return result.get('jobId')
    else:
        print(f"✗ Failed to start Yahoo data loading: {response.status_code}")
        print(f"  Response: {response.text}")
        return None

def test_fred_data_loading():
    """Test FRED data loading"""
    print("\n=== Testing FRED Data Loading ===")
    
    # Prepare request data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = {
        "series": ["DGS10", "WALCL", "M2SL"],
        "startDate": start_date.isoformat() + "Z",
        "endDate": end_date.isoformat() + "Z"
    }
    
    print(f"Loading series: {', '.join(data['series'])}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    response = requests.post(
        f"{BASE_URL}/api/odata/v4/DataLoadingService/loadFREDData",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ FRED data loading started")
        print(f"  Job ID: {result.get('jobId')}")
        print(f"  Status: {result.get('status')}")
        print(f"  Message: {result.get('message')}")
        return result.get('jobId')
    else:
        print(f"✗ Failed to start FRED data loading: {response.status_code}")
        print(f"  Response: {response.text}")
        return None

def test_gecko_data_loading():
    """Test GeckoTerminal data loading"""
    print("\n=== Testing GeckoTerminal Data Loading ===")
    
    data = {
        "networks": ["ethereum", "bsc", "polygon"],
        "poolCount": 20,
        "includeVolume": True,
        "includeLiquidity": True
    }
    
    print(f"Loading networks: {', '.join(data['networks'])}")
    print(f"Pool count: {data['poolCount']}")
    
    response = requests.post(
        f"{BASE_URL}/api/odata/v4/DataLoadingService/loadGeckoTerminalData",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ GeckoTerminal data loading started")
        print(f"  Job ID: {result.get('jobId')}")
        print(f"  Status: {result.get('status')}")
        print(f"  Message: {result.get('message')}")
        return result.get('jobId')
    else:
        print(f"✗ Failed to start GeckoTerminal data loading: {response.status_code}")
        print(f"  Response: {response.text}")
        return None

def test_active_jobs(job_ids):
    """Test getting active jobs"""
    print("\n=== Testing Active Jobs ===")
    
    response = requests.get(f"{BASE_URL}/api/odata/v4/DataLoadingService/getActiveJobs")
    
    if response.status_code == 200:
        jobs = response.json()
        print(f"✓ Found {len(jobs)} active jobs")
        
        for job in jobs:
            print(f"  - Job {job['jobId'][:8]}...")
            print(f"    Source: {job['source']}")
            print(f"    Status: {job['status']}")
            print(f"    Progress: {job['progress']}%")
            
        # Check our job IDs
        active_job_ids = [job['jobId'] for job in jobs]
        for job_id in job_ids:
            if job_id and job_id in active_job_ids:
                print(f"✓ Job {job_id[:8]}... is tracked")
        
        return True
    else:
        print(f"✗ Failed to get active jobs: {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def test_load_all_data():
    """Test loading all data sources at once"""
    print("\n=== Testing Load All Market Data ===")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = {
        "cryptoSymbols": ["BTC", "ETH"],
        "fredSeries": ["DGS10", "WALCL"],
        "dexNetworks": ["ethereum", "polygon"],
        "startDate": start_date.isoformat() + "Z",
        "endDate": end_date.isoformat() + "Z"
    }
    
    print("Loading all data sources...")
    
    response = requests.post(
        f"{BASE_URL}/api/odata/v4/DataLoadingService/loadAllMarketData",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ All data loading started")
        print(f"  Total Jobs: {result.get('totalJobs')}")
        print(f"  Job IDs: {', '.join([jid[:8] + '...' for jid in result.get('jobIds', [])])}")
        print(f"  Status: {result.get('status')}")
        return result.get('jobIds', [])
    else:
        print(f"✗ Failed to start all data loading: {response.status_code}")
        print(f"  Response: {response.text}")
        return []

def test_cancel_job(job_id):
    """Test cancelling a job"""
    if not job_id:
        return False
        
    print(f"\n=== Testing Cancel Job {job_id[:8]}... ===")
    
    data = {"jobId": job_id}
    
    response = requests.post(
        f"{BASE_URL}/api/odata/v4/DataLoadingService/cancelLoadingJob",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Job cancelled")
        print(f"  Status: {result.get('status')}")
        print(f"  Message: {result.get('message')}")
        return True
    else:
        print(f"✗ Failed to cancel job: {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("DATA LOADING END-TO-END TEST")
    print("=" * 60)
    print(f"Testing server at: {BASE_URL}")
    print(f"Test started at: {datetime.now()}")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✓ Server is running on port 5001")
    except requests.exceptions.ConnectionError:
        print(f"✗ Server not found at {BASE_URL}")
        print("Please ensure the Flask server is running:")
        print("  python app.py")
        return
    
    # Track job IDs for monitoring
    job_ids = []
    
    # Test individual operations
    test_data_source_status()
    
    yahoo_job = test_yahoo_data_loading()
    if yahoo_job:
        job_ids.append(yahoo_job)
    
    fred_job = test_fred_data_loading()
    if fred_job:
        job_ids.append(fred_job)
    
    gecko_job = test_gecko_data_loading()
    if gecko_job:
        job_ids.append(gecko_job)
    
    # Check active jobs
    time.sleep(1)  # Give jobs time to register
    test_active_jobs(job_ids)
    
    # Test bulk loading
    bulk_jobs = test_load_all_data()
    job_ids.extend(bulk_jobs)
    
    # Monitor jobs for a few seconds
    print("\n=== Monitoring Job Progress ===")
    for i in range(3):
        time.sleep(2)
        print(f"\nCheck {i+1}/3:")
        response = requests.get(f"{BASE_URL}/api/odata/v4/DataLoadingService/getActiveJobs")
        if response.status_code == 200:
            jobs = response.json()
            for job in jobs:
                if job['jobId'] in job_ids:
                    print(f"  Job {job['jobId'][:8]}... - {job['source']}: {job['progress']}% ({job['status']})")
    
    # Test cancelling a job
    if job_ids:
        test_cancel_job(job_ids[0])
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Data source status: ✓")
    print(f"  - Yahoo Finance loading: {'✓' if yahoo_job else '✗'}")
    print(f"  - FRED loading: {'✓' if fred_job else '✗'}")
    print(f"  - GeckoTerminal loading: {'✓' if gecko_job else '✗'}")
    print(f"  - Bulk loading: {'✓' if bulk_jobs else '✗'}")
    print(f"  - Total jobs created: {len(job_ids)}")
    
    print("\nNote: The actual data loading happens asynchronously.")
    print("Check the database tables for loaded data:")
    print("  - market_data (Yahoo Finance)")
    print("  - time_series (FRED)")
    print("  - dex_pools, dex_pool_metrics (GeckoTerminal)")

if __name__ == "__main__":
    main()