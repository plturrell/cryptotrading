#!/usr/bin/env python3
"""
LOAD HISTORICAL DATA FROM CSV FILES TO DATABASE
This script loads the already-downloaded historical data into the database
"""

import sys
sys.path.append('src')

import asyncio
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cryptotrading.data.database.models import (
    Base, MarketData, TimeSeries, FactorData,
    DataSourceEnum, FactorFrequencyEnum
)
from cryptotrading.infrastructure.database.unified_database import UnifiedDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_yahoo_data_to_db():
    """Load Yahoo Finance CSV data into database"""
    logger.info("Loading Yahoo Finance data into database...")
    
    # Get unified database
    db = UnifiedDatabase()
    await db.initialize()
    
    # Path to Yahoo data
    yahoo_dir = Path("data/historical/yahoo")
    if not yahoo_dir.exists():
        logger.error(f"Yahoo data directory not found: {yahoo_dir}")
        return False
    
    # Get all CSV files
    csv_files = list(yahoo_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files to load")
    
    total_records_loaded = 0
    
    for csv_file in csv_files:
        try:
            # Extract symbol from filename (e.g., BTC-USD_1d_2023-08-18_2025-08-17.csv)
            filename_parts = csv_file.stem.split('_')
            symbol = filename_parts[0]
            
            logger.info(f"Loading {symbol} from {csv_file.name}...")
            
            # Read CSV data
            df = pd.read_csv(csv_file)
            
            # Ensure Date column is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            records_loaded = 0
            
            with db.get_session() as session:
                # Load data in batches
                batch_size = 100
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    for idx, row in batch.iterrows():
                        # Create TimeSeries record
                        ts_record = TimeSeries(
                            symbol=symbol,
                            timestamp=idx.to_pydatetime(),
                            frequency=FactorFrequencyEnum.DAILY,
                            source=DataSourceEnum.YAHOO,
                            open_price=float(row.get('Open', 0)),
                            high_price=float(row.get('High', 0)),
                            low_price=float(row.get('Low', 0)),
                            close_price=float(row.get('Close', 0)),
                            volume=float(row.get('Volume', 0)),
                            data_quality_score=1.0
                        )
                        session.add(ts_record)
                        
                        # Skip MarketData - use TimeSeries for OHLC data
                        # MarketData is for real-time snapshot data, not historical OHLC
                        
                        # Create FactorData record for spot price
                        factor_record = FactorData(
                            symbol=symbol,
                            timestamp=idx.to_pydatetime(),
                            factor_name='spot_price',
                            value=float(row.get('Close', 0)),
                            quality_score=1.0,  # Assuming Yahoo data is high quality
                            calculation_method='backfill'
                        )
                        session.add(factor_record)
                        
                        records_loaded += 1
                    
                    # Commit batch
                    session.commit()
                    
                    if (i + batch_size) % 1000 == 0:
                        logger.info(f"  Loaded {i + batch_size} records for {symbol}...")
            
            logger.info(f"✅ Loaded {records_loaded} records for {symbol}")
            total_records_loaded += records_loaded
            
        except Exception as e:
            logger.error(f"❌ Error loading {csv_file.name}: {e}")
            continue
    
    logger.info(f"✅ Total records loaded: {total_records_loaded}")
    return total_records_loaded > 0


async def load_fred_data_to_db():
    """Load FRED economic data into database"""
    logger.info("Loading FRED data into database...")
    
    # Get database
    db = get_db()
    
    # Path to FRED data
    fred_dir = Path("data/historical/fred")
    if not fred_dir.exists():
        logger.error(f"FRED data directory not found: {fred_dir}")
        return False
    
    # Get all CSV files
    csv_files = list(fred_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} FRED CSV files to load")
    
    total_records_loaded = 0
    
    for csv_file in csv_files:
        try:
            # Extract series ID from filename
            series_id = csv_file.stem.split('_')[0]
            
            logger.info(f"Loading FRED series {series_id} from {csv_file.name}...")
            
            # Read CSV data
            df = pd.read_csv(csv_file)
            
            # FRED data has different column names
            date_col = 'DATE' if 'DATE' in df.columns else df.columns[0]
            value_col = series_id if series_id in df.columns else df.columns[1]
            
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            records_loaded = 0
            
            with db.get_session() as session:
                # Load data in batches
                batch_size = 100
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    for idx, row in batch.iterrows():
                        # Skip if value is not numeric
                        try:
                            value = float(row[value_col])
                        except:
                            continue
                        
                        # Map FRED series to factor names
                        factor_name = {
                            'DGS10': 'us_treasury_10y',
                            'VIX': 'vix_index',
                            'WALCL': 'fed_balance_sheet',
                            'RRPONTSYD': 'reverse_repo',
                            'WTREGEN': 'treasury_general_account'
                        }.get(series_id, f'fred_{series_id.lower()}')
                        
                        # Create FactorData record
                        factor_record = FactorData(
                            symbol='MACRO',  # Use MACRO for economic indicators
                            timestamp=idx.to_pydatetime(),
                            factor_name=factor_name,
                            value=value,
                            quality_score=1.0,
                            calculation_method='backfill'
                        )
                        session.add(factor_record)
                        
                        records_loaded += 1
                    
                    # Commit batch
                    session.commit()
            
            logger.info(f"✅ Loaded {records_loaded} records for {series_id}")
            total_records_loaded += records_loaded
            
        except Exception as e:
            logger.error(f"❌ Error loading {csv_file.name}: {e}")
            continue
    
    logger.info(f"✅ Total FRED records loaded: {total_records_loaded}")
    return total_records_loaded > 0


async def verify_loaded_data():
    """Verify what data has been loaded into the database"""
    logger.info("\n🔍 Verifying loaded data...")
    
    db = get_db()
    
    with db.get_session() as session:
        # Count records in each table
        from sqlalchemy import select, func
        
        # MarketData
        market_count = session.scalar(select(func.count(MarketData.id)))
        logger.info(f"MarketData records: {market_count}")
        
        # Get date range for MarketData
        if market_count > 0:
            min_date = session.scalar(select(func.min(MarketData.timestamp)))
            max_date = session.scalar(select(func.max(MarketData.timestamp)))
            logger.info(f"  Date range: {min_date} to {max_date}")
            
            # Get unique symbols
            symbols = session.execute(
                select(MarketData.symbol).distinct()
            )
            symbol_list = [s[0] for s in symbols]
            logger.info(f"  Symbols: {', '.join(symbol_list)}")
        
        # TimeSeries
        ts_count = session.scalar(select(func.count(TimeSeries.id)))
        logger.info(f"TimeSeries records: {ts_count}")
        
        # FactorData
        factor_count = session.scalar(select(func.count(FactorData.id)))
        logger.info(f"FactorData records: {factor_count}")
        
        if factor_count > 0:
            # Get unique factors
            factors = session.execute(
                select(FactorData.factor_name).distinct()
            )
            factor_list = [f[0] for f in factors]
            logger.info(f"  Factors: {', '.join(factor_list[:5])}...")
        
        # Data quality metrics
        from cryptotrading.data.database.models import DataQualityMetrics
        quality_count = session.scalar(select(func.count(DataQualityMetrics.id)))
        logger.info(f"DataQualityMetrics records: {quality_count}")
        
        return {
            'market_data': market_count,
            'time_series': ts_count,
            'factor_data': factor_count,
            'quality': quality_count
        }


async def main():
    """Main function to load all historical data"""
    logger.info("🚀 LOADING HISTORICAL DATA INTO DATABASE")
    logger.info("=" * 60)
    
    # Check current state
    initial_counts = await verify_loaded_data()
    
    if initial_counts['market_data'] > 1000:
        logger.warning("⚠️  Database already contains significant data")
        response = input("Continue loading? This may create duplicates (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return
    
    # Load Yahoo data
    yahoo_success = await load_yahoo_data_to_db()
    
    # Load FRED data
    fred_success = await load_fred_data_to_db()
    
    # Verify final state
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL DATABASE STATE")
    final_counts = await verify_loaded_data()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📈 LOADING SUMMARY")
    logger.info(f"MarketData: {initial_counts['market_data']} → {final_counts['market_data']}")
    logger.info(f"TimeSeries: {initial_counts['time_series']} → {final_counts['time_series']}")
    logger.info(f"FactorData: {initial_counts['factor_data']} → {final_counts['factor_data']}")
    
    if yahoo_success and fred_success:
        logger.info("\n✅ Historical data successfully loaded into database!")
        logger.info("   2+ years of crypto and economic data now available")
    else:
        logger.error("\n❌ Some data loading failed. Check logs for details.")


if __name__ == "__main__":
    asyncio.run(main())