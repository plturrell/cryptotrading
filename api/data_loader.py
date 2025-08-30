"""
Real Data Loading Service Implementation
Actually fetches data from Yahoo Finance, FRED, and GeckoTerminal APIs
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import json
import uuid
from typing import Dict, List, Optional
import logging
from sqlalchemy import create_engine, text, Column, String, Integer, DateTime, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import sys
import pandas as pd
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
from src.cryptotrading.data.historical.fred_client import FREDClient
from src.cryptotrading.infrastructure.defi.dex_service import DEXService
from src.cryptotrading.data.database.models import MarketData, Base

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "sqlite:///cryptotrading.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

LocalBase = declarative_base()

# Use existing MarketData from models.py, create additional tables here


class EconomicData(LocalBase):
    __tablename__ = "economic_data"
    id = Column(String(50), primary_key=True)
    series_id = Column(String(50), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    value = Column(Float)
    title = Column(String(200))
    units = Column(String(50))
    frequency = Column(String(20))
    source = Column(String(20), default="fred")
    created_at = Column(DateTime, default=datetime.utcnow)


class DEXData(LocalBase):
    __tablename__ = "dex_data"
    id = Column(String(50), primary_key=True)
    network = Column(String(50), nullable=False, index=True)
    pool_address = Column(String(100), nullable=False)
    base_token_symbol = Column(String(20))
    quote_token_symbol = Column(String(20))
    price = Column(Float)
    volume_24h = Column(Float)
    liquidity = Column(Float)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(String(20), default="geckoterminal")
    created_at = Column(DateTime, default=datetime.utcnow)


# Job tracking tables (reuse from existing)
class DataSources(LocalBase):
    __tablename__ = "data_sources"
    id = Column(String(50), primary_key=True)
    name = Column(String(50))
    type = Column(String(20))
    isActive = Column(Boolean, default=True)
    createdAt = Column(DateTime)


class LoadingJobs(LocalBase):
    __tablename__ = "loading_jobs"
    id = Column(String(50), primary_key=True)
    source_id = Column(String(50))
    jobType = Column(String(50))
    status = Column(String(20), default="pending")
    parameters = Column(Text)
    symbols = Column(Text)
    startDate = Column(DateTime)
    endDate = Column(DateTime)
    interval = Column(String(10))
    progress = Column(Integer, default=0)
    scheduledAt = Column(DateTime)
    startedAt = Column(DateTime)
    completedAt = Column(DateTime)
    createdAt = Column(DateTime)
    recordsLoaded = Column(Integer, default=0)
    recordsFailed = Column(Integer, default=0)
    errorMessage = Column(Text)


# Create tables (existing MarketData comes from models.py, create new ones)
LocalBase.metadata.create_all(engine)
Base.metadata.create_all(engine)

# Initialize data clients
yahoo_client = YahooFinanceClient()
try:
    fred_client = FREDClient()
except Exception as e:
    logger.warning(f"FRED client initialization failed: {e}")
    fred_client = None

try:
    dex_service = DEXService()
except Exception as e:
    logger.warning(f"DEX service initialization failed: {e}")
    dex_service = None

# In-memory job tracking
active_jobs = {}


class RealDataLoader:
    def __init__(self):
        self.session = Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def load_yahoo_data(
        self, symbols: List[str], start_date: str, end_date: str, interval: str = "1d"
    ) -> Dict:
        """Actually load data from Yahoo Finance"""
        logger.info(f"Loading Yahoo data for {symbols} from {start_date} to {end_date}")

        results = {"symbols_processed": [], "symbols_failed": [], "total_records": 0, "errors": []}

        try:
            for symbol in symbols:
                try:
                    logger.info(f"Fetching data for {symbol}")

                    # Convert to Yahoo Finance symbol format
                    yahoo_symbol = f"{symbol}-USD" if symbol not in ["USD"] else symbol

                    # Download data using the existing Yahoo client
                    df = yahoo_client.download_data(
                        yahoo_symbol, start_date, end_date, interval=interval, save=False
                    )

                    if df is not None and not df.empty:
                        # Store in database
                        records_saved = self._save_yahoo_data(symbol, df, interval)
                        results["symbols_processed"].append(symbol)
                        results["total_records"] += records_saved
                        logger.info(f"Saved {records_saved} records for {symbol}")
                    else:
                        results["symbols_failed"].append(symbol)
                        results["errors"].append(f"No data returned for {symbol}")
                        logger.warning(f"No data returned for {symbol}")

                except Exception as e:
                    logger.error(f"Error loading {symbol}: {e}")
                    results["symbols_failed"].append(symbol)
                    results["errors"].append(f"{symbol}: {str(e)}")

        except Exception as e:
            logger.error(f"Yahoo data loading failed: {e}")
            results["errors"].append(f"General error: {str(e)}")

        return results

    def _save_yahoo_data(self, symbol: str, df: pd.DataFrame, interval: str) -> int:
        """Save Yahoo Finance data to database"""
        try:
            records_saved = 0

            # Reset index to make Date a column if it's the index
            if "Date" not in df.columns and df.index.name == "Date":
                df = df.reset_index()
            elif "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.columns = ["Date"] + list(df.columns[1:])

            for _, row in df.iterrows():
                try:
                    # Create unique ID
                    record_id = str(uuid.uuid4())

                    # Parse date
                    if "Date" in row:
                        date_val = pd.to_datetime(row["Date"]).to_pydatetime()
                    else:
                        # Use index if Date column doesn't exist
                        date_val = pd.to_datetime(row.name).to_pydatetime()

                    # Check if record already exists
                    existing = (
                        self.session.query(MarketData)
                        .filter(
                            MarketData.symbol == symbol,
                            MarketData.timestamp == date_val,
                        )
                        .first()
                    )

                    if existing:
                        continue  # Skip duplicate

                    # Get price data (try both cases)
                    close_price = row.get("close") or row.get("Close")
                    volume = row.get("volume") or row.get("Volume") 
                    high_price = row.get("high") or row.get("High")
                    low_price = row.get("low") or row.get("Low")
                    
                    # Skip records with no price data
                    if pd.isna(close_price) or close_price is None:
                        logger.warning(f"Skipping record with no price data for {symbol} on {date_val}")
                        continue
                    
                    # Create new record using existing MarketData schema
                    market_record = MarketData(
                        symbol=symbol,
                        timestamp=date_val,
                        price=float(close_price),
                        volume_24h=float(volume) if pd.notna(volume) and volume is not None else None,
                        high_24h=float(high_price) if pd.notna(high_price) and high_price is not None else None,
                        low_24h=float(low_price) if pd.notna(low_price) and low_price is not None else None,
                    )

                    self.session.add(market_record)
                    records_saved += 1

                except Exception as e:
                    logger.error(f"Error saving record for {symbol}: {e}")
                    continue

            self.session.commit()
            return records_saved

        except Exception as e:
            logger.error(f"Error saving Yahoo data: {e}")
            self.session.rollback()
            return 0

    def load_fred_data(self, series_list: List[str], start_date: str, end_date: str) -> Dict:
        """Actually load data from FRED"""
        if not fred_client:
            return {
                "series_processed": [],
                "series_failed": series_list,
                "total_records": 0,
                "errors": ["FRED client not available - API key required"],
            }

        logger.info(f"Loading FRED data for {series_list} from {start_date} to {end_date}")

        results = {"series_processed": [], "series_failed": [], "total_records": 0, "errors": []}

        try:
            for series_id in series_list:
                try:
                    logger.info(f"Fetching FRED series {series_id}")

                    # Use FRED client to get data
                    df = fred_client.get_series_data(series_id, start_date, end_date)

                    if df is not None and not df.empty:
                        # Store in database
                        records_saved = self._save_fred_data(series_id, df)
                        results["series_processed"].append(series_id)
                        results["total_records"] += records_saved
                        logger.info(f"Saved {records_saved} records for {series_id}")
                    else:
                        results["series_failed"].append(series_id)
                        results["errors"].append(f"No data returned for {series_id}")

                except Exception as e:
                    logger.error(f"Error loading {series_id}: {e}")
                    results["series_failed"].append(series_id)
                    results["errors"].append(f"{series_id}: {str(e)}")

        except Exception as e:
            logger.error(f"FRED data loading failed: {e}")
            results["errors"].append(f"General error: {str(e)}")

        return results

    def _save_fred_data(self, series_id: str, df: pd.DataFrame) -> int:
        """Save FRED data to database"""
        try:
            records_saved = 0

            for date, value in df.items():
                try:
                    if pd.isna(value):
                        continue

                    # Create unique ID
                    record_id = str(uuid.uuid4())
                    date_val = pd.to_datetime(date).to_pydatetime()

                    # Check if record already exists
                    existing = (
                        self.session.query(EconomicData)
                        .filter(EconomicData.series_id == series_id, EconomicData.date == date_val)
                        .first()
                    )

                    if existing:
                        continue  # Skip duplicate

                    # Create new record
                    econ_record = EconomicData(
                        id=record_id,
                        series_id=series_id,
                        date=date_val,
                        value=float(value),
                        source="fred",
                    )

                    self.session.add(econ_record)
                    records_saved += 1

                except Exception as e:
                    logger.error(f"Error saving FRED record: {e}")
                    continue

            self.session.commit()
            return records_saved

        except Exception as e:
            logger.error(f"Error saving FRED data: {e}")
            self.session.rollback()
            return 0

    def load_dex_data(self, networks: List[str], pool_count: int = 20) -> Dict:
        """Load DEX data from GeckoTerminal"""
        if not dex_service:
            return {
                "networks_processed": [],
                "networks_failed": networks,
                "total_records": 0,
                "errors": ["DEX service not available"],
            }

        logger.info(f"Loading DEX data for {networks}, {pool_count} pools each")

        results = {
            "networks_processed": [],
            "networks_failed": [],
            "total_records": 0,
            "errors": [],
        }

        try:
            import asyncio
            from dataclasses import asdict
            
            # Use async context manager for DEX service
            async def fetch_all_networks():
                async with dex_service as dex:
                    all_results = []
                    for network in networks:
                        try:
                            logger.info(f"Fetching DEX data for {network}")
                            pools_data = await dex.get_trending_pools(network, limit=pool_count)
                            # Convert DEXPool objects to dict format for compatibility
                            pools = [asdict(pool) for pool in pools_data] if pools_data else []
                            all_results.append((network, pools))
                        except Exception as e:
                            logger.error(f"Error fetching DEX data for {network}: {e}")
                            all_results.append((network, None))
                    return all_results
            
            # Run async fetching
            network_results = asyncio.run(fetch_all_networks())
            
            for network, pools in network_results:
                if pools is not None and len(pools) > 0:
                    records_saved = self._save_dex_data(network, pools)
                    results["networks_processed"].append(network)
                    results["total_records"] += records_saved
                    logger.info(f"Saved {records_saved} pool records for {network}")
                else:
                    results["networks_failed"].append(network)
                    results["errors"].append(f"No pools returned for {network}")

        except Exception as e:
            logger.error(f"DEX data loading failed: {e}")
            results["errors"].append(f"General error: {str(e)}")

        return results

    def _save_dex_data(self, network: str, pools: List[Dict]) -> int:
        """Save DEX pool data to database"""
        try:
            records_saved = 0

            for pool in pools:
                try:
                    # Create unique ID
                    record_id = str(uuid.uuid4())
                    pool_address = pool.get("address", "")

                    # Check if record already exists for this pool today
                    today = datetime.now().date()
                    existing = (
                        self.session.query(DEXData)
                        .filter(
                            DEXData.network == network,
                            DEXData.pool_address == pool_address,
                            DEXData.timestamp >= datetime.combine(today, datetime.min.time()),
                        )
                        .first()
                    )

                    if existing:
                        continue  # Skip duplicate

                    # Create new record (adapted for DEXPool dataclass format)
                    dex_record = DEXData(
                        id=record_id,
                        network=network,
                        pool_address=pool_address,
                        base_token_symbol=pool.get("token0_symbol", ""),
                        quote_token_symbol=pool.get("token1_symbol", ""),
                        price=None,  # DEXPool doesn't have direct price, could calculate from liquidity
                        volume_24h=float(pool.get("volume_24h_usd", 0))
                        if pool.get("volume_24h_usd")
                        else None,
                        liquidity=float(pool.get("liquidity_usd", 0))
                        if pool.get("liquidity_usd")
                        else None,
                        timestamp=datetime.utcnow(),
                        source="geckoterminal",
                    )

                    self.session.add(dex_record)
                    records_saved += 1

                except Exception as e:
                    logger.error(f"Error saving DEX record: {e}")
                    continue

            self.session.commit()
            return records_saved

        except Exception as e:
            logger.error(f"Error saving DEX data: {e}")
            self.session.rollback()
            return 0


# Create blueprint for real data loading
real_data_loading_bp = Blueprint(
    "real_data_loading", __name__, url_prefix="/api/odata/v4/RealDataLoadingService"
)


@real_data_loading_bp.route("/loadRealYahooData", methods=["POST"])
def load_real_yahoo_data():
    """Load real data from Yahoo Finance"""
    try:
        data = request.json
        symbols = data.get("symbols", [])
        start_date = data.get("startDate")
        end_date = data.get("endDate")
        interval = data.get("interval", "1d")

        if not symbols:
            return jsonify({"error": "No symbols provided"}), 400

        # Parse dates
        start_dt = datetime.fromisoformat(start_date.replace("Z", ""))
        end_dt = datetime.fromisoformat(end_date.replace("Z", ""))

        with RealDataLoader() as loader:
            results = loader.load_yahoo_data(
                symbols, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), interval
            )

        return (
            jsonify(
                {
                    "status": "completed",
                    "symbols_processed": results["symbols_processed"],
                    "symbols_failed": results["symbols_failed"],
                    "total_records_loaded": results["total_records"],
                    "errors": results["errors"],
                    "summary": f"Loaded {results['total_records']} records for {len(results['symbols_processed'])} symbols",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Real Yahoo data loading failed: {e}")
        return jsonify({"error": str(e)}), 500


@real_data_loading_bp.route("/loadRealFREDData", methods=["POST"])
def load_real_fred_data():
    """Load real data from FRED"""
    try:
        data = request.json
        series = data.get("series", [])
        start_date = data.get("startDate")
        end_date = data.get("endDate")

        if not series:
            return jsonify({"error": "No series provided"}), 400

        # Parse dates
        start_dt = datetime.fromisoformat(start_date.replace("Z", ""))
        end_dt = datetime.fromisoformat(end_date.replace("Z", ""))

        with RealDataLoader() as loader:
            results = loader.load_fred_data(
                series, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
            )

        return (
            jsonify(
                {
                    "status": "completed",
                    "series_processed": results["series_processed"],
                    "series_failed": results["series_failed"],
                    "total_records_loaded": results["total_records"],
                    "errors": results["errors"],
                    "summary": f"Loaded {results['total_records']} records for {len(results['series_processed'])} series",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Real FRED data loading failed: {e}")
        return jsonify({"error": str(e)}), 500


@real_data_loading_bp.route("/loadRealDEXData", methods=["POST"])
def load_real_dex_data():
    """Load real data from GeckoTerminal"""
    try:
        data = request.json
        networks = data.get("networks", [])
        pool_count = data.get("poolCount", 20)

        if not networks:
            return jsonify({"error": "No networks provided"}), 400

        with RealDataLoader() as loader:
            results = loader.load_dex_data(networks, pool_count)

        return (
            jsonify(
                {
                    "status": "completed",
                    "networks_processed": results["networks_processed"],
                    "networks_failed": results["networks_failed"],
                    "total_records_loaded": results["total_records"],
                    "errors": results["errors"],
                    "summary": f"Loaded {results['total_records']} pool records for {len(results['networks_processed'])} networks",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Real DEX data loading failed: {e}")
        return jsonify({"error": str(e)}), 500


@real_data_loading_bp.route("/getDataSummary", methods=["GET"])
def get_data_summary():
    """Get summary of stored data"""
    try:
        session = Session()

        # Count records by source
        market_data_count = session.query(MarketData).count()
        economic_data_count = session.query(EconomicData).count()
        dex_data_count = session.query(DEXData).count()

        # Get unique symbols/series
        unique_symbols = session.query(MarketData.symbol).distinct().all()
        unique_series = session.query(EconomicData.series_id).distinct().all()
        unique_networks = session.query(DEXData.network).distinct().all()

        # Get date ranges
        market_date_range = session.execute(
            text("SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM market_data")
        ).first()

        economic_date_range = session.execute(
            text("SELECT MIN(date) as min_date, MAX(date) as max_date FROM economic_data")
        ).first()

        dex_date_range = session.execute(
            text("SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM dex_data")
        ).first()

        session.close()

        return (
            jsonify(
                {
                    "summary": {
                        "market_data": {
                            "total_records": market_data_count,
                            "unique_symbols": len(unique_symbols),
                            "symbols": [s[0] for s in unique_symbols],
                            "date_range": {
                                "start": str(market_date_range.min_date)
                                if market_date_range.min_date
                                else None,
                                "end": str(market_date_range.max_date)
                                if market_date_range.max_date
                                else None,
                            },
                        },
                        "economic_data": {
                            "total_records": economic_data_count,
                            "unique_series": len(unique_series),
                            "series": [s[0] for s in unique_series],
                            "date_range": {
                                "start": str(economic_date_range.min_date)
                                if economic_date_range.min_date
                                else None,
                                "end": str(economic_date_range.max_date)
                                if economic_date_range.max_date
                                else None,
                            },
                        },
                        "dex_data": {
                            "total_records": dex_data_count,
                            "unique_networks": len(unique_networks),
                            "networks": [n[0] for n in unique_networks],
                            "date_range": {
                                "start": str(dex_date_range.min_date)
                                if dex_date_range.min_date
                                else None,
                                "end": str(dex_date_range.max_date)
                                if dex_date_range.max_date
                                else None,
                            },
                        },
                    },
                    "total_records": market_data_count + economic_data_count + dex_data_count,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        return jsonify({"error": str(e)}), 500
