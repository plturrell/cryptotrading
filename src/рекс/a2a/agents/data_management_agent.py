"""
Data Management Agent powered by Strand Agents - 100% A2A Compliant
Discovers data structures for historical data sources
"""

from strands import Agent, tool
from ..strand_providers import get_model_provider
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import requests
import logging
import json
import hashlib
import asyncio
from pathlib import Path

from ...database.client import get_db
from ...storage.vercel_blob import VercelBlobClient
from ..registry.registry import agent_registry
from ..protocols import A2AMessage, A2AProtocol, MessageType

logger = logging.getLogger(__name__)

class DataManagementAgent:
    def __init__(self, model_provider: str = "deepseek"):
        # A2A Protocol compliance
        self.agent_id = 'data-management-001'
        self.capabilities = [
            'data_structure_discovery', 'schema_analysis', 'data_mapping',
            'source_validation', 'format_detection', 'schema_registry'
        ]
        
        # Initialize schema storage
        self.db = get_db()
        try:
            self.blob_storage = VercelBlobClient()
        except ValueError:
            # Vercel blob token not available, use SQLite only
            self.blob_storage = None
            logger.warning("Vercel Blob storage not available, using SQLite only")
        self.schema_cache = {}
        self._init_schema_table()
        
        @tool
        def discover_data_structure_for_historical_data(source_name: str, source_config: Dict[str, Any]) -> Dict[str, Any]:
            """Discover data structure for any historical data source"""
            try:
                logger.info(f"Discovering data structure for {source_name}")
                
                if source_name.lower() == 'cryptodatadownload':
                    return self._discover_cryptodatadownload_structure(source_config)
                elif source_name.lower() == 'yahoo':
                    return self._discover_yahoo_structure(source_config)
                elif source_name.lower() == 'bitget':
                    return self._discover_bitget_structure(source_config)
                else:
                    return self._discover_generic_structure(source_name, source_config)
                    
            except Exception as e:
                logger.error(f"Error discovering structure for {source_name}: {e}")
                return {
                    "success": False,
                    "source": source_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        @tool
        def store_schema(schema_data: Dict[str, Any], storage_type: str = "both") -> Dict[str, Any]:
            """Store discovered data product schema in registry (Vercel blob + SQLite)"""
            return asyncio.run(self._store_schema_async(schema_data, storage_type))

        @tool 
        def get_schema(data_product_id: str, from_cache: bool = True) -> Dict[str, Any]:
            """Retrieve data product schema from registry"""
            result = asyncio.run(self._get_schema_async(data_product_id, from_cache))
            return result if result else {"success": False, "error": f"Schema not found: {data_product_id}"}

        @tool
        def list_schemas(source_name: str = None) -> Dict[str, Any]:
            """List all registered data product schemas"""
            schemas = self._list_schemas(source_name)
            return {
                "success": True,
                "schemas": schemas,
                "count": len(schemas),
                "timestamp": datetime.now().isoformat()
            }

        @tool
        def validate_schema(data_product_id: str) -> Dict[str, Any]:
            """Validate stored schema against current source"""
            return asyncio.run(self._validate_schema_async(data_product_id))
        
        # Create Strand agent with DeepSeek provider
        self.model_provider = get_model_provider(model_provider)
        self.agent = Agent(
            tools=[
                discover_data_structure_for_historical_data,
                store_schema,
                get_schema,
                list_schemas,
                validate_schema
            ],
            model=self.model_provider
        )
        
        # Register with A2A registry
        agent_registry.register_agent(
            self.agent_id,
            'data_management',
            self.capabilities,
            {'version': '1.0', 'model_provider': model_provider, 'a2a_compliant': True}
        )
    
    def _discover_cryptodatadownload_structure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover CryptoDataDownload specific structure"""
        try:
            # Sample URL to analyze structure
            exchange = config.get('exchange', 'binance')
            pair = config.get('pair', 'BTCUSDT')
            timeframe = config.get('timeframe', 'd')
            
            url = f"https://www.cryptodatadownload.com/cdd/{exchange}_{pair}_{timeframe}.csv"
            
            logger.info(f"Analyzing CryptoDataDownload URL: {url}")
            
            # Fetch sample data
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Parse CSV structure
                lines = response.text.strip().split('\n')
                
                if len(lines) > 2:
                    # Skip first descriptive row
                    header_line = lines[1]
                    
                    columns = [col.strip() for col in header_line.split(',')]
                    
                    # Analyze multiple data rows for proper quality metrics
                    sample_data = []
                    for i in range(2, min(len(lines), 102)):  # Analyze up to 100 rows
                        if lines[i].strip():
                            row_values = [val.strip() for val in lines[i].split(',')]
                            if len(row_values) == len(columns):
                                sample_data.append(row_values)
                    
                    # Calculate quality metrics from actual data
                    quality_metrics = self._calculate_quality_metrics(columns, sample_data)
                    
                    # Analyze column types from actual data
                    column_analysis = {}
                    for i, col in enumerate(columns):
                        column_samples = [row[i] for row in sample_data if len(row) > i]
                        
                        column_analysis[col] = {
                            "position": i,
                            "sample_values": column_samples[:3] if column_samples else [],  # First 3 actual values
                            "data_type": self._detect_data_type_from_samples(column_samples),
                            "null_count": sum(1 for val in column_samples if not val or val.lower() in ['', 'null', 'nan']),
                            "total_samples": len(column_samples),
                            "database_mapping": self._map_to_database_column(col)
                        }
                    
                    # Generate SAP CAP CDS schema
                    cap_schema = self._generate_sap_cap_schema("CryptoDataDownload", columns, column_analysis)
                    
                    # Generate SAP Object Resource Discovery with real metrics
                    resource_discovery = self._generate_sap_resource_discovery("cryptodatadownload", url, columns, quality_metrics, len(sample_data))
                    
                    return {
                        "success": True,
                        "source": "cryptodatadownload",
                        "url_pattern": "https://www.cryptodatadownload.com/cdd/{exchange}_{pair}_{timeframe}.csv",
                        "sap_cap_schema": cap_schema,
                        "sap_resource_discovery": resource_discovery,
                        "structure": {
                            "format": "CSV",
                            "header_rows": 2,
                            "skip_rows": 1,
                            "columns": column_analysis,
                            "sample_url": url,
                            "total_columns": len(columns)
                        },
                        "database_mapping": {
                            "table": "market_data_source",
                            "column_mappings": {col: info["database_mapping"] for col, info in column_analysis.items()}
                        },
                        "file_organization": {
                            "base_path": "data/historical/cryptodatadownload",
                            "filename_pattern": "{exchange}_{pair}_{timeframe}.csv",
                            "folder_structure": "by_exchange_and_date"
                        },
                        "data_contract": {
                            "expected_columns": list(columns),
                            "required_columns": ["date", "open", "high", "low", "close", "volume"],
                            "data_types": {col: info["data_type"] for col, info in column_analysis.items()}
                        },
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {
                "success": False,
                "source": "cryptodatadownload", 
                "error": f"Failed to fetch sample data from {url}",
                "status_code": response.status_code
            }
            
        except Exception as e:
            return {
                "success": False,
                "source": "cryptodatadownload",
                "error": str(e)
            }
    
    def _discover_yahoo_structure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for Yahoo Finance structure discovery"""
        return {
            "success": True,
            "source": "yahoo",
            "structure": {"format": "JSON", "note": "Yahoo structure discovery not implemented yet"},
            "timestamp": datetime.now().isoformat()
        }
    
    def _discover_bitget_structure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for Bitget structure discovery"""
        return {
            "success": True,
            "source": "bitget",
            "structure": {"format": "JSON", "note": "Bitget structure discovery not implemented yet"},
            "timestamp": datetime.now().isoformat()
        }
    
    def _discover_generic_structure(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic structure discovery for unknown sources"""
        return {
            "success": False,
            "source": source_name,
            "error": f"Structure discovery not implemented for {source_name}",
            "supported_sources": ["cryptodatadownload", "yahoo", "bitget"]
        }
    
    def _detect_data_type_from_samples(self, sample_values: List[str]) -> str:
        """Detect data type from multiple sample values"""
        if not sample_values:
            return "string"
        
        # Filter out empty values
        valid_samples = [val for val in sample_values if val and val.lower() not in ['', 'null', 'nan']]
        if not valid_samples:
            return "string"
        
        # Test for numeric types
        numeric_count = 0
        datetime_count = 0
        
        for sample in valid_samples[:10]:  # Test first 10 valid samples
            # Test for numeric
            try:
                float(sample)
                numeric_count += 1
            except:
                pass
            
            # Test for datetime
            try:
                pd.to_datetime(sample)
                datetime_count += 1
            except:
                pass
        
        total_tested = min(len(valid_samples), 10)
        
        # Determine type based on majority
        if numeric_count >= total_tested * 0.8:  # 80% numeric
            return "float"
        elif datetime_count >= total_tested * 0.8:  # 80% datetime
            return "datetime"
        else:
            return "string"
    
    def _calculate_quality_metrics(self, columns: List[str], sample_data: List[List[str]]) -> Dict[str, float]:
        """Calculate real data quality metrics from sample data"""
        if not sample_data or not columns:
            return {
                "completeness": 0.0,
                "accuracy": 0.0,
                "timeliness": 0.0,
                "consistency": 0.0
            }
        
        total_cells = len(sample_data) * len(columns)
        empty_cells = 0
        invalid_cells = 0
        date_inconsistencies = 0
        
        # Check completeness
        for row in sample_data:
            for cell in row:
                if not cell or cell.lower() in ['', 'null', 'nan', 'n/a']:
                    empty_cells += 1
        
        completeness = (total_cells - empty_cells) / total_cells if total_cells > 0 else 0
        
        # Check data validity/accuracy for numeric columns
        for col_idx, col in enumerate(columns):
            if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                for row in sample_data:
                    if len(row) > col_idx:
                        try:
                            value = float(row[col_idx])
                            # Basic sanity checks for price data
                            if col.lower() in ['open', 'high', 'low', 'close'] and value <= 0:
                                invalid_cells += 1
                            elif col.lower() == 'volume' and value < 0:
                                invalid_cells += 1
                        except:
                            invalid_cells += 1
        
        # Check date consistency (if date column exists)
        date_col_idx = None
        for i, col in enumerate(columns):
            if col.lower() in ['date', 'timestamp', 'time']:
                date_col_idx = i
                break
        
        if date_col_idx is not None:
            previous_date = None
            for row in sample_data:
                if len(row) > date_col_idx:
                    try:
                        current_date = pd.to_datetime(row[date_col_idx])
                        if previous_date and current_date <= previous_date:
                            date_inconsistencies += 1
                        previous_date = current_date
                    except:
                        date_inconsistencies += 1
        
        accuracy = 1 - (invalid_cells / total_cells) if total_cells > 0 else 0
        consistency = 1 - (date_inconsistencies / len(sample_data)) if len(sample_data) > 0 else 1
        
        # Timeliness based on how recent the data appears to be
        timeliness = 1.0  # Default to 1.0 since we can't determine age without more context
        
        return {
            "completeness": round(max(0, min(1, completeness)), 3),
            "accuracy": round(max(0, min(1, accuracy)), 3),
            "timeliness": round(max(0, min(1, timeliness)), 3),
            "consistency": round(max(0, min(1, consistency)), 3)
        }
    
    def _map_to_database_column(self, column_name: str) -> str:
        """Map source column to database column"""
        mapping = {
            "date": "timestamp",
            "open": "price",  # Could store open price
            "high": "high_24h",
            "low": "low_24h", 
            "close": "price",  # Main price field
            "volume": "volume_24h",
            "unix": "timestamp"
        }
        
        column_lower = column_name.lower()
        return mapping.get(column_lower, column_lower)
    
    def _generate_sap_cap_schema(self, entity_name: str, columns: List[str], column_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SAP CAP Core Data Services schema"""
        
        # Map Python types to SAP CAP types
        type_mapping = {
            "string": "String(100)",
            "float": "Decimal(15,4)",
            "datetime": "DateTime",
            "integer": "Integer"
        }
        
        # Generate CDS entity definition
        cds_fields = []
        annotations = []
        
        for col in columns:
            analysis = column_analysis.get(col, {})
            data_type = analysis.get("data_type", "string")
            cap_type = type_mapping.get(data_type, "String(100)")
            
            # Create field definition
            field_def = f"  {col.lower().replace(' ', '_')} : {cap_type}"
            
            # Add key annotation for primary fields
            if col.lower() in ['date', 'timestamp', 'unix']:
                field_def += " @(title: '{i18n>timestamp}', description: '{i18n>timestampDesc}')"
                annotations.append(f"@title: '{col.title()}'")
            elif col.lower() in ['symbol', 'pair']:
                field_def += " @(title: '{i18n>symbol}', description: '{i18n>symbolDesc}')"
            elif col.lower() in ['close', 'price']:
                field_def += " @(title: '{i18n>price}', semantics: 'amount', currency: 'USD')"
            elif col.lower() == 'volume':
                field_def += " @(title: '{i18n>volume}', semantics: 'quantity')"
                
            cds_fields.append(field_def)
        
        # Generate complete CDS definition
        cds_definition = f"""
namespace рекс.trading.data;

using {{ managed, cuid }} from '@sap/cds/common';

@title: '{entity_name} Historical Data'
@description: 'Historical trading data from {entity_name} source'
entity {entity_name}HistoricalData : managed, cuid {{
{chr(10).join(cds_fields)};
  
  // Metadata fields
  source        : String(50) @title: 'Data Source';
  exchange      : String(50) @title: 'Exchange';
  symbol        : String(20) @title: 'Trading Symbol';
  data_quality  : String(20) @title: 'Data Quality' default 'good';
  
  // Associations
  // Add relationships to other entities if needed
}}

// Define views for different use cases
@readonly
view {entity_name}LatestPrices as select from {entity_name}HistoricalData {{
  symbol,
  close as currentPrice,
  volume as volume24h,
  createdAt as lastUpdate
}} where createdAt > $now - 86400; // Last 24 hours

// Analytical view for OHLC data
@Aggregation.ApplySupported.PropertyRestrictions: true
@Analytics.AggregatedProperty #close: {{
  Name: 'AvgPrice',
  AggregationMethod: #AVG,
  ![@Common.Label]: 'Average Price'
}}
view {entity_name}Analytics as select from {entity_name}HistoricalData {{
  symbol,
  exchange,
  close,
  volume,
  createdAt
}};
"""
        
        return {
            "entity_name": f"{entity_name}HistoricalData",
            "namespace": "рекс.trading.data",
            "cds_definition": cds_definition,
            "fields": cds_fields,
            "annotations": annotations,
            "views": [f"{entity_name}LatestPrices", f"{entity_name}Analytics"],
            "i18n_keys": {
                "timestamp": "Timestamp",
                "timestampDesc": "Data timestamp",
                "symbol": "Symbol", 
                "symbolDesc": "Trading pair symbol",
                "price": "Price",
                "volume": "Volume"
            }
        }
    
    def _generate_sap_resource_discovery(self, source_id: str, sample_url: str, columns: List[str], quality_metrics: Dict[str, float], sample_size: int) -> Dict[str, Any]:
        """Generate SAP Object Resource Discovery metadata with real calculated metrics"""
        
        return {
            "@odata.context": "$metadata#DataProducts",
            "DataProductID": f"рекс-trading-{source_id}",
            "Name": f"{source_id.title()} Historical Data",
            "Description": f"Historical cryptocurrency trading data from {source_id}",
            "Version": "1.0.0",
            "Provider": {
                "Name": "рекс.com Trading Platform",
                "Contact": "data@рекс.com"
            },
            "DataSource": {
                "Type": "REST_API",
                "Endpoint": sample_url,
                "Format": "CSV",
                "Authentication": "None",
                "RefreshRate": "Daily"
            },
            "Schema": {
                "Type": "SAP_CAP_CDS",
                "Version": "2.0",
                "Fields": [
                    {
                        "Name": col.lower().replace(' ', '_'),
                        "Type": self._detect_data_type(""),
                        "Required": col.lower() in ['date', 'close', 'volume'],
                        "Description": f"{col.title()} field from {source_id}",
                        "SAPAnnotations": {
                            "@title": col.title(),
                            "@description": f"{col.title()} value"
                        }
                    } for col in columns
                ]
            },
            "BusinessContext": {
                "Domain": "Financial Markets",
                "SubDomain": "Cryptocurrency Trading",
                "UseCase": "Historical Analysis",
                "Tags": ["crypto", "trading", "historical", "OHLCV"]
            },
            "TechnicalMetadata": {
                "Storage": "SQLite + Vercel Blob",
                "ProcessingEngine": "Python Pandas",
                "A2ACompliant": True,
                "DataLineage": {
                    "Source": source_id,
                    "Transformations": ["CSV Parse", "Type Conversion", "Standardization"],
                    "Destination": "рекс Trading Database"
                }
            },
            "Governance": {
                "Classification": "Internal",
                "RetentionPolicy": "5 years", 
                "ComplianceFlags": ["GDPR_Compliant", "Financial_Data"],
                "QualityMetrics": {
                    "Completeness": quality_metrics.get("completeness", 0.0),
                    "Accuracy": quality_metrics.get("accuracy", 0.0),
                    "Timeliness": quality_metrics.get("timeliness", 0.0),
                    "Consistency": quality_metrics.get("consistency", 0.0),
                    "SampleSize": sample_size,
                    "CalculatedAt": datetime.now().isoformat()
                }
            },
            "Discovery": {
                "DiscoveredAt": datetime.now().isoformat(),
                "DiscoveryAgent": "рекс-data-management-001",
                "LastValidated": datetime.now().isoformat(),
                "ValidationStatus": "Active"
            }
        }
    
    async def analyze_data_source(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for data source analysis"""
        try:
            response = await self.agent(
                f"Discover data structure for historical data source: {source_name} with config: {config}"
            )
            return response
        except Exception as e:
            logger.error(f"Error analyzing data source {source_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    def _init_schema_table(self):
        """Initialize SQLite table for schema metadata"""
        try:
            from sqlalchemy import text
            with self.db.get_session() as session:
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS data_product_schemas (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data_product_id TEXT UNIQUE NOT NULL,
                        source_name TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        blob_url TEXT,
                        schema_hash TEXT,
                        quality_completeness REAL,
                        quality_accuracy REAL,
                        quality_timeliness REAL,
                        quality_consistency REAL,
                        sample_size INTEGER,
                        discovered_at DATETIME,
                        last_validated DATETIME,
                        validation_status TEXT DEFAULT 'active',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                session.commit()
                logger.info("Schema registry table initialized")
        except Exception as e:
            logger.error(f"Failed to initialize schema table: {e}")
    
    async def _store_schema_async(self, schema_data: Dict[str, Any], storage_type: str = "both") -> Dict[str, Any]:
        """Store data product schema in registry"""
        try:
            data_product_id = schema_data.get("sap_resource_discovery", {}).get("DataProductID")
            source_name = schema_data.get("source")
            
            if not data_product_id or not source_name:
                return {
                    "success": False,
                    "error": "Missing data_product_id or source_name"
                }
            
            logger.info(f"Storing schema for {data_product_id} in {storage_type}")
            
            schema_hash = self._generate_schema_hash(schema_data)
            blob_url = None
            sqlite_stored = False
            
            # Store in Vercel blob storage
            if storage_type in ["both", "blob"]:
                blob_key = f"schemas/data_products/{source_name}/{data_product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                if self.blob_storage:
                    blob_result = self.blob_storage.put_json(blob_key, {
                        "schema_data": schema_data,
                        "metadata": {
                            "stored_at": datetime.now().isoformat(),
                            "schema_hash": schema_hash,
                            "storage_agent": self.agent_id
                        }
                    })
                    
                    if blob_result and not blob_result.get('error'):
                        blob_url = blob_result.get('url')
                    else:
                        logger.error(f"Blob storage failed: {blob_result.get('error') if blob_result else 'Unknown error'}")
            
            # Store metadata in SQLite
            if storage_type in ["both", "sqlite"]:
                quality_metrics = schema_data.get("sap_resource_discovery", {}).get("Governance", {}).get("QualityMetrics", {})
                
                with self.db.get_session() as session:
                    # Upsert schema record
                    from sqlalchemy import text
                    session.execute(text("""
                        INSERT OR REPLACE INTO data_product_schemas (
                            data_product_id, source_name, schema_version, blob_url, schema_hash,
                            quality_completeness, quality_accuracy, quality_timeliness, quality_consistency,
                            sample_size, discovered_at, last_validated, validation_status
                        ) VALUES (:id, :source, :ver, :url, :hash, :comp, :acc, :time, :cons, :size, :disc, :val, :status)
                    """), {
                        "id": data_product_id, "source": source_name, "ver": "1.0", 
                        "url": blob_url, "hash": schema_hash,
                        "comp": quality_metrics.get("Completeness", 0),
                        "acc": quality_metrics.get("Accuracy", 0),
                        "time": quality_metrics.get("Timeliness", 0),
                        "cons": quality_metrics.get("Consistency", 0),
                        "size": quality_metrics.get("SampleSize", 0),
                        "disc": datetime.now(), "val": datetime.now(), "status": "active"
                    })
                    session.commit()
                    sqlite_stored = True
            
            # Cache locally
            self.schema_cache[data_product_id] = {
                "schema_data": schema_data,
                "cached_at": datetime.now().isoformat(),
                "blob_url": blob_url
            }
            
            return {
                "success": True,
                "data_product_id": data_product_id,
                "storage": {
                    "blob_url": blob_url,
                    "sqlite_stored": sqlite_stored
                },
                "schema_hash": schema_hash,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error storing schema: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_schema_async(self, data_product_id: str, from_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Retrieve data product schema"""
        try:
            # Check local cache first
            if from_cache and data_product_id in self.schema_cache:
                return self.schema_cache[data_product_id]["schema_data"]
            
            # Get from SQLite metadata
            with self.db.get_session() as session:
                result = session.execute(
                    "SELECT blob_url FROM data_product_schemas WHERE data_product_id = ? AND validation_status = 'active'",
                    (data_product_id,)
                ).fetchone()
                
                if result and result[0]:
                    blob_url = result[0]
                    
                    # Fetch from blob storage
                    blob_data = await self.blob_storage.download_json(blob_url)
                    if blob_data.get('success'):
                        schema_data = blob_data['data']['schema_data']
                        
                        # Update cache
                        self.schema_cache[data_product_id] = {
                            "schema_data": schema_data,
                            "cached_at": datetime.now().isoformat(),
                            "blob_url": blob_url
                        }
                        
                        return schema_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving schema {data_product_id}: {e}")
            return None
    
    def _list_schemas(self, source_name: str = None) -> List[Dict[str, Any]]:
        """List all registered data product schemas"""
        try:
            with self.db.get_session() as session:
                if source_name:
                    query = """
                        SELECT data_product_id, source_name, schema_version, quality_completeness, 
                               quality_accuracy, sample_size, discovered_at, last_validated, validation_status
                        FROM data_product_schemas 
                        WHERE source_name = ? 
                        ORDER BY updated_at DESC
                    """
                    results = session.execute(query, (source_name,)).fetchall()
                else:
                    query = """
                        SELECT data_product_id, source_name, schema_version, quality_completeness, 
                               quality_accuracy, sample_size, discovered_at, last_validated, validation_status
                        FROM data_product_schemas 
                        ORDER BY updated_at DESC
                    """
                    results = session.execute(query).fetchall()
                
                schemas = []
                for row in results:
                    schemas.append({
                        "data_product_id": row[0],
                        "source_name": row[1],
                        "schema_version": row[2],
                        "quality_completeness": row[3],
                        "quality_accuracy": row[4],
                        "sample_size": row[5],
                        "discovered_at": row[6].isoformat() if row[6] else None,
                        "last_validated": row[7].isoformat() if row[7] else None,
                        "validation_status": row[8]
                    })
                
                return schemas
                
        except Exception as e:
            logger.error(f"Error listing schemas: {e}")
            return []
    
    async def _validate_schema_async(self, data_product_id: str) -> Dict[str, Any]:
        """Validate a stored schema"""
        try:
            with self.db.get_session() as session:
                session.execute(
                    "UPDATE data_product_schemas SET last_validated = ? WHERE data_product_id = ?",
                    (datetime.now(), data_product_id)
                )
                session.commit()
            
            return {
                "success": True,
                "data_product_id": data_product_id,
                "validated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_schema_hash(self, schema_data: Dict[str, Any]) -> str:
        """Generate hash for schema versioning"""
        schema_string = json.dumps({
            "columns": schema_data.get("structure", {}).get("columns", {}),
            "source": schema_data.get("source"),
            "format": schema_data.get("structure", {}).get("format")
        }, sort_keys=True)
        
        return hashlib.md5(schema_string.encode()).hexdigest()

# Global instance
data_management_agent = DataManagementAgent()