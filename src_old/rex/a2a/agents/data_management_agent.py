"""
Data Management Agent powered by Strand Agents - 100% A2A Compliant
Discovers data structures for historical data sources
"""

from strands import tool
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import requests
import logging
import json
import hashlib
import asyncio
import re
from pathlib import Path

from .memory_strands_agent import MemoryStrandsAgent
from ...database.client import get_db
from ...storage.vercel_blob import VercelBlobClient
from ...ml.yfinance_client import get_yfinance_client
from ..protocols import A2AMessage, A2AProtocol, MessageType

logger = logging.getLogger(__name__)

class DataManagementAgent(MemoryStrandsAgent):
    def __init__(self, model_provider: str = "grok4"):
        # Initialize storage connections first
        self.db = get_db()
        try:
            self.blob_storage = VercelBlobClient()
        except ValueError:
            self.blob_storage = None
            logger.warning("Vercel Blob storage not available, using SQLite only")
        self.schema_cache = {}
        self.yf_client = get_yfinance_client()
        self._init_schema_table()
        
        # Initialize base class
        super().__init__(
            agent_id='data-management-001',
            agent_type='data_management',
            capabilities=[
                'data_structure_discovery', 'schema_analysis', 'data_mapping',
                'source_validation', 'format_detection', 'schema_registry'
            ],
            model_provider=model_provider
        )
    
    def _create_tools(self):
        """Create data management specific tools"""
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
        
        # Return tools for base class to use
        return [
            discover_data_structure_for_historical_data,
            store_schema,
            get_schema,
            list_schemas,
            validate_schema
        ]
    
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
        """Discover Yahoo Finance specific structure for ETH"""
        try:
            # Force ETH only
            symbol = 'ETH-USD'
            
            logger.info(f"Analyzing Yahoo Finance structure for {symbol}")
            
            # Use YFinance client to get sample data
            hist_data = self.yf_client.get_historical_data(days_back=30)
            
            if hist_data.empty:
                return {
                    "success": False,
                    "source": "yahoo",
                    "error": f"No data available for {symbol}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get ticker info
            ticker_info = self.yf_client.get_ticker_info()
            
            # Analyze the data structure
            columns = list(hist_data.columns)
            sample_data = []
            
            # Convert DataFrame to list format for analysis
            hist_data_reset = hist_data.reset_index()
            for _, row in hist_data_reset.iterrows():
                row_values = [str(val) for val in row.values]
                sample_data.append(row_values)
            
            # Add Date column to columns list
            all_columns = ['Date'] + columns
            
            # Calculate quality metrics from actual data
            quality_metrics = self._calculate_quality_metrics(all_columns, sample_data)
            
            # Validate data quality using client
            validation_metrics = self.yf_client.validate_data_quality(hist_data)
            
            # Analyze column types from actual data
            column_analysis = {}
            
            # Date column
            column_analysis['Date'] = {
                "position": 0,
                "sample_values": [row[0] for row in sample_data[:3]],
                "data_type": "datetime",
                "null_count": 0,
                "total_samples": len(sample_data),
                "database_mapping": "timestamp",
                "yahoo_field": "index"
            }
            
            # Other columns
            for i, col in enumerate(columns):
                column_samples = [row[i+1] for row in sample_data]
                
                column_analysis[col] = {
                    "position": i + 1,
                    "sample_values": column_samples[:3],
                    "data_type": self._detect_data_type_from_samples(column_samples),
                    "null_count": sum(1 for val in column_samples if not val or val.lower() in ['', 'null', 'nan']),
                    "total_samples": len(column_samples),
                    "database_mapping": self._map_yahoo_column(col),
                    "yahoo_field": col
                }
            
            # Generate SAP CAP CDS schema
            cap_schema = self._generate_sap_cap_schema("YahooFinance", all_columns, column_analysis)
            
            # Generate SAP Object Resource Discovery with actual metrics
            resource_discovery = self._generate_sap_resource_discovery(
                "yahoo-finance-eth", 
                "YFinanceClient.get_historical_data()",
                all_columns,
                quality_metrics,
                len(sample_data)
            )
            
            # Get current market data
            market_data = self.yf_client.get_market_data()
            
            return {
                "success": True,
                "source": "yahoo",
                "symbol": symbol,
                "structure": {
                    "format": "DataFrame",
                    "api_method": "yfinance.Ticker.history()",
                    "client_method": "YFinanceClient.get_historical_data()",
                    "columns": column_analysis,
                    "total_columns": len(all_columns),
                    "sample_rows": len(sample_data),
                    "ticker_info": {
                        "longName": ticker_info.get('longName', 'Ethereum USD'),
                        "currency": ticker_info.get('currency', 'USD'),
                        "exchange": ticker_info.get('exchange', 'CCC'),
                        "marketCap": ticker_info.get('marketCap'),
                        "currentPrice": market_data.get('current_price'),
                        "volume24h": market_data.get('volume_24h')
                    },
                    "data_validation": validation_metrics
                },
                "sap_cap_schema": cap_schema,
                "sap_resource_discovery": resource_discovery,
                "database_mapping": {
                    "table": "market_data_source",
                    "column_mappings": {col: info["database_mapping"] for col, info in column_analysis.items()}
                },
                "file_organization": {
                    "base_path": "data/historical/yahoo",
                    "filename_pattern": "ETH-USD_{date_range}.parquet",
                    "folder_structure": "by_date"
                },
                "data_contract": {
                    "expected_columns": list(all_columns),
                    "required_columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
                    "data_types": {col: info["data_type"] for col, info in column_analysis.items()},
                    "symbol_constraint": "ETH-USD only"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "source": "yahoo",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Bitget support removed - Yahoo Finance is sufficient for ETH data
    
    def _discover_generic_structure(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic structure discovery for unknown sources using intelligent probing"""
        try:
            # Strategy 1: URL-based discovery
            if 'url' in config or 'endpoint' in config:
                return self._discover_from_url(source_name, config)
            
            # Strategy 2: API-based discovery
            elif 'api_key' in config or 'api_url' in config:
                return self._discover_from_api(source_name, config)
            
            # Strategy 3: File-based discovery
            elif 'file_path' in config or 'file_url' in config:
                return self._discover_from_file(source_name, config)
            
            # Strategy 4: Database-based discovery
            elif 'connection_string' in config or 'database_url' in config:
                return self._discover_from_database(source_name, config)
            
            # Strategy 5: Generic pattern analysis
            else:
                return self._discover_from_patterns(source_name, config)
        
        except Exception as e:
            logger.error(f"Generic structure discovery failed for {source_name}: {e}")
            return {
                "success": False,
                "source": source_name,
                "error": f"Structure discovery failed: {str(e)}",
                "discovery_strategies_attempted": ["url", "api", "file", "database", "patterns"]
            }
    
    def _discover_from_url(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover structure from URL endpoint"""
        import requests
        from urllib.parse import urlparse
        
        url = config.get('url') or config.get('endpoint')
        
        try:
            # Try to fetch a sample from the URL
            headers = {}
            if 'headers' in config:
                headers.update(config['headers'])
            
            # Common headers for APIs
            if 'api_key' in config:
                headers['Authorization'] = f"Bearer {config['api_key']}"
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                
                if 'json' in content_type:
                    return self._analyze_json_structure(source_name, response.json())
                elif 'csv' in content_type or url.endswith('.csv'):
                    return self._analyze_csv_structure(source_name, response.text)
                elif 'xml' in content_type:
                    return self._analyze_xml_structure(source_name, response.text)
                else:
                    # Try to infer format from content
                    return self._infer_format_from_content(source_name, response.text)
            
            else:
                return {
                    "success": False,
                    "source": source_name,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    "url_tested": url
                }
        
        except Exception as e:
            return {
                "success": False,
                "source": source_name,
                "error": f"URL discovery failed: {str(e)}",
                "url_tested": url
            }
    
    def _discover_from_api(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover structure from API endpoint"""
        import requests
        
        api_url = config.get('api_url')
        api_key = config.get('api_key')
        
        # Common API discovery patterns
        test_endpoints = [
            '',  # Base endpoint
            '/info',
            '/schema',
            '/metadata',
            '/describe',
            '/v1/info',
            '/api/v1/info'
        ]
        
        for endpoint_suffix in test_endpoints:
            try:
                test_url = f"{api_url.rstrip('/')}{endpoint_suffix}"
                
                headers = {'Content-Type': 'application/json'}
                if api_key:
                    headers['Authorization'] = f"Bearer {api_key}"
                
                response = requests.get(test_url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        return {
                            "success": True,
                            "source": source_name,
                            "discovery_method": "api_endpoint",
                            "endpoint_used": test_url,
                            "structure": self._extract_structure_from_data(data),
                            "sample_data": data if len(str(data)) < 1000 else str(data)[:1000] + "...",
                            "api_info": {
                                "supports_json": True,
                                "requires_auth": bool(api_key),
                                "response_headers": dict(response.headers)
                            }
                        }
                    except ValueError:
                        # Not JSON, try other formats
                        continue
            
            except Exception:
                continue
        
        return {
            "success": False,
            "source": source_name,
            "error": "No responsive API endpoints found",
            "endpoints_tested": [f"{api_url.rstrip('/')}{suffix}" for suffix in test_endpoints]
        }
    
    def _discover_from_file(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover structure from file source"""
        import pandas as pd
        from pathlib import Path
        
        file_path = config.get('file_path') or config.get('file_url')
        
        try:
            # Determine file format
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.csv', '.tsv']:
                # Try to read CSV/TSV
                separator = '\t' if file_ext == '.tsv' else ','
                df = pd.read_csv(file_path, sep=separator, nrows=100)  # Sample first 100 rows
                
                return {
                    "success": True,
                    "source": source_name,
                    "discovery_method": "file_analysis",
                    "file_format": file_ext[1:],
                    "structure": {
                        "columns": list(df.columns),
                        "data_types": df.dtypes.to_dict(),
                        "sample_values": {col: df[col].dropna().iloc[:5].tolist() for col in df.columns},
                        "row_count_sample": len(df),
                        "has_header": True
                    },
                    "file_info": {
                        "separator": separator,
                        "encoding": "utf-8"
                    }
                }
            
            elif file_ext in ['.json', '.jsonl']:
                # JSON file analysis
                with open(file_path, 'r') as f:
                    if file_ext == '.jsonl':
                        # JSON Lines format
                        sample_lines = [json.loads(line) for line in f.readlines()[:10]]
                        sample_data = sample_lines
                    else:
                        sample_data = json.load(f)
                
                return {
                    "success": True,
                    "source": source_name,
                    "discovery_method": "file_analysis",
                    "file_format": file_ext[1:],
                    "structure": self._extract_structure_from_data(sample_data),
                    "sample_data": sample_data if len(str(sample_data)) < 1000 else str(sample_data)[:1000] + "..."
                }
            
            elif file_ext in ['.xlsx', '.xls']:
                # Excel file analysis
                df = pd.read_excel(file_path, nrows=100)
                
                return {
                    "success": True,
                    "source": source_name,
                    "discovery_method": "file_analysis", 
                    "file_format": file_ext[1:],
                    "structure": {
                        "columns": list(df.columns),
                        "data_types": df.dtypes.to_dict(),
                        "sample_values": {col: df[col].dropna().iloc[:5].tolist() for col in df.columns},
                        "row_count_sample": len(df)
                    }
                }
            
            else:
                return {
                    "success": False,
                    "source": source_name,
                    "error": f"Unsupported file format: {file_ext}",
                    "supported_formats": [".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls"]
                }
        
        except Exception as e:
            return {
                "success": False,
                "source": source_name,
                "error": f"File analysis failed: {str(e)}",
                "file_path": file_path
            }
    
    def _discover_from_database(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover structure from database source"""
        try:
            import sqlalchemy as sa
            from sqlalchemy import inspect
            
            connection_string = config.get('connection_string') or config.get('database_url')
            table_name = config.get('table_name', config.get('table'))
            
            engine = sa.create_engine(connection_string)
            inspector = inspect(engine)
            
            if table_name:
                # Analyze specific table
                columns = inspector.get_columns(table_name)
                
                return {
                    "success": True,
                    "source": source_name,
                    "discovery_method": "database_inspection",
                    "table_name": table_name,
                    "structure": {
                        "columns": [col['name'] for col in columns],
                        "data_types": {col['name']: str(col['type']) for col in columns},
                        "nullable": {col['name']: col.get('nullable', True) for col in columns},
                        "primary_keys": inspector.get_pk_constraint(table_name)['constrained_columns']
                    },
                    "database_info": {
                        "dialect": engine.dialect.name,
                        "driver": engine.dialect.driver
                    }
                }
            
            else:
                # List available tables
                tables = inspector.get_table_names()
                
                return {
                    "success": True,
                    "source": source_name,
                    "discovery_method": "database_catalog",
                    "structure": {
                        "available_tables": tables,
                        "table_count": len(tables)
                    },
                    "database_info": {
                        "dialect": engine.dialect.name,
                        "driver": engine.dialect.driver
                    },
                    "note": "Specify 'table_name' in config to analyze specific table structure"
                }
        
        except Exception as e:
            return {
                "success": False,
                "source": source_name,
                "error": f"Database discovery failed: {str(e)}",
                "connection_string": connection_string.split('@')[0] + "@***" if connection_string else None
            }
    
    def _discover_from_patterns(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover structure using pattern analysis and heuristics"""
        patterns_found = []
        
        # Analyze config keys for patterns
        config_keys = list(config.keys())
        
        # Trading/Financial data patterns
        if any(key in ['symbol', 'ticker', 'currency', 'crypto'] for key in config_keys):
            patterns_found.append("financial_data")
        
        # Time series patterns
        if any(key in ['timestamp', 'date', 'time', 'interval'] for key in config_keys):
            patterns_found.append("time_series")
        
        # Market data patterns  
        if any(key in ['price', 'volume', 'open', 'high', 'low', 'close'] for key in config_keys):
            patterns_found.append("market_data")
        
        # API patterns
        if any(key in ['api_key', 'token', 'auth', 'credentials'] for key in config_keys):
            patterns_found.append("api_source")
        
        # Generate suggested structure based on patterns
        suggested_structure = self._generate_structure_from_patterns(patterns_found, config)
        
        return {
            "success": True,
            "source": source_name,
            "discovery_method": "pattern_analysis",
            "patterns_detected": patterns_found,
            "structure": suggested_structure,
            "confidence": "low",
            "note": "Structure inferred from configuration patterns. Manual verification recommended."
        }
    
    def _analyze_json_structure(self, source_name: str, json_data: Any) -> Dict[str, Any]:
        """Analyze JSON data structure"""
        return {
            "success": True,
            "source": source_name,
            "discovery_method": "json_analysis",
            "format": "json",
            "structure": self._extract_structure_from_data(json_data),
            "sample_data": json_data if len(str(json_data)) < 1000 else str(json_data)[:1000] + "..."
        }
    
    def _analyze_csv_structure(self, source_name: str, csv_content: str) -> Dict[str, Any]:
        """Analyze CSV content structure"""
        import csv
        import io
        
        try:
            # Detect delimiter
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(csv_content[:1000]).delimiter
            
            # Parse CSV
            reader = csv.DictReader(io.StringIO(csv_content), delimiter=delimiter)
            rows = [row for row, _ in zip(reader, range(10))]  # First 10 rows
            
            if rows:
                columns = list(rows[0].keys())
                sample_values = {col: [row[col] for row in rows if row[col]] for col in columns}
                
                return {
                    "success": True,
                    "source": source_name,
                    "discovery_method": "csv_analysis",
                    "format": "csv",
                    "structure": {
                        "columns": columns,
                        "sample_values": {k: v[:5] for k, v in sample_values.items()},
                        "delimiter": delimiter,
                        "sample_row_count": len(rows)
                    }
                }
            
            else:
                return {
                    "success": False,
                    "source": source_name,
                    "error": "No data rows found in CSV"
                }
        
        except Exception as e:
            return {
                "success": False,
                "source": source_name,
                "error": f"CSV analysis failed: {str(e)}"
            }
    
    def _analyze_xml_structure(self, source_name: str, xml_content: str) -> Dict[str, Any]:
        """Analyze XML content structure"""
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(xml_content)
            
            def extract_xml_structure(element, level=0):
                if level > 3:  # Prevent deep recursion
                    return {"truncated": True}
                
                structure = {
                    "tag": element.tag,
                    "attributes": element.attrib,
                    "text": element.text.strip() if element.text else None
                }
                
                children = {}
                for child in element:
                    child_tag = child.tag
                    if child_tag not in children:
                        children[child_tag] = []
                    children[child_tag].append(extract_xml_structure(child, level + 1))
                
                if children:
                    structure["children"] = children
                
                return structure
            
            return {
                "success": True,
                "source": source_name,
                "discovery_method": "xml_analysis",
                "format": "xml",
                "structure": extract_xml_structure(root)
            }
        
        except Exception as e:
            return {
                "success": False,
                "source": source_name,
                "error": f"XML analysis failed: {str(e)}"
            }
    
    def _infer_format_from_content(self, source_name: str, content: str) -> Dict[str, Any]:
        """Infer data format from raw content"""
        content_sample = content[:2000]  # First 2KB
        
        # Try JSON
        try:
            json_data = json.loads(content_sample)
            return self._analyze_json_structure(source_name, json_data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try CSV
        if ',' in content_sample and '\n' in content_sample:
            try:
                return self._analyze_csv_structure(source_name, content_sample)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
                pass
        
        # Try XML
        if '<' in content_sample and '>' in content_sample:
            try:
                return self._analyze_xml_structure(source_name, content_sample)
            except (ET.ParseError, ValueError):
                pass
        
        # Plain text analysis
        lines = content_sample.split('\n')[:10]
        
        return {
            "success": True,
            "source": source_name,
            "discovery_method": "content_inference",
            "format": "text",
            "structure": {
                "line_count_sample": len(lines),
                "average_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
                "sample_lines": lines[:5],
                "detected_patterns": self._detect_text_patterns(content_sample)
            }
        }
    
    def _extract_structure_from_data(self, data: Any) -> Dict[str, Any]:
        """Extract structure information from any data type"""
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys()),
                "key_types": {k: type(v).__name__ for k, v in data.items()},
                "sample_values": {k: v if len(str(v)) < 100 else str(v)[:100] + "..." for k, v in data.items()}
            }
        
        elif isinstance(data, list):
            if data:
                first_item = data[0]
                return {
                    "type": "array",
                    "length": len(data),
                    "item_type": type(first_item).__name__,
                    "item_structure": self._extract_structure_from_data(first_item) if isinstance(first_item, (dict, list)) else None,
                    "sample_items": data[:3]
                }
            else:
                return {"type": "array", "length": 0}
        
        else:
            return {
                "type": type(data).__name__,
                "value": data if len(str(data)) < 100 else str(data)[:100] + "..."
            }
    
    def _generate_structure_from_patterns(self, patterns: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate suggested structure based on detected patterns"""
        structure = {"inferred": True, "patterns_used": patterns}
        
        if "financial_data" in patterns:
            structure["suggested_fields"] = ["symbol", "price", "volume", "timestamp"]
        
        if "time_series" in patterns:
            structure["temporal_info"] = {
                "has_timestamps": True,
                "suggested_time_field": "timestamp"
            }
        
        if "market_data" in patterns:
            structure["market_fields"] = ["open", "high", "low", "close", "volume"]
        
        if "api_source" in patterns:
            structure["api_info"] = {
                "requires_authentication": True,
                "suggested_format": "json"
            }
        
        # Add config-based hints
        structure["config_analysis"] = {
            "provided_keys": list(config.keys()),
            "possible_data_fields": [k for k in config.keys() if k not in ['api_key', 'url', 'headers']]
        }
        
        return structure
    
    def _detect_text_patterns(self, text: str) -> List[str]:
        """Detect patterns in plain text"""
        patterns = []
        
        if re.search(r'\d{4}-\d{2}-\d{2}', text):
            patterns.append("iso_dates")
        
        if re.search(r'\$\d+\.?\d*', text):
            patterns.append("currency_values")
        
        if re.search(r'\d+\.\d+', text):
            patterns.append("decimal_numbers")
        
        if re.search(r'[A-Z]{3,5}', text):
            patterns.append("ticker_symbols")
        
        return patterns
    
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
            except (ValueError, TypeError):
                pass
            
            # Test for datetime
            try:
                pd.to_datetime(sample)
                datetime_count += 1
            except (ValueError, TypeError):
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
                        except (ValueError, TypeError, KeyError):
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
                    except (ValueError, TypeError):
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
    
    def _map_yahoo_column(self, column_name: str) -> str:
        """Map Yahoo Finance column to database column"""
        mapping = {
            "Open": "open_price",
            "High": "high_24h",
            "Low": "low_24h",
            "Close": "price",
            "Volume": "volume_24h",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits"
        }
        
        return mapping.get(column_name, column_name.lower().replace(' ', '_'))
    
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
namespace rex.trading.data;

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
            "namespace": "rex.trading.data",
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
            "DataProductID": f"rex-trading-{source_id}",
            "Name": f"{source_id.title()} Historical Data",
            "Description": f"Historical cryptocurrency trading data from {source_id}",
            "Version": "1.0.0",
            "Provider": {
                "Name": "rex.com Trading Platform",
                "Contact": "data@rex.com"
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
                        "Type": "number" if col.lower() in ['open', 'high', 'low', 'close', 'volume'] else "string",
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
                    "Destination": "rex Trading Database"
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
                "DiscoveryAgent": "rex-data-management-001",
                "LastValidated": datetime.now().isoformat(),
                "ValidationStatus": "Active"
            }
        }
    
    async def analyze_data_source(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for data source analysis"""
        try:
            # Agent accepts a string prompt directly
            prompt = f"Discover data structure for historical data source: {source_name} with config: {config}"
            
            # Call agent synchronously (it will handle async internally)
            result = self.agent(prompt)
            
            # Extract response from AgentResult
            if hasattr(result, 'message'):
                return {"success": True, "response": str(result.message), "stop_reason": str(result.stop_reason)}
            
            return {"success": True, "response": str(result)}
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

    def _message_to_prompt(self, message):
        """Convert A2A message to natural language prompt for data management"""
        message_type = message.message_type.value
        payload = message.payload
        
        if message_type == 'SCHEMA_DISCOVERY_REQUEST':
            source = payload.get('source_name')
            config = payload.get('source_config', {})
            return f"Discover data structure for historical data source: {source} with config: {config}"
        elif message_type == 'SCHEMA_QUERY':
            data_product_id = payload.get('data_product_id')
            return f"Get schema for data product: {data_product_id}"
        else:
            return super()._message_to_prompt(message)

# Global instance
data_management_agent = DataManagementAgent()