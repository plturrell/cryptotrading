#!/usr/bin/env python3
"""
Test Implementation for UC001: CryptoDataDownload Schema Discovery
ISO/IEC/IEEE 29148:2018 Compliant Test Suite
SAP TDD-UC-001 Standard

This test suite validates the complete schema discovery workflow for CryptoDataDownload
including SAP CAP CDS generation and schema registry operations.
"""

import asyncio
import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from рекс.a2a.agents.data_management_agent import DataManagementAgent
from рекс.a2a.protocols import MessageType, A2AProtocol

class TestUC001CryptoDataDownloadSchemaDiscovery:
    """Test class for UC001: CryptoDataDownload Schema Discovery"""
    
    @pytest.fixture
    def setup_agent(self):
        """Setup Data Management Agent for testing"""
        # Mock database and storage
        with patch('рекс.a2a.agents.data_management_agent.get_db') as mock_db, \
             patch('рекс.a2a.agents.data_management_agent.VercelBlobStorage') as mock_blob:
            
            # Setup mock database
            mock_session = Mock()
            mock_db.return_value.get_session.return_value.__enter__.return_value = mock_session
            
            # Setup mock blob storage
            mock_blob_instance = Mock()
            mock_blob_instance.upload_json = AsyncMock(return_value={
                'success': True,
                'url': 'https://blob.vercel.com/test-schema.json'
            })
            mock_blob.return_value = mock_blob_instance
            
            agent = DataManagementAgent()
            agent.db = mock_db.return_value
            agent.blob_storage = mock_blob_instance
            
            yield agent, mock_session, mock_blob_instance
    
    @pytest.fixture
    def sample_csv_response(self):
        """Sample CSV response from CryptoDataDownload"""
        return """CryptoDataDownload.com - Daily Bitcoin data
unix,date,symbol,open,high,low,close,Volume BTC,Volume USD
1640995200,2022-01-01,BTC/USD,47686.81,47936.14,47169.37,47738.22,13015.79,620996966.49
1641081600,2022-01-02,BTC/USD,47738.22,47944.87,47311.41,47345.31,9829.51,466782275.83
1641168000,2022-01-03,BTC/USD,47345.31,47596.17,45704.03,46430.37,24083.48,1125686296.60"""
    
    @pytest.mark.asyncio
    async def test_ts001_valid_discovery_request(self, setup_agent, sample_csv_response):
        """
        Test Scenario TS001: Valid discovery request
        Expected: Schema stored successfully
        Code Link: src/рекс/a2a/agents/data_management_agent.py#L35-61
        """
        agent, mock_session, mock_blob = setup_agent
        
        # Mock HTTP request to CryptoDataDownload
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = sample_csv_response
            mock_get.return_value = mock_response
            
            # Execute discovery
            result = await agent.agent(
                "Discover data structure for historical data source: cryptodatadownload with config: "
                '{"exchange": "binance", "pair": "BTCUSDT", "timeframe": "d"}'
            )
            
            # Parse result
            result_data = json.loads(result) if isinstance(result, str) else result
            
            # Assertions based on UC001 postconditions
            assert result_data['success'] == True
            assert result_data['source'] == 'cryptodatadownload'
            assert 'sap_cap_schema' in result_data
            assert 'sap_resource_discovery' in result_data
            
            # Verify SAP CAP schema structure
            cap_schema = result_data['sap_cap_schema']
            assert cap_schema['entity_name'] == 'CryptoDataDownloadHistoricalData'
            assert cap_schema['namespace'] == 'рекс.trading.data'
            assert 'cds_definition' in cap_schema
            
            # Verify quality metrics are calculated
            quality_metrics = result_data['sap_resource_discovery']['Governance']['QualityMetrics']
            assert 0 <= quality_metrics['Completeness'] <= 1
            assert 0 <= quality_metrics['Accuracy'] <= 1
            assert quality_metrics['SampleSize'] == 3  # 3 data rows in sample
            
            # Verify columns discovered
            columns = result_data['structure']['columns']
            expected_columns = ['unix', 'date', 'symbol', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']
            assert all(col in columns for col in expected_columns)
    
    @pytest.mark.asyncio
    async def test_ts002_network_timeout(self, setup_agent):
        """
        Test Scenario TS002: Network timeout
        Expected: Graceful error handling
        Code Link: src/рекс/a2a/agents/data_management_agent.py#L139-147
        """
        agent, _, _ = setup_agent
        
        # Mock network timeout
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network timeout")
            
            result = await agent.agent(
                "Discover data structure for historical data source: cryptodatadownload with config: "
                '{"exchange": "binance", "pair": "BTCUSDT", "timeframe": "d"}'
            )
            
            result_data = json.loads(result) if isinstance(result, str) else result
            
            # Assertions
            assert result_data['success'] == False
            assert 'error' in result_data
            assert 'Network timeout' in result_data['error']
    
    @pytest.mark.asyncio
    async def test_ts003_malformed_csv(self, setup_agent):
        """
        Test Scenario TS003: Malformed CSV
        Expected: Error with details
        """
        agent, _, _ = setup_agent
        
        # Mock malformed CSV response
        malformed_csv = "This is not a valid CSV format\nRandom text"
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = malformed_csv
            mock_get.return_value = mock_response
            
            result = await agent.agent(
                "Discover data structure for historical data source: cryptodatadownload with config: "
                '{"exchange": "binance", "pair": "BTCUSDT", "timeframe": "d"}'
            )
            
            result_data = json.loads(result) if isinstance(result, str) else result
            
            # Should handle gracefully
            assert 'structure' in result_data or result_data['success'] == False
    
    @pytest.mark.asyncio
    async def test_ts004_concurrent_requests(self, setup_agent, sample_csv_response):
        """
        Test Scenario TS004: Concurrent requests
        Expected: No race conditions
        """
        agent, mock_session, _ = setup_agent
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = sample_csv_response
            mock_get.return_value = mock_response
            
            # Execute multiple concurrent discoveries
            tasks = []
            for i in range(5):
                task = agent.agent(
                    f"Discover data structure for historical data source: cryptodatadownload with config: "
                    f'{{"exchange": "binance", "pair": "BTCUSDT{i}", "timeframe": "d"}}'
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed without conflicts
            for result in results:
                result_data = json.loads(result) if isinstance(result, str) else result
                assert result_data['success'] == True
    
    @pytest.mark.asyncio
    async def test_ts005_schema_storage_and_retrieval(self, setup_agent, sample_csv_response):
        """
        Test Scenario TS005: Schema storage and retrieval
        Expected: Schema stored and retrieved correctly
        Code Link: src/рекс/a2a/agents/data_management_agent.py#L577-697
        """
        agent, mock_session, mock_blob = setup_agent
        
        # First discover and store schema
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = sample_csv_response
            mock_get.return_value = mock_response
            
            # Discover schema
            discovery_result = await agent.agent(
                "Discover data structure for historical data source: cryptodatadownload with config: "
                '{"exchange": "binance", "pair": "BTCUSDT", "timeframe": "d"}'
            )
            
            discovery_data = json.loads(discovery_result) if isinstance(discovery_result, str) else discovery_result
            
            # Store schema
            store_result = await agent.agent(
                f"Store schema with data: {json.dumps(discovery_data)} and storage_type: both"
            )
            
            store_data = json.loads(store_result) if isinstance(store_result, str) else store_result
            
            # Assertions for storage
            assert store_data['success'] == True
            assert 'data_product_id' in store_data
            assert store_data['storage']['blob_url'] is not None
            
            # Test retrieval
            data_product_id = store_data['data_product_id']
            
            # Mock blob download
            mock_blob.download_json = AsyncMock(return_value={
                'success': True,
                'data': {
                    'schema_data': discovery_data
                }
            })
            
            retrieve_result = await agent.agent(
                f"Get schema for data_product_id: {data_product_id}"
            )
            
            retrieve_data = json.loads(retrieve_result) if isinstance(retrieve_result, str) else retrieve_result
            
            # Verify retrieved schema matches original
            assert retrieve_data['source'] == 'cryptodatadownload'
            assert 'sap_cap_schema' in retrieve_data
    
    def test_business_rule_br001_performance(self, setup_agent, sample_csv_response):
        """
        Test Business Rule BR001: Schema discovery must complete within 30 seconds
        """
        import time
        
        agent, _, _ = setup_agent
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = sample_csv_response
            mock_get.return_value = mock_response
            
            start_time = time.time()
            
            # Run synchronously for timing
            result = asyncio.run(agent.agent(
                "Discover data structure for historical data source: cryptodatadownload with config: "
                '{"exchange": "binance", "pair": "BTCUSDT", "timeframe": "d"}'
            ))
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Assert performance requirement
            assert execution_time < 30.0, f"Discovery took {execution_time}s, exceeding 30s limit"
    
    def test_business_rule_br002_minimum_rows(self, setup_agent):
        """
        Test Business Rule BR002: Quality metrics must be calculated from minimum 10 data rows
        """
        # Create CSV with exactly 10 data rows
        csv_with_10_rows = """CryptoDataDownload.com - Daily Bitcoin data
unix,date,symbol,open,high,low,close,Volume BTC,Volume USD
""" + "\n".join([
            f"{1641168000 + i*86400},2022-01-{3+i:02d},BTC/USD,{47000+i*100},{47100+i*100},{46900+i*100},{47050+i*100},{1000+i*10},{47000000+i*1000000}"
            for i in range(10)
        ])
        
        agent, _, _ = setup_agent
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = csv_with_10_rows
            mock_get.return_value = mock_response
            
            result = asyncio.run(agent.agent(
                "Discover data structure for historical data source: cryptodatadownload with config: "
                '{"exchange": "binance", "pair": "BTCUSDT", "timeframe": "d"}'
            ))
            
            result_data = json.loads(result) if isinstance(result, str) else result
            
            # Verify quality metrics calculated
            quality_metrics = result_data['sap_resource_discovery']['Governance']['QualityMetrics']
            assert quality_metrics['SampleSize'] == 10
            assert all(0 <= quality_metrics[key] <= 1 for key in ['Completeness', 'Accuracy', 'Timeliness', 'Consistency'])
    
    def test_code_linkage_verification(self):
        """
        Verify all code links in UC001 document are valid
        """
        import os
        import re
        
        # Read UC001 document
        uc001_path = Path(__file__).parent / "UC001_CryptoDataDownload_Schema_Discovery.md"
        with open(uc001_path, 'r') as f:
            uc001_content = f.read()
        
        # Extract all code links
        code_links = re.findall(r'Code Link: `([^`]+)`', uc001_content)
        
        # Verify each code link
        for link in code_links:
            if '#' in link:
                file_path, line_range = link.split('#')
                file_full_path = Path(__file__).parent.parent.parent.parent.parent / file_path
                
                # Check file exists
                assert file_full_path.exists(), f"Code file not found: {file_path}"
                
                # Verify line range format (e.g., L78-99)
                assert re.match(r'L\d+-\d+', line_range), f"Invalid line range format: {line_range}"

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])