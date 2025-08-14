# Use Case UC001: CryptoDataDownload Schema Discovery

**Document Version:** 1.0  
**ISO Standard:** ISO/IEC/IEEE 29148:2018  
**SAP Standard:** SAP TDD-UC-001  
**Created:** 2025-01-12  
**Author:** rex Trading Platform Team  
**Status:** Active  

## 1. Use Case Identification

| Attribute | Value |
|-----------|--------|
| Use Case ID | UC001 |
| Use Case Name | CryptoDataDownload Schema Discovery |
| Priority | High |
| Complexity | Medium |
| Sprint/Release | v1.0 |

## 2. Use Case Description

### 2.1 Brief Description
The Data Management Agent discovers the data structure of CryptoDataDownload historical data source, generates SAP CAP CDS schema, and stores it in the schema registry for use by other A2A agents.

### 2.2 Business Context
- **Domain:** Financial Markets / Cryptocurrency Trading
- **Sub-Domain:** Historical Data Management
- **Business Value:** Enables automated data ingestion and standardization

## 3. Actors

| Actor | Type | Description |
|-------|------|-------------|
| Data Management Agent | Primary | Discovers and manages data schemas |
| Historical Loader Agent | Secondary | Uses schema to load data |
| Database Agent | Secondary | Uses schema to store data |
| A2A Coordinator | Supporting | Orchestrates agent communication |

## 4. Preconditions

1. Data Management Agent is initialized and registered in A2A registry
2. Network connectivity to CryptoDataDownload website
3. SQLite database is accessible
4. Vercel blob storage is configured (if enabled)

## 5. Basic Flow

1. **Initiate Discovery**
   - Actor: A2A Coordinator
   - Action: Sends discovery request to Data Management Agent
   - Data: `{"source": "cryptodatadownload", "exchange": "binance", "pair": "BTCUSDT", "timeframe": "d"}`

2. **Fetch Sample Data**
   - Actor: Data Management Agent
   - Action: Downloads sample CSV from CryptoDataDownload
   - Code Link: `src/rex/a2a/agents/data_management_agent.py#L78-99`

3. **Analyze Data Structure**
   - Actor: Data Management Agent
   - Action: Analyzes columns, data types, quality metrics
   - Code Link: `src/rex/a2a/agents/data_management_agent.py#L101-113`

4. **Generate SAP CAP Schema**
   - Actor: Data Management Agent
   - Action: Creates CDS entity definition
   - Code Link: `src/rex/a2a/agents/data_management_agent.py#L209-306`

5. **Generate SAP Resource Discovery**
   - Actor: Data Management Agent
   - Action: Creates OData metadata
   - Code Link: `src/rex/a2a/agents/data_management_agent.py#L416-486`

6. **Store Schema in Registry**
   - Actor: Data Management Agent
   - Action: Saves to SQLite and/or Vercel blob
   - Code Link: `src/rex/a2a/agents/data_management_agent.py#L577-660`

## 6. Alternative Flows

### 6.1 Network Error
- **Trigger:** CryptoDataDownload website unreachable
- **Action:** Return error with cached schema if available
- **Result:** `{"success": false, "error": "Network error", "cached_schema": {...}}`

### 6.2 Invalid Data Format
- **Trigger:** CSV format changed or corrupted
- **Action:** Log error and notify administrator
- **Result:** `{"success": false, "error": "Invalid CSV format"}`

## 7. Postconditions

### 7.1 Success Postconditions
1. Schema stored in registry with unique data_product_id
2. Quality metrics calculated and stored
3. Schema available for retrieval by other agents
4. Cache updated with latest schema

### 7.2 Failure Postconditions
1. Error logged with details
2. No partial schema stored
3. Previous valid schema remains active

## 8. Data Elements

### 8.1 Input Data
```json
{
  "source": "cryptodatadownload",
  "exchange": "binance",
  "pair": "BTCUSDT",
  "timeframe": "d"
}
```

### 8.2 Output Data
```json
{
  "success": true,
  "source": "cryptodatadownload",
  "data_product_id": "rex-trading-cryptodatadownload",
  "sap_cap_schema": {
    "entity_name": "CryptoDataDownloadHistoricalData",
    "namespace": "rex.trading.data",
    "cds_definition": "...",
    "fields": ["date", "open", "high", "low", "close", "volume"]
  },
  "sap_resource_discovery": {
    "DataProductID": "rex-trading-cryptodatadownload",
    "QualityMetrics": {
      "Completeness": 0.998,
      "Accuracy": 0.995,
      "Timeliness": 1.0,
      "Consistency": 0.997
    }
  }
}
```

## 9. Business Rules

1. **BR001:** Schema discovery must complete within 30 seconds
2. **BR002:** Quality metrics must be calculated from minimum 10 data rows
3. **BR003:** Schema versioning must use MD5 hash of structure
4. **BR004:** Cached schemas expire after 7 days

## 10. Non-Functional Requirements

| Requirement | Specification |
|-------------|---------------|
| Performance | Discovery < 30 seconds |
| Availability | 99.9% uptime |
| Security | HTTPS only, no credentials stored |
| Scalability | Support 100+ data sources |
| Compliance | GDPR compliant |

## 11. Test Scenarios

| Test ID | Scenario | Expected Result |
|---------|----------|-----------------|
| TS001 | Valid discovery request | Schema stored successfully |
| TS002 | Network timeout | Graceful error handling |
| TS003 | Malformed CSV | Error with details |
| TS004 | Concurrent requests | No race conditions |
| TS005 | Schema retrieval | Correct schema returned |

## 12. Traceability Matrix

| Requirement | Use Case Step | Code Component | Test Case |
|-------------|---------------|----------------|-----------|
| Schema Discovery | Step 2-3 | `_discover_cryptodatadownload_structure` | TS001 |
| SAP CAP Generation | Step 4 | `_generate_sap_cap_schema` | TS001 |
| Storage | Step 6 | `_store_schema_async` | TS001, TS004 |
| Error Handling | Alt Flow 6.1 | Exception handlers | TS002, TS003 |

## 13. Dependencies

- **External:** CryptoDataDownload API availability
- **Internal:** SQLite database, Vercel blob storage
- **Libraries:** pandas, requests, strands-agents

## 14. Open Issues

1. Rate limiting for CryptoDataDownload API
2. Schema migration strategy for breaking changes
3. Multi-language support for i18n keys

## 15. Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Owner | | | |
| Technical Lead | | | |
| QA Lead | | | |
| Compliance Officer | | | |

---

**Document Control:**
- Review Cycle: Quarterly
- Next Review: 2025-04-12
- Distribution: Development Team, QA Team, Product Management