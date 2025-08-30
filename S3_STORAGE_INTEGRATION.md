# S3 Storage Integration - Secure Data Management

## 🔐 Security-First Approach

This S3 storage integration uses **AWS Secrets Manager** for credential management, ensuring sensitive AWS credentials are never exposed in code or version control.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│ AWS Secrets      │───▶│   S3 Bucket     │
│   Layer         │    │ Manager          │    │ tentimecrypto   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                               │
        ▼                                               ▼
┌─────────────────┐                           ┌─────────────────┐
│ CryptoData      │                           │  Organized      │
│ Manager         │                           │  Data Structure │
└─────────────────┘                           └─────────────────┘
```

## 📂 Data Organization Structure

```
s3://tentimecrypto/
├── market-data/
│   ├── ohlcv/
│   │   └── BTC-USD/
│   │       └── 1h/
│   │           └── 2024/01/15/
│   │               └── 2024-01-15T10:00:00Z.json
│   ├── orderbook/
│   │   └── BTC-USD/
│   │       └── 2024/01/15/10/
│   │           └── 30_2024-01-15T10:30:00Z.json
│   └── trades/
│       └── BTC-USD/
│           └── 2024/01/15/10/
│               └── 2024-01-15T10:15:00Z.json
├── user-data/
│   └── {user_id}/
│       ├── portfolio/
│       │   └── 2024/01/15/
│       │       └── 2024-01-15T12:00:00Z.json
│       └── trades/
│           └── 2024/01/
│               └── 2024-01-15T14:30:00Z.json
├── analytics/
│   └── {report_type}/
│       └── 2024/01/15/
│           └── 2024-01-15T16:00:00Z.json
├── backups/
│   └── database/
│       └── {table_name}/
│           └── 2024/01/15/
│               └── 2024-01-15T02:00:00Z.json
└── logs/
    └── {log_type}/
        └── 2024/01/15/
            └── 2024-01-15T00:00:00Z_app.log
```

## 🔧 Core Components

### 1. AWS Secrets Manager Integration

**File:** `src/cryptotrading/infrastructure/storage/aws_secrets_manager.py`

**Features:**
- Secure credential storage and retrieval
- Automatic credential rotation support
- Error handling and logging
- Secret lifecycle management

```python
from storage.aws_secrets_manager import SecretsManager

# Initialize and use
secrets_manager = SecretsManager(region_name="us-east-1")
credentials = secrets_manager.get_secret("cryptotrading/s3-storage")
```

### 2. S3 Storage Service

**File:** `src/cryptotrading/infrastructure/storage/s3_storage_service.py`

**Features:**
- Automatic credential management via Secrets Manager
- File upload/download operations
- Direct data upload without local files
- Presigned URL generation
- Object lifecycle management
- Comprehensive error handling

```python
from storage.s3_storage_service import S3StorageService

# Initialize (automatically loads credentials)
s3_service = S3StorageService()

# Upload data
success = s3_service.upload_data(
    data=json.dumps({"key": "value"}),
    s3_key="path/to/data.json",
    metadata={"type": "test"}
)
```

### 3. Crypto Data Manager

**File:** `src/cryptotrading/infrastructure/storage/crypto_data_manager.py`

**Features:**
- High-level cryptocurrency data operations
- Organized data structure with timestamps
- Market data handling (OHLCV, orderbook, trades)
- User data management (portfolio, trades)
- Analytics and reporting data
- Database backups and log file uploads

```python
from storage.crypto_data_manager import CryptoDataManager

# Initialize with S3 service
data_manager = CryptoDataManager(s3_service)

# Save market data
success = data_manager.save_ohlcv_data(
    symbol="BTC-USD",
    timeframe="1h",
    ohlcv_data=market_data
)
```

## 🛠️ Setup Instructions

### Step 1: Install Dependencies

```bash
pip install -r requirements_aws.txt
```

### Step 2: Configure AWS Credentials

**Option A: Using AWS CLI (Recommended)**
```bash
aws configure
```

**Option B: Using IAM Role (for EC2/Lambda)**
- Attach appropriate IAM role with S3 and Secrets Manager permissions

**Option C: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### Step 3: Setup S3 Credentials in Secrets Manager

```bash
python3 scripts/setup_s3_credentials.py
```

This interactive script will:
1. Connect to AWS Secrets Manager
2. Securely store your S3 credentials
3. Test the S3 connection
4. Verify upload/download operations

### Step 4: Test the Integration

```bash
python3 scripts/test_s3_storage.py
```

## 🔐 Required AWS Permissions

### IAM Policy for Secrets Manager Access

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue",
                "secretsmanager:CreateSecret",
                "secretsmanager:UpdateSecret",
                "secretsmanager:ListSecrets"
            ],
            "Resource": [
                "arn:aws:secretsmanager:*:*:secret:cryptotrading/*"
            ]
        }
    ]
}
```

### IAM Policy for S3 Bucket Access

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::tentimecrypto",
                "arn:aws:s3:::tentimecrypto/*"
            ]
        }
    ]
}
```

## 💡 Usage Examples

### Market Data Storage

```python
# Save OHLCV data
ohlcv_data = [
    {
        'timestamp': '2024-01-15T10:00:00Z',
        'open': 50000.0,
        'high': 51000.0,
        'low': 49500.0,
        'close': 50500.0,
        'volume': 125.5
    }
]

data_manager.save_ohlcv_data(
    symbol="BTC-USD",
    timeframe="1h",
    ohlcv_data=ohlcv_data
)
```

### User Portfolio Backup

```python
portfolio = {
    'user_id': 'craig',
    'total_value_usd': 125000.50,
    'assets': [
        {
            'symbol': 'BTC',
            'quantity': 2.5,
            'value_usd': 125000.00
        }
    ]
}

data_manager.save_user_portfolio(
    user_id="craig",
    portfolio=portfolio
)
```

### Analytics Report Storage

```python
analysis = {
    'symbol': 'BTC-USD',
    'indicators': {
        'rsi': 65.5,
        'macd': 1250.5
    },
    'recommendation': 'hold'
}

data_manager.save_analytics_report(
    report_type="technical_analysis",
    analysis_data=analysis
)
```

## 📊 Storage Statistics and Monitoring

```python
# Get storage usage statistics
stats = data_manager.get_storage_stats()
print(f"Total objects: {stats['total_objects']}")
print(f"Total size: {stats['total_size_mb']} MB")
```

## 🔄 Data Lifecycle Management

### Automated Cleanup

Consider implementing S3 lifecycle policies for cost optimization:

```json
{
    "Rules": [
        {
            "Id": "ArchiveOldData",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "market-data/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ]
        }
    ]
}
```

## 🚨 Security Best Practices

1. **Never commit AWS credentials** to version control
2. **Use IAM roles** when running on AWS infrastructure
3. **Rotate credentials regularly** using Secrets Manager
4. **Apply least privilege** IAM policies
5. **Enable S3 bucket encryption** at rest
6. **Use VPC endpoints** for private AWS communication
7. **Monitor access logs** and set up CloudTrail

## 🔧 Troubleshooting

### Common Issues

1. **Credentials Not Found**
   ```
   Error: Could not retrieve S3 credentials from secret
   ```
   **Solution:** Run `python3 scripts/setup_s3_credentials.py`

2. **Access Denied**
   ```
   Error: Access denied to S3 bucket
   ```
   **Solution:** Check IAM permissions and bucket policies

3. **Region Mismatch**
   ```
   Error: Bucket is in different region
   ```
   **Solution:** Ensure consistent region configuration

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Integration with Crypto Trading Platform

### Database Backups

```python
# Backup users table
import sqlite3
conn = sqlite3.connect('rex_trading.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM users")
users_data = cursor.fetchall()

data_manager.backup_database_table(
    table_name="users",
    data=[dict(zip([col[0] for col in cursor.description], row)) for row in users_data]
)
```

### Real-time Market Data Ingestion

```python
# In your market data collection script
def save_market_tick(symbol, tick_data):
    data_manager.save_trade_data(
        symbol=symbol,
        trades=[tick_data],
        timestamp=datetime.utcnow()
    )
```

### User Activity Logging

```python
# Log user trading activity
def log_user_trade(user_id, trade_data):
    data_manager.save_user_trades(
        user_id=user_id,
        trades=[trade_data],
        timestamp=datetime.utcnow()
    )
```

## 🔄 Backup and Recovery

### Automated Database Backups

Create a scheduled backup script:

```python
def daily_backup():
    # Backup all critical tables
    tables = ['users', 'orders', 'portfolio', 'market_data']
    for table in tables:
        backup_table_to_s3(table)
```

### Recovery Procedures

```python
def restore_from_backup(table_name, backup_date):
    # Download backup from S3
    # Restore to database
    pass
```

## 📈 Performance Optimization

### Batch Operations

```python
# Batch upload multiple files
def batch_upload_market_data(data_batch):
    for symbol, data in data_batch.items():
        data_manager.save_ohlcv_data(symbol, "1h", data)
```

### Compression

Enable compression for large datasets:

```python
import gzip
import json

compressed_data = gzip.compress(json.dumps(large_dataset).encode())
s3_service.upload_data(
    data=compressed_data,
    s3_key="compressed/data.json.gz",
    content_type="application/gzip"
)
```

## 🎉 Success! S3 Storage Integration Complete

Your cryptocurrency trading platform now has:

✅ **Secure credential management** via AWS Secrets Manager  
✅ **Scalable data storage** in S3  
✅ **Organized data structure** for efficient queries  
✅ **Comprehensive error handling** and logging  
✅ **High-level crypto data operations**  
✅ **Backup and recovery capabilities**  
✅ **Performance optimization features**  

The storage system is ready to handle market data, user portfolios, analytics reports, and system backups with enterprise-grade security and scalability.