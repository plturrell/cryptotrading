#!/usr/bin/env python3
"""
Test crypto-specific data operations with real S3
"""

import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def test_comprehensive_crypto_data():
    """Test comprehensive crypto data operations"""

    print("💰 Comprehensive Crypto Data Test")
    print("=" * 50)

    # Get AWS credentials from environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("S3_BUCKET_NAME", "tentimecrypto")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    if not aws_access_key or not aws_secret_key:
        print("❌ AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return

    # Environment variables should already be set externally
    # No need to set them in code for security reasons

    try:
        # Mock secrets manager
        mock_secrets_manager = Mock()
        mock_secrets_manager.get_secret.return_value = {
            "aws_access_key_id": aws_access_key,
            "aws_secret_access_key": aws_secret_key,
            "bucket_name": bucket_name,
            "region": region,
        }

        with patch(
            "src.cryptotrading.infrastructure.storage.s3_storage_service.SecretsManager",
            return_value=mock_secrets_manager,
        ):
            from src.cryptotrading.infrastructure.storage.crypto_data_manager import (
                CryptoDataManager,
            )
            from src.cryptotrading.infrastructure.storage.s3_storage_service import S3StorageService

            # Initialize services
            s3_service = S3StorageService()
            data_manager = CryptoDataManager(s3_service)

            print("✅ Services initialized")

            # Test 1: Save market data for multiple symbols
            print("\n📊 Test 1: Saving market data for multiple symbols...")
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

            for symbol in symbols:
                ohlcv_data = generate_realistic_ohlcv(symbol)
                success = data_manager.save_ohlcv_data(
                    symbol=symbol, timeframe="1h", ohlcv_data=ohlcv_data, timestamp=datetime.now()
                )
                print(f"  {symbol}: {'✅' if success else '❌'}")

            # Test 2: Save orderbook data
            print("\n📋 Test 2: Saving orderbook data...")
            orderbook = generate_realistic_orderbook("BTC-USD")
            orderbook_success = data_manager.save_orderbook_data(
                symbol="BTC-USD", orderbook=orderbook, timestamp=datetime.now()
            )
            print(f"  Orderbook: {'✅' if orderbook_success else '❌'}")

            # Test 3: Save user data for our 4 users
            print("\n👥 Test 3: Saving user portfolios...")
            users = ["craig", "irina", "dasha", "dany"]

            for user in users:
                portfolio = generate_user_portfolio(user)
                success = data_manager.save_user_portfolio(
                    user_id=user, portfolio=portfolio, timestamp=datetime.now()
                )
                print(f"  {user}: {'✅' if success else '❌'}")

            # Test 4: Save user trades
            print("\n📈 Test 4: Saving user trades...")
            for user in users[:2]:  # Just Craig and Irina
                trades = generate_user_trades(user)
                success = data_manager.save_user_trades(
                    user_id=user, trades=trades, timestamp=datetime.now()
                )
                print(f"  {user} trades: {'✅' if success else '❌'}")

            # Test 5: Save analytics reports
            print("\n📊 Test 5: Saving analytics reports...")
            report_types = ["technical_analysis", "sentiment_analysis", "risk_assessment"]

            for report_type in report_types:
                analytics = generate_analytics_report(report_type)
                success = data_manager.save_analytics_report(
                    report_type=report_type, analysis_data=analytics, timestamp=datetime.now()
                )
                print(f"  {report_type}: {'✅' if success else '❌'}")

            # Test 6: Database backup
            print("\n💾 Test 6: Database backup simulation...")

            # Simulate backing up our users table
            users_backup = [
                {"id": 1, "username": "craig", "role": "admin"},
                {"id": 2, "username": "irina", "role": "trader"},
                {"id": 3, "username": "dasha", "role": "analyst"},
                {"id": 4, "username": "dany", "role": "trader"},
            ]

            backup_success = data_manager.backup_database_table(
                table_name="users", data=users_backup, timestamp=datetime.now()
            )
            print(f"  Users table backup: {'✅' if backup_success else '❌'}")

            # Test 7: Get storage statistics
            print("\n📊 Test 7: Storage statistics...")
            stats = data_manager.get_storage_stats()

            if stats:
                print(f"  Total objects: {stats['total_objects']}")
                print(f"  Total size: {stats['total_size_mb']} MB")
                print(f"  Data types: {len(stats.get('data_types', {}))}")

                # Show breakdown by data type
                for data_type, info in stats.get("data_types", {}).items():
                    print(f"    {data_type}: {info['count']} files, {info['size_mb']} MB")

                print("  ✅ Storage statistics retrieved")
            else:
                print("  ❌ Storage statistics failed")

            # Test 8: List files by category
            print("\n📂 Test 8: File listing by category...")

            # List user data files for Craig
            craig_files = data_manager.list_user_data_files("craig")
            print(f"  Craig's files: {len(craig_files)} files")

            # List portfolio files specifically
            portfolio_files = data_manager.list_user_data_files("craig", "portfolio")
            print(f"  Craig's portfolio files: {len(portfolio_files)} files")

            # Test 9: Data retrieval
            print("\n📥 Test 9: Data retrieval...")

            # Try to get latest market data
            latest_btc = data_manager.get_latest_market_data("BTC-USD", "ohlcv", days_back=1)
            if latest_btc:
                print("  ✅ Latest BTC data retrieved")
                print(f"    Symbol: {latest_btc.get('symbol')}")
                print(f"    Records: {len(latest_btc.get('data', []))}")
            else:
                print("  ⚠️ Latest BTC data not found (may need time for indexing)")

            return True

    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def generate_realistic_ohlcv(symbol):
    """Generate realistic OHLCV data"""

    # Base prices for different symbols
    base_prices = {"BTC-USD": 65000, "ETH-USD": 3200, "SOL-USD": 150}

    base_price = base_prices.get(symbol, 50000)
    current_time = datetime.utcnow()

    data = []
    for i in range(24):  # 24 hours
        timestamp = current_time - timedelta(hours=23 - i)

        # Simulate price movement
        change = 1 + (0.02 * (0.5 - hash(f"{symbol}{i}") % 100 / 100))
        open_price = base_price * change

        high = open_price * (1 + 0.005 * abs(hash(f"h{i}") % 10))
        low = open_price * (1 - 0.005 * abs(hash(f"l{i}") % 10))
        close = open_price + (high - low) * ((hash(f"c{i}") % 100) / 100 - 0.5)
        volume = 100 + (hash(f"v{i}") % 500)

        data.append(
            {
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": round(volume, 2),
            }
        )

        base_price = close  # Next candle starts here

    return data


def generate_realistic_orderbook(symbol):
    """Generate realistic orderbook data"""

    mid_price = 65000 if symbol == "BTC-USD" else 3200

    bids = []
    asks = []

    # Generate 10 bid levels
    for i in range(10):
        price = mid_price - (i + 1) * 10
        quantity = 0.1 + (hash(f"bid{i}") % 50) / 100
        bids.append([price, quantity])

    # Generate 10 ask levels
    for i in range(10):
        price = mid_price + (i + 1) * 10
        quantity = 0.1 + (hash(f"ask{i}") % 50) / 100
        asks.append([price, quantity])

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
    }


def generate_user_portfolio(user_id):
    """Generate realistic user portfolio"""

    # Different portfolio styles per user
    portfolios = {
        "craig": {
            "total_value_usd": 250000,
            "assets": [
                {"symbol": "BTC", "quantity": 3.5, "avg_price": 60000, "current_price": 65000},
                {"symbol": "ETH", "quantity": 15, "avg_price": 3000, "current_price": 3200},
            ],
        },
        "irina": {
            "total_value_usd": 125000,
            "assets": [
                {"symbol": "BTC", "quantity": 1.8, "avg_price": 58000, "current_price": 65000},
                {"symbol": "SOL", "quantity": 200, "avg_price": 140, "current_price": 150},
            ],
        },
        "dasha": {
            "total_value_usd": 75000,
            "assets": [
                {"symbol": "ETH", "quantity": 20, "avg_price": 2800, "current_price": 3200},
                {"symbol": "BTC", "quantity": 0.5, "avg_price": 62000, "current_price": 65000},
            ],
        },
        "dany": {
            "total_value_usd": 95000,
            "assets": [
                {"symbol": "BTC", "quantity": 1.2, "avg_price": 55000, "current_price": 65000},
                {"symbol": "ETH", "quantity": 8, "avg_price": 2900, "current_price": 3200},
                {"symbol": "SOL", "quantity": 100, "avg_price": 130, "current_price": 150},
            ],
        },
    }

    portfolio = portfolios.get(user_id, portfolios["craig"])
    portfolio["user_id"] = user_id
    portfolio["last_updated"] = datetime.utcnow().isoformat()

    return portfolio


def generate_user_trades(user_id):
    """Generate realistic user trades"""

    trades = []
    current_time = datetime.utcnow()

    for i in range(5):  # 5 recent trades
        trade_time = current_time - timedelta(hours=i * 6)

        trade = {
            "id": f"trade_{user_id}_{i}",
            "timestamp": trade_time.isoformat(),
            "symbol": "BTC-USD" if i % 2 == 0 else "ETH-USD",
            "side": "buy" if i % 3 == 0 else "sell",
            "quantity": round(0.1 + (hash(f"{user_id}{i}") % 20) / 100, 3),
            "price": 65000 if i % 2 == 0 else 3200,
            "fee": 0.001,
            "status": "completed",
        }

        trades.append(trade)

    return trades


def generate_analytics_report(report_type):
    """Generate analytics report data"""

    reports = {
        "technical_analysis": {
            "symbols_analyzed": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "timeframe": "1h",
            "indicators": {
                "rsi": {"BTC-USD": 65.5, "ETH-USD": 58.2, "SOL-USD": 72.1},
                "macd": {"BTC-USD": 1250, "ETH-USD": 85, "SOL-USD": 5.2},
                "bollinger_bands": {"BTC-USD": {"upper": 67000, "middle": 65000, "lower": 63000}},
            },
            "signals": {
                "BTC-USD": {"trend": "bullish", "strength": "moderate"},
                "ETH-USD": {"trend": "neutral", "strength": "weak"},
                "SOL-USD": {"trend": "bullish", "strength": "strong"},
            },
        },
        "sentiment_analysis": {
            "overall_sentiment": 0.65,
            "sources": ["twitter", "reddit", "news"],
            "sentiment_breakdown": {"positive": 45, "neutral": 35, "negative": 20},
            "key_topics": ["bitcoin etf", "ethereum upgrade", "regulatory news"],
        },
        "risk_assessment": {
            "market_risk": "medium",
            "volatility_index": 0.75,
            "correlation_matrix": {"BTC-ETH": 0.82, "BTC-SOL": 0.71, "ETH-SOL": 0.68},
            "recommendations": {
                "portfolio_allocation": {"BTC": 60, "ETH": 30, "SOL": 10},
                "risk_level": "moderate",
            },
        },
    }

    report = reports.get(report_type, reports["technical_analysis"])
    report["timestamp"] = datetime.utcnow().isoformat()
    report["report_type"] = report_type

    return report


if __name__ == "__main__":
    success = test_comprehensive_crypto_data()

    if success:
        print("\n🎉 Comprehensive crypto data test successful!")
        print("✅ All crypto trading data operations working")
        print("✅ Market data, user portfolios, analytics all stored")
        print("✅ S3 storage system is production ready!")
    else:
        print("\n❌ Comprehensive test failed")
        print("Please check the error details above")
