#!/usr/bin/env python3
"""
Populate the factor_data table with technical analysis indicators
Week 2 - Only using Yahoo Finance data
"""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta  # Technical Analysis library
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import our database models
from src.cryptotrading.data.database.models import FactorData
from src.cryptotrading.data.providers.real_only_provider import RealOnlyDataProvider
from src.cryptotrading.infrastructure.database.unified_database import UnifiedDatabase


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators from OHLCV data"""

    # Price indicators
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)

    # Volume indicators
    df["volume_sma_20"] = ta.trend.sma_indicator(df["volume"], window=20)
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

    # Technical indicators
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high=df["high"], low=df["low"], close=df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Volatility
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["volatility"] = df["close"].pct_change().rolling(window=24).std() * np.sqrt(24)

    return df


async def store_factors_in_db(symbol: str, df: pd.DataFrame):
    """Store calculated factors in the database"""

    # Initialize database
    db = UnifiedDatabase()
    await db.initialize()

    # List of factors to store
    factors_to_store = [
        "sma_20",
        "ema_50",
        "volume_sma_20",
        "obv",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_position",
        "stoch_k",
        "atr",
        "volatility",
    ]

    records = []

    # Prepare records for bulk insert
    for idx, row in df.iterrows():
        timestamp = idx if isinstance(idx, pd.Timestamp) else pd.to_datetime(idx)

        for factor in factors_to_store:
            if factor in row and not pd.isna(row[factor]):
                records.append(
                    {
                        "symbol": symbol,
                        "factor_name": factor,
                        "timestamp": timestamp,
                        "value": float(row[factor]),
                        "quality_score": 1.0,
                        "confidence": 1.0,
                        "calculation_method": "batch",
                        "input_data_points": len(df),
                        "passed_validation": True,
                        "calculation_version": "1.0",
                    }
                )

    # Bulk insert using raw SQL for efficiency
    if records:
        print(f"Inserting {len(records)} factor records for {symbol}...")

        # Use the database connection directly
        cursor = db.db_conn.cursor()

        # Delete existing factors for this symbol (optional - for clean slate)
        cursor.execute("DELETE FROM factor_data WHERE symbol = ?", (symbol,))

        # Bulk insert
        cursor.executemany(
            """
            INSERT INTO factor_data 
            (symbol, factor_name, timestamp, value, quality_score, confidence, 
             calculation_method, input_data_points, passed_validation, calculation_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                (
                    r["symbol"],
                    r["factor_name"],
                    r["timestamp"],
                    r["value"],
                    r["quality_score"],
                    r["confidence"],
                    r["calculation_method"],
                    r["input_data_points"],
                    r["passed_validation"],
                    r["calculation_version"],
                )
                for r in records
            ],
        )

        db.db_conn.commit()
        print(f"Successfully stored {len(records)} factor values")


async def main():
    """Main function to calculate and store factors"""

    symbols = ["BTC-USD", "ETH-USD", "BNB-USD"]  # Top 3 cryptos
    days_back = 90  # 3 months of data

    # Get data provider
    provider = RealOnlyDataProvider()

    for symbol in symbols:
        try:
            print(f"\nProcessing {symbol}...")

            # Get historical data
            df = provider.get_historical_data(symbol, days=days_back)

            if df is None or df.empty:
                print(f"No data available for {symbol}")
                continue

            print(f"Got {len(df)} rows of data")

            # Calculate indicators
            df_with_indicators = calculate_technical_indicators(df)

            # Store in database
            await store_factors_in_db(symbol, df_with_indicators)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
