#!/usr/bin/env python3
"""
VERIFY ACTUAL HISTORICAL DATA AVAILABILITY
Test what date ranges are actually available from Yahoo Finance
"""

import sys

sys.path.append("src")

from datetime import datetime, timedelta

import pandas as pd


def test_actual_historical_data_range():
    """Test what historical data is actually available"""
    print("🔍 VERIFYING ACTUAL HISTORICAL DATA AVAILABILITY")
    print("=" * 60)

    try:
        from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient

        client = YahooFinanceClient()

        # Test different date ranges to see what's actually available
        test_ranges = [
            (
                "5 years",
                (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            ),
            (
                "3 years",
                (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            ),
            (
                "2 years",
                (datetime.now() - timedelta(days=2 * 365)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            ),
            (
                "1 year",
                (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            ),
            (
                "6 months",
                (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d"),
            ),
        ]

        symbols = ["BTC-USD", "ETH-USD"]

        for symbol in symbols:
            print(f"\n📊 Testing {symbol} historical data ranges...")
            print("-" * 40)

            for period_name, start_date, end_date in test_ranges:
                try:
                    data = client.download_data(
                        symbol=symbol, start_date=start_date, end_date=end_date, save=False
                    )

                    if data is not None and len(data) > 0:
                        actual_start = data.index[0].strftime("%Y-%m-%d")
                        actual_end = data.index[-1].strftime("%Y-%m-%d")
                        record_count = len(data)

                        print(
                            f"✅ {period_name:<10}: {record_count:4d} records ({actual_start} to {actual_end})"
                        )

                        # Show sample of actual data
                        if period_name == "2 years":
                            print(
                                f"   Sample prices: ${data['close'].iloc[0]:.2f} → ${data['close'].iloc[-1]:.2f}"
                            )
                            print(
                                f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}"
                            )
                    else:
                        print(f"❌ {period_name:<10}: No data available")

                except Exception as e:
                    print(f"❌ {period_name:<10}: Error - {str(e)}")

        # Test specific 2-year claim
        print(f"\n🎯 SPECIFIC TEST: 2+ YEAR CLAIM VERIFICATION")
        print("-" * 50)

        two_years_ago = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        now = datetime.now().strftime("%Y-%m-%d")

        print(f"Requesting data from {two_years_ago} to {now}")

        btc_2yr = client.download_data(
            symbol="BTC-USD", start_date=two_years_ago, end_date=now, save=False
        )

        if btc_2yr is not None and len(btc_2yr) > 0:
            actual_start = btc_2yr.index[0]
            actual_end = btc_2yr.index[-1]
            days_span = (actual_end - actual_start).days
            years_span = days_span / 365.25

            print(f"✅ VERIFIED: BTC data available")
            print(
                f"   Actual range: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}"
            )
            print(f"   Days span: {days_span} days")
            print(f"   Years span: {years_span:.2f} years")
            print(f"   Records: {len(btc_2yr)}")

            if years_span >= 2.0:
                print(f"✅ CLAIM VERIFIED: 2+ years of data IS available ({years_span:.2f} years)")
                return True, years_span
            else:
                print(f"❌ CLAIM FALSE: Only {years_span:.2f} years available (less than 2 years)")
                return False, years_span
        else:
            print(f"❌ CLAIM FALSE: No 2-year data available")
            return False, 0

    except Exception as e:
        print(f"❌ Error testing historical data: {e}")
        return False, 0


def test_yahoo_finance_limits():
    """Test Yahoo Finance data availability limits"""
    print(f"\n🔍 TESTING YAHOO FINANCE DATA LIMITS")
    print("=" * 50)

    try:
        import yfinance as yf

        # Test BTC directly with yfinance
        btc = yf.Ticker("BTC-USD")

        # Test maximum history
        print("📊 Testing maximum historical data...")
        max_history = btc.history(period="max")

        if len(max_history) > 0:
            start_date = max_history.index[0]
            end_date = max_history.index[-1]
            total_days = (end_date - start_date).days
            total_years = total_days / 365.25

            print(f"✅ Maximum BTC history available:")
            print(f"   Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   End: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Total days: {total_days}")
            print(f"   Total years: {total_years:.2f}")
            print(f"   Total records: {len(max_history)}")

            # Test specific periods
            periods = ["5y", "3y", "2y", "1y", "6mo", "3mo", "1mo"]

            print(f"\n📈 Testing specific periods:")
            for period in periods:
                try:
                    data = btc.history(period=period)
                    if len(data) > 0:
                        days = (data.index[-1] - data.index[0]).days
                        years = days / 365.25
                        print(
                            f"   {period:<4}: {len(data):4d} records, {days:4d} days ({years:.2f} years)"
                        )
                    else:
                        print(f"   {period:<4}: No data")
                except Exception as e:
                    print(f"   {period:<4}: Error - {str(e)[:30]}")

            return total_years >= 2.0, total_years
        else:
            print("❌ No historical data available")
            return False, 0

    except Exception as e:
        print(f"❌ Error testing Yahoo Finance limits: {e}")
        return False, 0


def main():
    """Main verification function"""
    print("🎯 HISTORICAL DATA AVAILABILITY VERIFICATION")
    print("=" * 70)
    print("CHECKING ACTUAL DATA AVAILABILITY (NOT CLAIMS)")
    print("=" * 70)

    # Test 1: Our client
    client_result, client_years = test_actual_historical_data_range()

    # Test 2: Direct Yahoo Finance
    yahoo_result, yahoo_years = test_yahoo_finance_limits()

    # Final verdict
    print("\n" + "=" * 70)
    print("🎯 FINAL VERIFICATION RESULTS")
    print("=" * 70)

    print(
        f"Our Client Test: {'✅ VERIFIED' if client_result else '❌ FAILED'} ({client_years:.2f} years)"
    )
    print(
        f"Yahoo Direct Test: {'✅ VERIFIED' if yahoo_result else '❌ FAILED'} ({yahoo_years:.2f} years)"
    )

    if client_result or yahoo_result:
        max_years = max(client_years, yahoo_years)
        print(f"\n✅ CLAIM VERIFICATION: {max_years:.2f} years of historical data IS available")
        print(f"   This {'SUPPORTS' if max_years >= 2.0 else 'CONTRADICTS'} the '2+ years' claim")

        if max_years >= 2.0:
            print(f"\n🎉 CLAIM CONFIRMED: Historical data analysis (2+ years available) is TRUE")
        else:
            print(
                f"\n❌ CLAIM REFUTED: Only {max_years:.2f} years available, less than claimed 2+ years"
            )
    else:
        print(f"\n❌ CLAIM REFUTED: Unable to access significant historical data")

    return client_result or yahoo_result


if __name__ == "__main__":
    main()
