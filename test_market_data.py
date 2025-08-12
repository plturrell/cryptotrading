#!/usr/bin/env python3
"""
Test market data feed integrations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.рекс.market_data import (
    GeckoTerminalClient,
    CoinGeckoClient,
    MarketDataAggregator
)

def test_geckoterminal():
    """Test GeckoTerminal API"""
    print("\n=== Testing GeckoTerminal API ===")
    client = GeckoTerminalClient()
    
    # Test getting networks
    print("\n1. Getting supported networks...")
    networks = client.get_networks()
    if "data" in networks:
        print(f"Found {len(networks['data'])} networks")
        print(f"First network: {networks['data'][0]['id'] if networks['data'] else 'None'}")
    
    # Test trending pools
    print("\n2. Getting trending pools...")
    trending = client.get_trending_pools()
    if "data" in trending:
        print(f"Found {len(trending['data'])} trending pools")
        if trending['data']:
            pool = trending['data'][0]
            attrs = pool.get('attributes', {})
            print(f"Top pool: {attrs.get('name', 'Unknown')}")
            print(f"Liquidity: ${attrs.get('reserve_in_usd', 0):,.2f}")
    
    # Test ETH price on Ethereum
    print("\n3. Getting ETH price...")
    weth_address = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    price = client.get_token_price("eth", weth_address)
    if price:
        print(f"ETH price: ${price:,.2f}")
    
    return True

def test_coingecko():
    """Test CoinGecko API"""
    print("\n=== Testing CoinGecko API ===")
    client = CoinGeckoClient()
    
    # Test getting BTC price
    print("\n1. Getting Bitcoin price...")
    prices = client.get_price(["bitcoin"])
    if "bitcoin" in prices:
        btc_data = prices["bitcoin"]
        print(f"BTC price: ${btc_data['usd']:,.2f}")
        print(f"24h change: {btc_data.get('usd_24h_change', 0):.2f}%")
        print(f"Market cap: ${btc_data.get('usd_market_cap', 0):,.0f}")
    
    # Test trending coins
    print("\n2. Getting trending coins...")
    trending = client.get_trending_coins()
    if "coins" in trending:
        print(f"Found {len(trending['coins'])} trending coins")
        if trending['coins']:
            coin = trending['coins'][0]['item']
            print(f"Top trending: {coin['name']} ({coin['symbol']})")
    
    # Test global DeFi data
    print("\n3. Getting DeFi market data...")
    defi = client.get_defi_data()
    if "data" in defi:
        defi_data = defi["data"]
        print(f"DeFi market cap: ${defi_data.get('defi_market_cap', 0):,.0f}")
        print(f"DeFi dominance: {defi_data.get('defi_dominance', 0):.2f}%")
    
    return True

def test_aggregator():
    """Test Market Data Aggregator"""
    print("\n=== Testing Market Data Aggregator ===")
    aggregator = MarketDataAggregator()
    
    # Test aggregated BTC price
    print("\n1. Getting aggregated Bitcoin price...")
    btc_data = aggregator.get_aggregated_price("bitcoin")
    if "prices" in btc_data:
        prices = btc_data["prices"]
        print(f"Average price: ${prices['average']:,.2f}")
        print(f"Median price: ${prices['median']:,.2f}")
        print(f"Price range: ${prices['min']:,.2f} - ${prices['max']:,.2f}")
        print(f"Sources: {btc_data['sources']}")
    
    # Test market overview
    print("\n2. Getting market overview...")
    overview = aggregator.get_market_overview(["bitcoin", "ethereum"])
    print(f"Got data for {len(overview['symbols'])} symbols")
    
    # Test DEX opportunities
    print("\n3. Finding DEX opportunities...")
    opportunities = aggregator.get_dex_opportunities(min_liquidity=50000)
    print(f"Found {len(opportunities)} opportunities")
    if opportunities:
        opp = opportunities[0]
        print(f"Top opportunity: {opp.get('type', 'Unknown')}")
        if opp['type'] == 'trending_pool':
            print(f"Pool: {opp.get('name', 'Unknown')}")
            print(f"Liquidity: ${opp.get('liquidity_usd', 0):,.2f}")
            print(f"24h volume: ${opp.get('volume_24h', 0):,.2f}")
    
    return True

def main():
    """Run all tests"""
    print("Testing Market Data Feeds for рекс.com")
    print("=" * 50)
    
    tests = [
        ("GeckoTerminal", test_geckoterminal),
        ("CoinGecko", test_coingecko),
        ("Aggregator", test_aggregator)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "✓ PASSED" if result else "✗ FAILED"))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, f"✗ ERROR: {str(e)[:50]}..."))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for name, status in results:
        print(f"{name:<20} {status}")
    
    passed = sum(1 for _, status in results if "PASSED" in status)
    print(f"\nTotal: {passed}/{len(tests)} tests passed")

if __name__ == "__main__":
    main()