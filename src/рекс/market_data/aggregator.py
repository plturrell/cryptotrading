"""
Market Data Aggregator for рекс.com
Combines data from multiple sources for comprehensive market coverage
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics
import time

from .geckoterminal import GeckoTerminalClient
from .coingecko import CoinGeckoClient
from .coinmarketcap import CoinMarketCapClient
from .bitquery import BitqueryClient
from ..database import get_db

class MarketDataAggregator:
    def __init__(self):
        """Initialize all market data clients"""
        self.clients = {
            "geckoterminal": GeckoTerminalClient(),
            "coingecko": CoinGeckoClient(),
        }
        
        # Add optional clients if API keys are available
        try:
            self.clients["coinmarketcap"] = CoinMarketCapClient()
        except ValueError:
            print("CoinMarketCap client not initialized (API key required)")
            
        try:
            self.clients["bitquery"] = BitqueryClient()
        except ValueError:
            print("Bitquery client not initialized (API key required)")
        
        self.executor = ThreadPoolExecutor(max_workers=len(self.clients))
        self.db = get_db()
        
    def get_aggregated_price(self, symbol: str, network: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated price data from all available sources"""
        results = {}
        futures = []
        
        # GeckoTerminal (for DEX prices)
        if "geckoterminal" in self.clients and network:
            future = self.executor.submit(
                self._get_geckoterminal_price, network, symbol
            )
            futures.append(("geckoterminal", future))
        
        # CoinGecko
        if "coingecko" in self.clients:
            future = self.executor.submit(
                self._get_coingecko_price, symbol
            )
            futures.append(("coingecko", future))
        
        # CoinMarketCap
        if "coinmarketcap" in self.clients:
            future = self.executor.submit(
                self._get_coinmarketcap_price, symbol
            )
            futures.append(("coinmarketcap", future))
        
        # Bitquery
        if "bitquery" in self.clients and network:
            future = self.executor.submit(
                self._get_bitquery_price, network, symbol
            )
            futures.append(("bitquery", future))
        
        # Collect results
        for source, future in futures:
            try:
                result = future.result(timeout=5)
                if result:
                    results[source] = result
            except Exception as e:
                print(f"Error getting price from {source}: {e}")
        
        # Calculate aggregated data
        if results:
            prices = [r["price"] for r in results.values() if "price" in r and r["price"]]
            volumes = [r.get("volume_24h", 0) for r in results.values()]
            
            aggregated = {
                "symbol": symbol,
                "network": network,
                "timestamp": datetime.now().isoformat(),
                "sources": len(results),
                "prices": {
                    "average": statistics.mean(prices) if prices else 0,
                    "median": statistics.median(prices) if prices else 0,
                    "min": min(prices) if prices else 0,
                    "max": max(prices) if prices else 0,
                    "std_dev": statistics.stdev(prices) if len(prices) > 1 else 0
                },
                "volume_24h_total": sum(volumes),
                "data_by_source": results
            }
            
            # Save to database
            self._save_aggregated_data(aggregated)
            
            return aggregated
        
        return {"error": "No price data available"}
    
    def _get_geckoterminal_price(self, network: str, token_address: str) -> Optional[Dict]:
        """Get price from GeckoTerminal"""
        try:
            price = self.clients["geckoterminal"].get_token_price(network, token_address)
            if price:
                volume_data = self.clients["geckoterminal"].get_pool_volume(network, token_address)
                return {
                    "price": price,
                    "volume_24h": volume_data.get("volume_24h", 0),
                    "liquidity": volume_data.get("liquidity_usd", 0),
                    "source": "geckoterminal",
                    "type": "dex"
                }
        except Exception as e:
            print(f"GeckoTerminal error: {e}")
        return None
    
    def _get_coingecko_price(self, coin_id: str) -> Optional[Dict]:
        """Get price from CoinGecko"""
        try:
            data = self.clients["coingecko"].get_price([coin_id])
            if coin_id in data:
                return {
                    "price": data[coin_id]["usd"],
                    "volume_24h": data[coin_id].get("usd_24h_vol", 0),
                    "market_cap": data[coin_id].get("usd_market_cap", 0),
                    "change_24h": data[coin_id].get("usd_24h_change", 0),
                    "source": "coingecko",
                    "type": "aggregated"
                }
        except Exception as e:
            print(f"CoinGecko error: {e}")
        return None
    
    def _get_coinmarketcap_price(self, symbol: str) -> Optional[Dict]:
        """Get price from CoinMarketCap"""
        try:
            data = self.clients["coinmarketcap"].get_quotes([symbol])
            if "data" in data and symbol in data["data"]:
                quote = data["data"][symbol][0]["quote"]["USD"]
                return {
                    "price": quote["price"],
                    "volume_24h": quote["volume_24h"],
                    "market_cap": quote["market_cap"],
                    "change_24h": quote["percent_change_24h"],
                    "change_7d": quote["percent_change_7d"],
                    "source": "coinmarketcap",
                    "type": "aggregated"
                }
        except Exception as e:
            print(f"CoinMarketCap error: {e}")
        return None
    
    def _get_bitquery_price(self, network: str, token_address: str) -> Optional[Dict]:
        """Get price from Bitquery"""
        try:
            data = self.clients["bitquery"].get_token_price(network, token_address)
            if "data" in data and data["data"]["ethereum"]["dexTrades"]:
                trade = data["data"]["ethereum"]["dexTrades"][0]
                return {
                    "price": float(trade["quotePrice"]),
                    "base_amount": float(trade["baseAmount"]),
                    "quote_amount": float(trade["quoteAmount"]),
                    "trades": trade["trades"],
                    "source": "bitquery",
                    "type": "dex"
                }
        except Exception as e:
            print(f"Bitquery error: {e}")
        return None
    
    def _save_aggregated_data(self, data: Dict[str, Any]):
        """Save aggregated data to database"""
        try:
            self.db.execute_query(
                """INSERT INTO aggregated_market_data 
                   (symbol, network, avg_price, median_price, min_price, max_price, 
                    std_dev, volume_24h, sources_count, raw_data, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (data["symbol"], data.get("network"), 
                 data["prices"]["average"], data["prices"]["median"],
                 data["prices"]["min"], data["prices"]["max"],
                 data["prices"]["std_dev"], data["volume_24h_total"],
                 data["sources"], str(data["data_by_source"]),
                 datetime.now())
            )
        except Exception as e:
            print(f"Database save error: {e}")
    
    def get_market_overview(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market overview for multiple symbols"""
        overview = {
            "timestamp": datetime.now().isoformat(),
            "symbols": {}
        }
        
        futures = []
        for symbol in symbols:
            future = self.executor.submit(self.get_aggregated_price, symbol)
            futures.append((symbol, future))
        
        for symbol, future in futures:
            try:
                result = future.result(timeout=10)
                overview["symbols"][symbol] = result
            except Exception as e:
                overview["symbols"][symbol] = {"error": str(e)}
        
        return overview
    
    def get_dex_opportunities(self, min_liquidity: float = 10000) -> List[Dict[str, Any]]:
        """Find trading opportunities across DEXs"""
        opportunities = []
        
        if "geckoterminal" in self.clients:
            try:
                # Get trending pools
                trending = self.clients["geckoterminal"].get_trending_pools()
                if "data" in trending:
                    for pool in trending["data"][:20]:  # Top 20 pools
                        attrs = pool.get("attributes", {})
                        liquidity = float(attrs.get("reserve_in_usd", 0))
                        
                        if liquidity >= min_liquidity:
                            opportunities.append({
                                "type": "trending_pool",
                                "network": pool.get("relationships", {}).get("network", {}).get("data", {}).get("id"),
                                "pool_address": attrs.get("address"),
                                "name": attrs.get("name"),
                                "liquidity_usd": liquidity,
                                "volume_24h": float(attrs.get("volume_usd", {}).get("h24", 0)),
                                "price_change_24h": float(attrs.get("price_change_percentage", {}).get("h24", 0)),
                                "source": "geckoterminal"
                            })
            except Exception as e:
                print(f"Error getting DEX opportunities: {e}")
        
        if "bitquery" in self.clients:
            try:
                # Get arbitrage opportunities
                arb_data = self.clients["bitquery"].get_arbitrage_opportunities("ethereum", min_profit_percent=0.5)
                if "opportunities" in arb_data:
                    for arb in arb_data["opportunities"][:10]:  # Top 10 opportunities
                        opportunities.append({
                            "type": "arbitrage",
                            "network": "ethereum",
                            "pair": arb["pair"],
                            "base_symbol": arb["base_symbol"],
                            "quote_symbol": arb["quote_symbol"],
                            "profit_percent": arb["profit_percent"],
                            "buy_exchange": arb["buy_exchange"],
                            "sell_exchange": arb["sell_exchange"],
                            "buy_price": arb["min_price"],
                            "sell_price": arb["max_price"],
                            "source": "bitquery"
                        })
            except Exception as e:
                print(f"Error getting arbitrage opportunities: {e}")
        
        return opportunities
    
    def start_real_time_monitoring(self, symbols: List[str], callback: Callable, interval: int = 30):
        """Start real-time price monitoring for multiple symbols"""
        def monitor_symbol(symbol):
            while True:
                try:
                    data = self.get_aggregated_price(symbol)
                    callback(symbol, data)
                    time.sleep(interval)
                except Exception as e:
                    print(f"Monitoring error for {symbol}: {e}")
                    time.sleep(interval * 2)
        
        # Start monitoring threads
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=monitor_symbol, args=(symbol,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        return threads
    
    def get_historical_data(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Get historical data from available sources"""
        historical = {
            "symbol": symbol,
            "days": days,
            "data": {}
        }
        
        # CoinGecko historical
        if "coingecko" in self.clients:
            try:
                chart_data = self.clients["coingecko"].get_coin_market_chart(symbol, days=days)
                historical["data"]["coingecko"] = chart_data
            except Exception as e:
                print(f"Error getting CoinGecko historical: {e}")
        
        # CoinMarketCap historical
        if "coinmarketcap" in self.clients:
            try:
                ohlcv_data = self.clients["coinmarketcap"].get_ohlcv_historical(symbol)
                historical["data"]["coinmarketcap"] = ohlcv_data
            except Exception as e:
                print(f"Error getting CoinMarketCap historical: {e}")
        
        return historical
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        if self.db:
            self.db.close()