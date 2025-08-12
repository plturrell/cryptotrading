"""
GeckoTerminal API client for DEX market data
Free tier: 30 calls/min
Coverage: 1,600+ DEXs across 240+ networks
"""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import os
from ..database import get_db
from ..utils import rate_limiter

class GeckoTerminalClient:
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.geckoterminal.com/api/v2"
        self.api_key = api_key or os.getenv('GECKOTERMINAL_API_KEY')
        self.session = requests.Session()
        if self.api_key:
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
        self.session.headers['Accept'] = 'application/json'
        self.last_request_time = 0
        self.rate_limit = 2.0  # 30 calls/min = 1 call per 2 seconds
        
    def _rate_limit(self):
        """Enforce rate limiting using global rate limiter"""
        rate_limiter.wait_if_needed("geckoterminal")
        rate_limiter.record_call("geckoterminal")
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make rate-limited API request"""
        self._rate_limit()
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"GeckoTerminal API error: {e}")
            return {"error": str(e)}
    
    def get_networks(self) -> Dict[str, Any]:
        """Get list of supported networks"""
        return self._request("/networks")
    
    def get_dexes(self, network: str) -> Dict[str, Any]:
        """Get DEXs on a specific network"""
        return self._request(f"/networks/{network}/dexes")
    
    def get_trending_pools(self, network: Optional[str] = None) -> Dict[str, Any]:
        """Get trending pools across all networks or specific network"""
        if network:
            return self._request(f"/networks/{network}/trending_pools")
        return self._request("/networks/trending_pools")
    
    def get_pool_by_address(self, network: str, address: str) -> Dict[str, Any]:
        """Get specific pool data by address"""
        return self._request(f"/networks/{network}/pools/{address}")
    
    def get_pool_trades(self, network: str, pool_address: str, limit: int = 20) -> Dict[str, Any]:
        """Get recent trades for a pool"""
        params = {"limit": min(limit, 100)}
        return self._request(f"/networks/{network}/pools/{pool_address}/trades", params)
    
    def get_pool_ohlcv(self, network: str, pool_address: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get OHLCV data for a pool
        Timeframes: 1m, 5m, 15m, 1h, 4h, 12h, 1d
        """
        params = {"timeframe": timeframe}
        return self._request(f"/networks/{network}/pools/{pool_address}/ohlcv", params)
    
    def get_token_info(self, network: str, address: str) -> Dict[str, Any]:
        """Get token information"""
        return self._request(f"/networks/{network}/tokens/{address}")
    
    def get_token_pools(self, network: str, token_address: str) -> Dict[str, Any]:
        """Get all pools for a specific token"""
        return self._request(f"/networks/{network}/tokens/{token_address}/pools")
    
    def search_pools(self, query: str, network: Optional[str] = None) -> Dict[str, Any]:
        """Search for pools by token symbol or name"""
        params = {"query": query}
        if network:
            params["network"] = network
        return self._request("/search/pools", params)
    
    def get_new_pools(self, network: Optional[str] = None) -> Dict[str, Any]:
        """Get recently created pools"""
        if network:
            return self._request(f"/networks/{network}/new_pools")
        return self._request("/networks/new_pools")
    
    def get_pool_volume(self, network: str, pool_address: str) -> Dict[str, Any]:
        """Get 24h volume and volume chart data"""
        data = self.get_pool_by_address(network, pool_address)
        if "data" in data and "attributes" in data["data"]:
            attrs = data["data"]["attributes"]
            return {
                "volume_24h": attrs.get("volume_usd", {}).get("h24", 0),
                "volume_7d": attrs.get("volume_usd", {}).get("d7", 0),
                "volume_30d": attrs.get("volume_usd", {}).get("d30", 0),
                "liquidity_usd": attrs.get("reserve_in_usd", 0)
            }
        return {}
    
    def get_token_price(self, network: str, token_address: str) -> Optional[float]:
        """Get token price from most liquid pool"""
        pools = self.get_token_pools(network, token_address)
        if "data" in pools and pools["data"]:
            # Sort by liquidity and get price from most liquid pool
            sorted_pools = sorted(
                pools["data"], 
                key=lambda p: float(p.get("attributes", {}).get("reserve_in_usd", 0)),
                reverse=True
            )
            if sorted_pools:
                return float(sorted_pools[0]["attributes"].get("base_token_price_usd", 0))
        return None
    
    def monitor_token(self, network: str, token_address: str, callback=None):
        """Monitor token price changes in real-time"""
        last_price = None
        while True:
            try:
                current_price = self.get_token_price(network, token_address)
                if current_price and current_price != last_price:
                    price_data = {
                        "network": network,
                        "token": token_address,
                        "price": current_price,
                        "timestamp": datetime.now().isoformat(),
                        "change": (current_price - last_price) / last_price * 100 if last_price else 0
                    }
                    
                    # Save to database
                    try:
                        db = get_db()
                        db.execute_query(
                            """INSERT INTO market_data (source, symbol, price, volume_24h, timestamp)
                               VALUES (?, ?, ?, ?, ?)""",
                            ("geckoterminal", f"{network}:{token_address}", current_price, 0, datetime.now())
                        )
                    except Exception as e:
                        print(f"Database save error: {e}")
                    
                    if callback:
                        callback(price_data)
                    
                    last_price = current_price
                
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)