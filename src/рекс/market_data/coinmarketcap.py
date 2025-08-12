"""
CoinMarketCap API client for crypto market data
Free tier: Available with limits
Coverage: Major DEXs and CEXs
"""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import os
from ..database import get_db

class CoinMarketCapClient:
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://pro-api.coinmarketcap.com"
        self.api_key = api_key or os.getenv('COINMARKETCAP_API_KEY')
        
        if not self.api_key:
            raise ValueError("CoinMarketCap API key is required")
        
        self.session = requests.Session()
        self.session.headers.update({
            'X-CMC_PRO_API_KEY': self.api_key,
            'Accept': 'application/json',
            'Accept-Encoding': 'deflate, gzip'
        })
        
        self.last_request_time = 0
        self.rate_limit = 2.0  # Conservative rate limit
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _request(self, endpoint: str, params: Optional[Dict] = None, version: str = "v2") -> Dict[str, Any]:
        """Make rate-limited API request"""
        self._rate_limit()
        url = f"{self.base_url}/{version}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"CoinMarketCap API error: {e}")
            return {"error": str(e)}
    
    def get_latest_listings(self, limit: int = 100, convert: str = "USD") -> Dict[str, Any]:
        """Get latest cryptocurrency listings"""
        params = {
            "limit": limit,
            "convert": convert
        }
        return self._request("/cryptocurrency/listings/latest", params, "v1")
    
    def get_quotes(self, symbols: List[str], convert: str = "USD") -> Dict[str, Any]:
        """Get quotes for specific symbols"""
        params = {
            "symbol": ",".join(symbols),
            "convert": convert
        }
        return self._request("/cryptocurrency/quotes/latest", params, "v2")
    
    def get_quotes_by_id(self, ids: List[int], convert: str = "USD") -> Dict[str, Any]:
        """Get quotes by CMC IDs"""
        params = {
            "id": ",".join(map(str, ids)),
            "convert": convert
        }
        return self._request("/cryptocurrency/quotes/latest", params, "v2")
    
    def get_global_metrics(self, convert: str = "USD") -> Dict[str, Any]:
        """Get global market metrics"""
        params = {"convert": convert}
        return self._request("/global-metrics/quotes/latest", params, "v1")
    
    def get_trending_gainers_losers(self, limit: int = 10, time_period: str = "24h") -> Dict[str, Any]:
        """Get trending gainers and losers"""
        params = {
            "limit": limit,
            "time_period": time_period,
            "convert": "USD"
        }
        return self._request("/cryptocurrency/trending/gainers-losers", params, "v1")
    
    def get_trending_latest(self, limit: int = 10) -> Dict[str, Any]:
        """Get latest trending cryptocurrencies"""
        params = {
            "limit": limit
        }
        return self._request("/cryptocurrency/trending/latest", params, "v1")
    
    def get_price_performance(self, symbol: str, time_period: str = "24h", convert: str = "USD") -> Dict[str, Any]:
        """Get price performance stats"""
        params = {
            "symbol": symbol,
            "time_period": time_period,
            "convert": convert
        }
        return self._request("/cryptocurrency/price-performance-stats/latest", params, "v2")
    
    def get_exchange_map(self, listing_status: str = "active", limit: int = 100) -> Dict[str, Any]:
        """Get exchange map"""
        params = {
            "listing_status": listing_status,
            "limit": limit
        }
        return self._request("/exchange/map", params, "v1")
    
    def get_exchange_info(self, exchange_ids: List[str]) -> Dict[str, Any]:
        """Get exchange information"""
        params = {
            "id": ",".join(exchange_ids)
        }
        return self._request("/exchange/info", params, "v2")
    
    def get_exchange_listings(self, limit: int = 100, convert: str = "USD") -> Dict[str, Any]:
        """Get exchange listings with market data"""
        params = {
            "limit": limit,
            "convert": convert
        }
        return self._request("/exchange/listings/latest", params, "v1")
    
    def get_market_pairs(self, symbol: str, limit: int = 100, convert: str = "USD") -> Dict[str, Any]:
        """Get market pairs for a cryptocurrency"""
        params = {
            "symbol": symbol,
            "limit": limit,
            "convert": convert
        }
        return self._request("/cryptocurrency/market-pairs/latest", params, "v2")
    
    def get_ohlcv_historical(self, symbol: str, time_period: str = "daily", 
                           time_start: Optional[str] = None, time_end: Optional[str] = None) -> Dict[str, Any]:
        """Get historical OHLCV data"""
        params = {
            "symbol": symbol,
            "time_period": time_period,
            "convert": "USD"
        }
        if time_start:
            params["time_start"] = time_start
        if time_end:
            params["time_end"] = time_end
        
        return self._request("/cryptocurrency/ohlcv/historical", params, "v2")
    
    def get_categories(self, limit: int = 100) -> Dict[str, Any]:
        """Get cryptocurrency categories"""
        params = {
            "limit": limit
        }
        return self._request("/cryptocurrency/categories", params, "v1")
    
    def get_category(self, category_id: str, limit: int = 100) -> Dict[str, Any]:
        """Get cryptocurrencies in a category"""
        params = {
            "id": category_id,
            "limit": limit
        }
        return self._request("/cryptocurrency/category", params, "v1")
    
    def search(self, query: str) -> Dict[str, Any]:
        """Search for cryptocurrencies"""
        params = {
            "q": query
        }
        return self._request("/cryptocurrency/search", params, "v1")
    
    def get_fiat_map(self) -> Dict[str, Any]:
        """Get fiat currency map"""
        return self._request("/fiat/map", params={}, version="v1")
    
    def monitor_price(self, symbol: str, callback=None, interval: int = 60):
        """Monitor cryptocurrency price"""
        last_price = None
        while True:
            try:
                data = self.get_quotes([symbol])
                if "data" in data and symbol in data["data"]:
                    quote = data["data"][symbol][0]["quote"]["USD"]
                    current_price = quote["price"]
                    
                    if current_price != last_price:
                        price_data = {
                            "symbol": symbol,
                            "price": current_price,
                            "volume_24h": quote["volume_24h"],
                            "market_cap": quote["market_cap"],
                            "change_1h": quote["percent_change_1h"],
                            "change_24h": quote["percent_change_24h"],
                            "change_7d": quote["percent_change_7d"],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Save to database
                        try:
                            db = get_db()
                            db.execute_query(
                                """INSERT INTO market_data (source, symbol, price, volume_24h, market_cap, 
                                   change_1h, change_24h, change_7d, timestamp)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                ("coinmarketcap", symbol, current_price, 
                                 quote["volume_24h"], quote["market_cap"],
                                 quote["percent_change_1h"], quote["percent_change_24h"],
                                 quote["percent_change_7d"], datetime.now())
                            )
                        except Exception as e:
                            print(f"Database save error: {e}")
                        
                        if callback:
                            callback(price_data)
                        
                        last_price = current_price
                
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval * 2)