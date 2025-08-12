"""
CoinGecko API client for crypto market data
Free tier: Available with rate limits
Coverage: 200+ networks, DEX + CEX data
"""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import os
from ..database import get_db
from ..utils import rate_limiter

class CoinGeckoClient:
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_base_url = "https://pro-api.coingecko.com/api/v3"
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY')
        self.session = requests.Session()
        self.session.headers['Accept'] = 'application/json'
        
        # Use pro API if key is provided
        if self.api_key:
            self.base_url = self.pro_base_url
            self.session.headers['x-cg-pro-api-key'] = self.api_key
        
    def _rate_limit(self):
        """Enforce rate limiting using global rate limiter"""
        rate_limiter.wait_if_needed("coingecko")
        rate_limiter.record_call("coingecko")
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make rate-limited API request"""
        self._rate_limit()
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"CoinGecko API error: {e}")
            return {"error": str(e)}
    
    def get_price(self, coin_ids: List[str], vs_currencies: List[str] = ["usd"]) -> Dict[str, Any]:
        """Get current price for coins"""
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": ",".join(vs_currencies),
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true"
        }
        return self._request("/simple/price", params)
    
    def get_coin_data(self, coin_id: str) -> Dict[str, Any]:
        """Get detailed coin data"""
        params = {
            "localization": "false",
            "tickers": "true",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false"
        }
        return self._request(f"/coins/{coin_id}", params)
    
    def get_coin_market_chart(self, coin_id: str, vs_currency: str = "usd", days: int = 7) -> Dict[str, Any]:
        """Get historical market data"""
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        return self._request(f"/coins/{coin_id}/market_chart", params)
    
    def get_coin_ohlc(self, coin_id: str, vs_currency: str = "usd", days: int = 7) -> Dict[str, Any]:
        """Get OHLC data"""
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        return self._request(f"/coins/{coin_id}/ohlc", params)
    
    def get_trending_coins(self) -> Dict[str, Any]:
        """Get trending coins"""
        return self._request("/search/trending")
    
    def get_global_data(self) -> Dict[str, Any]:
        """Get global crypto market data"""
        return self._request("/global")
    
    def get_defi_data(self) -> Dict[str, Any]:
        """Get global DeFi market data"""
        return self._request("/global/decentralized_finance_defi")
    
    def get_exchanges(self, per_page: int = 100, page: int = 1) -> Dict[str, Any]:
        """Get list of exchanges"""
        params = {
            "per_page": per_page,
            "page": page
        }
        return self._request("/exchanges", params)
    
    def get_exchange_data(self, exchange_id: str) -> Dict[str, Any]:
        """Get specific exchange data"""
        return self._request(f"/exchanges/{exchange_id}")
    
    def get_exchange_tickers(self, exchange_id: str, coin_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get exchange tickers"""
        params = {}
        if coin_ids:
            params["coin_ids"] = ",".join(coin_ids)
        return self._request(f"/exchanges/{exchange_id}/tickers", params)
    
    def search_coins(self, query: str) -> Dict[str, Any]:
        """Search for coins"""
        params = {"query": query}
        return self._request("/search", params)
    
    def get_coin_by_contract(self, platform: str, contract_address: str) -> Dict[str, Any]:
        """Get coin data by contract address"""
        return self._request(f"/coins/{platform}/contract/{contract_address}")
    
    def get_categories(self) -> Dict[str, Any]:
        """Get all coin categories"""
        return self._request("/coins/categories/list")
    
    def get_category_data(self, category_id: str) -> Dict[str, Any]:
        """Get coins in a specific category"""
        params = {"category": category_id}
        return self._request("/coins/markets", params)
    
    def get_derivatives(self) -> Dict[str, Any]:
        """Get derivatives market data"""
        return self._request("/derivatives")
    
    def get_nfts(self, per_page: int = 100, page: int = 1) -> Dict[str, Any]:
        """Get NFT market data"""
        params = {
            "per_page": per_page,
            "page": page
        }
        return self._request("/nfts/list", params)
    
    def get_token_price_by_contract(self, platform: str, contract_addresses: List[str]) -> Dict[str, Any]:
        """Get token prices by contract addresses"""
        params = {
            "contract_addresses": ",".join(contract_addresses),
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true"
        }
        return self._request(f"/simple/token_price/{platform}", params)
    
    def monitor_price(self, coin_id: str, callback=None, interval: int = 60):
        """Monitor coin price changes"""
        last_price = None
        while True:
            try:
                data = self.get_price([coin_id])
                if coin_id in data:
                    current_price = data[coin_id]["usd"]
                    if current_price != last_price:
                        price_data = {
                            "coin_id": coin_id,
                            "price": current_price,
                            "market_cap": data[coin_id].get("usd_market_cap", 0),
                            "volume_24h": data[coin_id].get("usd_24h_vol", 0),
                            "change_24h": data[coin_id].get("usd_24h_change", 0),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Save to database
                        try:
                            db = get_db()
                            db.execute_query(
                                """INSERT INTO market_data (source, symbol, price, volume_24h, market_cap, change_24h, timestamp)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                ("coingecko", coin_id, current_price, 
                                 data[coin_id].get("usd_24h_vol", 0),
                                 data[coin_id].get("usd_market_cap", 0),
                                 data[coin_id].get("usd_24h_change", 0),
                                 datetime.now())
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