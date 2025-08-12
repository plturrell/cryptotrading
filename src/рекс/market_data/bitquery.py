"""
Bitquery GraphQL API client for DEX data
Features: Real-time DEX data, mempool transactions
Coverage: Nearly all major DEXs
"""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import time
import os
from ..database import get_db

class BitqueryClient:
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://graphql.bitquery.io"
        self.api_key = api_key or os.getenv('BITQUERY_API_KEY')
        
        if not self.api_key:
            raise ValueError("Bitquery API key is required")
        
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        })
        
        self.last_request_time = 0
        self.rate_limit = 1.0  # 1 request per second
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute GraphQL query"""
        self._rate_limit()
        
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        try:
            response = self.session.post(self.base_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Bitquery API error: {e}")
            return {"error": str(e)}
    
    def get_dex_trades(self, network: str, token_address: Optional[str] = None, 
                      limit: int = 100, from_time: Optional[str] = None) -> Dict[str, Any]:
        """Get DEX trades for a token or all trades"""
        query = """
        query ($network: Network!, $token: String, $limit: Int!, $from: ISO8601DateTime) {
          ethereum(network: $network) {
            dexTrades(
              options: {desc: "block.timestamp.time", limit: $limit}
              date: {since: $from}
              baseCurrency: {is: $token}
            ) {
              transaction {
                hash
                gasPrice
                gasUsed
              }
              block {
                timestamp {
                  time(format: "%Y-%m-%d %H:%M:%S")
                }
                height
              }
              exchange {
                fullName
                address {
                  address
                }
              }
              baseCurrency {
                symbol
                address
              }
              quoteCurrency {
                symbol
                address
              }
              baseAmount
              quoteAmount
              trades: count
              quotePrice
              side
            }
          }
        }
        """
        
        variables = {
            "network": network,
            "limit": limit
        }
        if token_address:
            variables["token"] = token_address
        if from_time:
            variables["from"] = from_time
            
        return self._query(query, variables)
    
    def get_token_price(self, network: str, token_address: str, 
                       quote_currency: str = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2") -> Dict[str, Any]:
        """Get current token price from DEX"""
        query = """
        query ($network: Network!, $token: String!, $quoteCurrency: String!) {
          ethereum(network: $network) {
            dexTrades(
              options: {desc: "block.timestamp.time", limit: 1}
              baseCurrency: {is: $token}
              quoteCurrency: {is: $quoteCurrency}
            ) {
              block {
                timestamp {
                  time(format: "%Y-%m-%d %H:%M:%S")
                }
              }
              baseCurrency {
                symbol
                address
              }
              quoteCurrency {
                symbol
                address
              }
              quotePrice
              trades: count
              baseAmount
              quoteAmount
            }
          }
        }
        """
        
        variables = {
            "network": network,
            "token": token_address,
            "quoteCurrency": quote_currency
        }
        
        return self._query(query, variables)
    
    def get_dex_liquidity(self, network: str, exchange_address: str) -> Dict[str, Any]:
        """Get DEX pool liquidity"""
        query = """
        query ($network: Network!, $exchange: String!) {
          ethereum(network: $network) {
            address(address: {is: $exchange}) {
              balances {
                currency {
                  address
                  symbol
                  tokenType
                }
                value
              }
            }
          }
        }
        """
        
        variables = {
            "network": network,
            "exchange": exchange_address
        }
        
        return self._query(query, variables)
    
    def get_mempool_transactions(self, network: str = "ethereum") -> Dict[str, Any]:
        """Get unconfirmed mempool transactions"""
        query = """
        query ($network: Network!) {
          ethereum(network: $network) {
            transactions(
              options: {desc: "any", limit: 100}
              txSender: {}
              success: null
            ) {
              hash
              gasPrice
              gas
              nonce
              index
              value
              creates {
                address
              }
              to {
                address
              }
              sender {
                address
              }
            }
          }
        }
        """
        
        variables = {"network": network}
        return self._query(query, variables)
    
    def get_dex_candles(self, network: str, base_currency: str, quote_currency: str,
                       interval: int = 1440, limit: int = 100) -> Dict[str, Any]:
        """Get OHLCV candle data"""
        query = """
        query ($network: Network!, $base: String!, $quote: String!, $interval: Int!, $limit: Int!) {
          ethereum(network: $network) {
            dexTrades(
              options: {limit: $limit}
              baseCurrency: {is: $base}
              quoteCurrency: {is: $quote}
              timeInterval: {in: minutes, count: $interval}
            ) {
              timeInterval {
                minute(format: "%Y-%m-%d %H:%M:%S", count: $interval)
              }
              volume: quoteAmount
              high: quotePrice(calculate: maximum)
              low: quotePrice(calculate: minimum)
              open: minimum(of: block, get: quote_price)
              close: maximum(of: block, get: quote_price)
            }
          }
        }
        """
        
        variables = {
            "network": network,
            "base": base_currency,
            "quote": quote_currency,
            "interval": interval,
            "limit": limit
        }
        
        return self._query(query, variables)
    
    def get_top_traders(self, network: str, token_address: str, limit: int = 10) -> Dict[str, Any]:
        """Get top traders for a token"""
        query = """
        query ($network: Network!, $token: String!, $limit: Int!) {
          ethereum(network: $network) {
            dexTrades(
              baseCurrency: {is: $token}
              options: {limit: $limit}
            ) {
              taker {
                address
                annotation
              }
              maker {
                address
                annotation
              }
              baseAmount
              quoteAmount
              count
            }
          }
        }
        """
        
        variables = {
            "network": network,
            "token": token_address,
            "limit": limit
        }
        
        return self._query(query, variables)
    
    def get_arbitrage_opportunities(self, network: str, min_profit_percent: float = 1.0) -> Dict[str, Any]:
        """Find arbitrage opportunities across DEXs"""
        query = """
        query ($network: Network!) {
          ethereum(network: $network) {
            dexTrades(
              options: {desc: "block.timestamp.time", limit: 1000}
              time: {since: "2024-01-01"}
            ) {
              exchange {
                fullName
                address {
                  address
                }
              }
              baseCurrency {
                symbol
                address
              }
              quoteCurrency {
                symbol
                address
              }
              quotePrice
              baseAmount
              block {
                timestamp {
                  time
                }
              }
            }
          }
        }
        """
        
        result = self._query(query, {"network": network})
        
        # Process results to find arbitrage opportunities
        if "data" in result and result["data"]:
            trades = result["data"]["ethereum"]["dexTrades"]
            opportunities = []
            
            # Group by token pair
            pairs = {}
            for trade in trades:
                pair_key = f"{trade['baseCurrency']['address']}-{trade['quoteCurrency']['address']}"
                if pair_key not in pairs:
                    pairs[pair_key] = []
                pairs[pair_key].append(trade)
            
            # Find price differences
            for pair_key, pair_trades in pairs.items():
                if len(pair_trades) >= 2:
                    prices = [float(t["quotePrice"]) for t in pair_trades]
                    max_price = max(prices)
                    min_price = min(prices)
                    profit_percent = ((max_price - min_price) / min_price) * 100
                    
                    if profit_percent >= min_profit_percent:
                        opportunities.append({
                            "pair": pair_key,
                            "base_symbol": pair_trades[0]["baseCurrency"]["symbol"],
                            "quote_symbol": pair_trades[0]["quoteCurrency"]["symbol"],
                            "min_price": min_price,
                            "max_price": max_price,
                            "profit_percent": profit_percent,
                            "buy_exchange": next(t["exchange"]["fullName"] for t in pair_trades if float(t["quotePrice"]) == min_price),
                            "sell_exchange": next(t["exchange"]["fullName"] for t in pair_trades if float(t["quotePrice"]) == max_price)
                        })
            
            return {"opportunities": opportunities}
        
        return {"opportunities": []}
    
    def monitor_dex_trades(self, network: str, token_address: str, callback=None):
        """Monitor DEX trades in real-time"""
        while True:
            try:
                trades = self.get_dex_trades(network, token_address, limit=10)
                
                if "data" in trades and trades["data"]:
                    for trade in trades["data"]["ethereum"]["dexTrades"]:
                        trade_data = {
                            "network": network,
                            "token": token_address,
                            "exchange": trade["exchange"]["fullName"],
                            "base_amount": float(trade["baseAmount"]),
                            "quote_amount": float(trade["quoteAmount"]),
                            "price": float(trade["quotePrice"]),
                            "side": trade["side"],
                            "timestamp": trade["block"]["timestamp"]["time"],
                            "tx_hash": trade["transaction"]["hash"]
                        }
                        
                        # Save to database
                        try:
                            db = get_db()
                            db.execute_query(
                                """INSERT INTO dex_trades (source, network, token, exchange, 
                                   base_amount, quote_amount, price, side, tx_hash, timestamp)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                ("bitquery", network, token_address, trade["exchange"]["fullName"],
                                 trade_data["base_amount"], trade_data["quote_amount"],
                                 trade_data["price"], trade_data["side"], trade_data["tx_hash"],
                                 datetime.now())
                            )
                        except Exception as e:
                            print(f"Database save error: {e}")
                        
                        if callback:
                            callback(trade_data)
                
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)