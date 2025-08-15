"""
Perplexity AI client for rex.com - Real-time crypto intelligence
"""

import os
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime

class PerplexityClient:
    def __init__(self):
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def search_crypto_news(self, symbol: str) -> Dict[str, any]:
        """Search latest crypto news and sentiment"""
        query = f"Latest {symbol} cryptocurrency news price analysis sentiment today"
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "pplx-7b-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a cryptocurrency market analyst. Provide concise trading insights."
                        },
                        {
                            "role": "user", 
                            "content": query
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 300
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "analysis": data['choices'][0]['message']['content'],
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol
                }
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_market_conditions(self, pairs: List[str]) -> Dict[str, any]:
        """Analyze multiple crypto pairs market conditions"""
        query = f"Current market analysis for {', '.join(pairs)} including support/resistance levels and trading volume"
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "pplx-70b-online",
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    "temperature": 0.1
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to analyze: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_trading_signals(self, symbol: str, timeframe: str = "4h") -> Dict[str, any]:
        """Get AI-powered trading signals"""
        query = f"Technical analysis {symbol} {timeframe} timeframe: RSI, MACD, moving averages. Should I buy, sell, or hold?"
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "pplx-7b-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a trading signal generator. Provide clear BUY/SELL/HOLD signals with reasoning."
                        },
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Parse signal from response
                signal = "HOLD"
                if "BUY" in content.upper() and "SELL" not in content.upper()[:50]:
                    signal = "BUY"
                elif "SELL" in content.upper() and "BUY" not in content.upper()[:50]:
                    signal = "SELL"
                
                return {
                    "signal": signal,
                    "analysis": content,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}