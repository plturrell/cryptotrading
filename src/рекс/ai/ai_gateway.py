"""
AI Gateway client for Claude-4-Sonnet via Anthropic
Advanced AI analysis for cryptocurrency trading
"""

import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import os
from ..database import get_db
from ..utils import rate_limiter

class AIGatewayClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize AI Gateway client with Claude-4-Sonnet"""
        self.api_key = api_key or os.getenv('AI_GATEWAY_API_KEY', 'FSoYtDBwMYn2FEfgcnXKYbPB')
        self.base_url = "https://api.anthropic.com/v1"
        self.model = "claude-4-sonnet"  # Using Claude-4-Sonnet for advanced analysis
        
        self.session = requests.Session()
        self.session.headers.update({
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        })
        
        # System prompt for crypto trading expertise
        self.system_prompt = """You are an expert cryptocurrency trading AI assistant specializing in:
        - Technical analysis and chart patterns
        - Market sentiment analysis
        - DeFi and DEX opportunities
        - Risk management strategies
        - Real-time trading signals
        
        Provide concise, actionable insights with confidence levels.
        Format responses with clear BUY/SELL/HOLD recommendations when applicable."""
    
    def analyze_market(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and provide trading recommendations"""
        # Rate limiting
        rate_limiter.wait_if_needed("ai_gateway")
        
        try:
            # Prepare context
            context = self._prepare_market_context(data)
            
            # Create message
            message = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze this crypto market data and provide trading recommendations:\n\n{json.dumps(context, indent=2)}"
                    }
                ],
                "system": self.system_prompt
            }
            
            # Make request
            response = self.session.post(
                f"{self.base_url}/messages",
                json=message,
                timeout=30
            )
            response.raise_for_status()
            
            # Record successful call
            rate_limiter.record_call("ai_gateway")
            
            # Parse response
            result = response.json()
            analysis = self._parse_analysis(result)
            
            # Save to database
            self._save_analysis(data.get('symbol', 'BTC'), analysis)
            
            return analysis
            
        except Exception as e:
            print(f"AI Gateway error: {e}")
            return {
                "error": str(e),
                "signal": "HOLD",
                "confidence": 0,
                "analysis": "Unable to analyze market data"
            }
    
    def get_trading_signals(self, symbol: str, timeframe: str = "4h",
                          technical_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate AI-powered trading signals"""
        rate_limiter.wait_if_needed("ai_gateway")
        
        try:
            # Build prompt
            prompt = f"""Analyze {symbol} on {timeframe} timeframe.
            
            Technical indicators:
            {json.dumps(technical_data, indent=2) if technical_data else 'Not provided'}
            
            Provide:
            1. Trading signal (BUY/SELL/HOLD)
            2. Confidence level (0-100%)
            3. Entry and exit points
            4. Stop loss and take profit levels
            5. Risk assessment
            """
            
            message = {
                "model": self.model,
                "max_tokens": 512,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "system": self.system_prompt
            }
            
            response = self.session.post(
                f"{self.base_url}/messages",
                json=message,
                timeout=30
            )
            response.raise_for_status()
            
            rate_limiter.record_call("ai_gateway")
            
            result = response.json()
            return self._parse_trading_signal(result)
            
        except Exception as e:
            print(f"Signal generation error: {e}")
            return {
                "error": str(e),
                "signal": "HOLD",
                "confidence": 0
            }
    
    def analyze_news_sentiment(self, news_items: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze crypto news sentiment using Claude-4"""
        rate_limiter.wait_if_needed("ai_gateway")
        
        try:
            # Format news for analysis
            news_text = "\n\n".join([
                f"Title: {item.get('title', '')}\n"
                f"Source: {item.get('source', '')}\n"
                f"Summary: {item.get('summary', '')}"
                for item in news_items[:10]  # Limit to 10 items
            ])
            
            prompt = f"""Analyze the sentiment of these crypto news items:

            {news_text}
            
            Provide:
            1. Overall market sentiment (BULLISH/BEARISH/NEUTRAL)
            2. Sentiment score (-100 to +100)
            3. Key themes and trends
            4. Impact on major cryptocurrencies
            5. Trading implications
            """
            
            message = {
                "model": self.model,
                "max_tokens": 768,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "system": self.system_prompt
            }
            
            response = self.session.post(
                f"{self.base_url}/messages",
                json=message,
                timeout=30
            )
            response.raise_for_status()
            
            rate_limiter.record_call("ai_gateway")
            
            result = response.json()
            return self._parse_sentiment(result)
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {
                "error": str(e),
                "sentiment": "NEUTRAL",
                "score": 0
            }
    
    def find_arbitrage_opportunities(self, dex_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use AI to identify complex arbitrage opportunities"""
        rate_limiter.wait_if_needed("ai_gateway")
        
        try:
            prompt = f"""Analyze these DEX trading pairs for arbitrage opportunities:

            {json.dumps(dex_data[:20], indent=2)}
            
            Identify:
            1. Direct arbitrage opportunities (same pair, different DEXs)
            2. Triangular arbitrage paths
            3. Cross-chain opportunities
            4. Estimated profit percentages
            5. Required capital and gas considerations
            6. Risk factors
            """
            
            message = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "system": self.system_prompt + "\n\nFocus on realistic, executable arbitrage opportunities."
            }
            
            response = self.session.post(
                f"{self.base_url}/messages",
                json=message,
                timeout=30
            )
            response.raise_for_status()
            
            rate_limiter.record_call("ai_gateway")
            
            result = response.json()
            return self._parse_arbitrage(result)
            
        except Exception as e:
            print(f"Arbitrage analysis error: {e}")
            return []
    
    def generate_trading_strategy(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized trading strategy"""
        rate_limiter.wait_if_needed("ai_gateway")
        
        try:
            prompt = f"""Create a personalized crypto trading strategy based on:

            User Profile:
            - Risk tolerance: {user_profile.get('risk_tolerance', 'medium')}
            - Capital: ${user_profile.get('capital', 10000)}
            - Experience: {user_profile.get('experience', 'intermediate')}
            - Goals: {user_profile.get('goals', 'steady growth')}
            - Time horizon: {user_profile.get('time_horizon', '6 months')}
            
            Provide:
            1. Portfolio allocation recommendations
            2. Entry and exit strategies
            3. Risk management rules
            4. Specific coins/tokens to consider
            5. DeFi opportunities
            6. Rebalancing schedule
            """
            
            message = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "system": self.system_prompt
            }
            
            response = self.session.post(
                f"{self.base_url}/messages",
                json=message,
                timeout=30
            )
            response.raise_for_status()
            
            rate_limiter.record_call("ai_gateway")
            
            result = response.json()
            return self._parse_strategy(result)
            
        except Exception as e:
            print(f"Strategy generation error: {e}")
            return {"error": str(e)}
    
    def _prepare_market_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare market data context for AI analysis"""
        return {
            "symbol": data.get("symbol", "BTC"),
            "price": data.get("price", 0),
            "volume_24h": data.get("volume_24h", 0),
            "change_24h": data.get("change_24h", 0),
            "technical_indicators": data.get("indicators", {}),
            "market_cap": data.get("market_cap", 0),
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_analysis(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response into structured analysis"""
        try:
            content = response['content'][0]['text']
            
            # Extract key information
            signal = "HOLD"
            confidence = 50
            
            # Simple parsing - could be enhanced with more sophisticated NLP
            content_lower = content.lower()
            if "buy" in content_lower and "sell" not in content_lower[:content_lower.index("buy")]:
                signal = "BUY"
            elif "sell" in content_lower and "buy" not in content_lower[:content_lower.index("sell")]:
                signal = "SELL"
            
            # Extract confidence if mentioned
            if "confidence" in content_lower:
                try:
                    # Look for percentage
                    import re
                    confidence_match = re.search(r'(\d+)%', content)
                    if confidence_match:
                        confidence = int(confidence_match.group(1))
                except:
                    pass
            
            return {
                "signal": signal,
                "confidence": confidence,
                "analysis": content,
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
        except Exception as e:
            print(f"Parse error: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0,
                "analysis": "Error parsing AI response",
                "error": str(e)
            }
    
    def _parse_trading_signal(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse trading signal from AI response"""
        try:
            content = response['content'][0]['text']
            
            # Initialize result
            result = {
                "signal": "HOLD",
                "confidence": 50,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "risk_assessment": "Medium",
                "reasoning": content
            }
            
            # Extract values using regex
            import re
            
            # Signal
            signal_match = re.search(r'(BUY|SELL|HOLD)', content, re.IGNORECASE)
            if signal_match:
                result["signal"] = signal_match.group(1).upper()
            
            # Confidence
            conf_match = re.search(r'(\d+)%?\s*confidence', content, re.IGNORECASE)
            if conf_match:
                result["confidence"] = int(conf_match.group(1))
            
            # Prices
            entry_match = re.search(r'entry.*?(\$?[\d,]+\.?\d*)', content, re.IGNORECASE)
            if entry_match:
                result["entry_price"] = float(entry_match.group(1).replace('$', '').replace(',', ''))
            
            stop_match = re.search(r'stop.*?loss.*?(\$?[\d,]+\.?\d*)', content, re.IGNORECASE)
            if stop_match:
                result["stop_loss"] = float(stop_match.group(1).replace('$', '').replace(',', ''))
            
            profit_match = re.search(r'take.*?profit.*?(\$?[\d,]+\.?\d*)', content, re.IGNORECASE)
            if profit_match:
                result["take_profit"] = float(profit_match.group(1).replace('$', '').replace(',', ''))
            
            return result
            
        except Exception as e:
            print(f"Signal parse error: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0,
                "error": str(e)
            }
    
    def _parse_sentiment(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse sentiment analysis from AI response"""
        try:
            content = response['content'][0]['text']
            
            result = {
                "sentiment": "NEUTRAL",
                "score": 0,
                "themes": [],
                "impact": {},
                "implications": content
            }
            
            # Extract sentiment
            import re
            sent_match = re.search(r'(BULLISH|BEARISH|NEUTRAL)', content, re.IGNORECASE)
            if sent_match:
                result["sentiment"] = sent_match.group(1).upper()
            
            # Extract score
            score_match = re.search(r'([-+]?\d+)\s*(?:score|sentiment)', content, re.IGNORECASE)
            if score_match:
                result["score"] = int(score_match.group(1))
            
            return result
            
        except Exception as e:
            print(f"Sentiment parse error: {e}")
            return {
                "sentiment": "NEUTRAL",
                "score": 0,
                "error": str(e)
            }
    
    def _parse_arbitrage(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse arbitrage opportunities from AI response"""
        try:
            content = response['content'][0]['text']
            opportunities = []
            
            # Simple extraction - could be enhanced
            lines = content.split('\n')
            current_opp = {}
            
            for line in lines:
                if 'profit' in line.lower() and '%' in line:
                    import re
                    profit_match = re.search(r'(\d+\.?\d*)%', line)
                    if profit_match and current_opp:
                        current_opp['profit_percent'] = float(profit_match.group(1))
                        opportunities.append(current_opp)
                        current_opp = {}
                elif any(dex in line.lower() for dex in ['uniswap', 'sushiswap', 'pancakeswap']):
                    current_opp['description'] = line.strip()
            
            return opportunities
            
        except Exception as e:
            print(f"Arbitrage parse error: {e}")
            return []
    
    def _parse_strategy(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse trading strategy from AI response"""
        try:
            content = response['content'][0]['text']
            
            return {
                "strategy": content,
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
            
        except Exception as e:
            print(f"Strategy parse error: {e}")
            return {"error": str(e)}
    
    def _save_analysis(self, symbol: str, analysis: Dict[str, Any]):
        """Save AI analysis to database"""
        try:
            db = get_db()
            db.save_ai_analysis(
                symbol=symbol,
                model=self.model,
                analysis_type="market",
                analysis=json.dumps(analysis),
                signal=analysis.get("signal"),
                confidence=analysis.get("confidence")
            )
        except Exception as e:
            print(f"Database save error: {e}")