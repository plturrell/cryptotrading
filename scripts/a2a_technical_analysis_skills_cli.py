#!/usr/bin/env python3
"""
A2A Technical Analysis Skills CLI - Advanced technical analysis calculations
Real implementation with comprehensive TA indicators and pattern recognition
"""

import os
import sys
import asyncio
import json
import click
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set environment variables for development
os.environ['ENVIRONMENT'] = 'development'
os.environ['SKIP_DB_INIT'] = 'true'

try:
    from src.cryptotrading.infrastructure.mcp.technical_analysis_skills_mcp_tools import TechnicalAnalysisSkillsMCPTools
    from src.cryptotrading.core.agents.specialized.technical_analysis.skill_1_momentum_indicators import MomentumIndicatorsSkill
    from src.cryptotrading.core.agents.specialized.technical_analysis.skill_2_momentum_volatility import MomentumVolatilitySkill
    from src.cryptotrading.core.agents.specialized.technical_analysis.skill_3_volume_analysis import VolumeAnalysisSkill
    from src.cryptotrading.core.agents.specialized.technical_analysis.skill_4_support_resistance import SupportResistanceSkill
    from src.cryptotrading.core.agents.specialized.technical_analysis.skill_6_harmonic_patterns import AdvancedPatternsSkill
    from src.cryptotrading.core.agents.specialized.technical_analysis.skill_7_comprehensive_system import ComprehensiveSystemSkill
    REAL_IMPLEMENTATION = True
except ImportError as e:
    print(f"⚠️ Using fallback implementation: {e}")
    REAL_IMPLEMENTATION = False

class TechnicalAnalysisSkillsAgent:
    """Technical Analysis Skills Agent with all TA capabilities"""
    
    def __init__(self):
        self.agent_id = "technical_analysis_skills_agent"
        self.capabilities = [
            'calculate_momentum_indicators', 'calculate_momentum_volatility',
            'analyze_volume_patterns', 'identify_support_resistance',
            'detect_chart_patterns', 'comprehensive_analysis'
        ]
        
        if REAL_IMPLEMENTATION:
            self.mcp_tools = TechnicalAnalysisSkillsMCPTools()
            self.momentum_indicators = MomentumIndicatorsSkill()
            self.momentum_volatility = MomentumVolatilitySkill()
            self.volume_analysis = VolumeAnalysisSkill()
            self.support_resistance = SupportResistanceSkill()
            self.harmonic_patterns = AdvancedPatternsSkill()
            self.comprehensive_system = ComprehensiveSystemSkill()
    
    def _generate_mock_ohlcv(self, symbol: str = "BTC-USD", days: int = 100) -> Dict[str, Any]:
        """Generate mock OHLCV data for testing"""
        import random
        
        base_price = 50000.0
        data = []
        current_price = base_price
        
        for i in range(days):
            # Random walk with trend
            change = random.uniform(-0.05, 0.05)
            current_price *= (1 + change)
            
            high = current_price * (1 + random.uniform(0, 0.03))
            low = current_price * (1 - random.uniform(0, 0.03))
            volume = random.uniform(1000000, 10000000)
            
            data.append({
                "timestamp": (datetime.now().timestamp() - (days - i) * 86400) * 1000,
                "open": current_price,
                "high": high,
                "low": low,
                "close": current_price,
                "volume": volume
            })
        
        return {"symbol": symbol, "data": data}
    
    async def calculate_momentum_indicators(self, market_data: str = None, 
                                         indicators: List[str] = None,
                                         periods: Dict[str, int] = None) -> Dict[str, Any]:
        """Calculate momentum technical indicators"""
        if not REAL_IMPLEMENTATION:
            return self._mock_momentum_indicators(indicators or ['SMA', 'EMA', 'RSI', 'MACD'])
        
        try:
            if not market_data:
                mock_data = self._generate_mock_ohlcv()
                market_data = json.dumps(mock_data)
            
            data = json.loads(market_data) if isinstance(market_data, str) else market_data
            result = await self.momentum_indicators.calculate_all(data, indicators, periods or {})
            
            return {
                "success": True,
                "indicators": result.get("indicators", {}),
                "signals": result.get("signals", []),
                "summary": result.get("summary", {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Momentum indicators calculation failed: {str(e)}"}
    
    def _mock_momentum_indicators(self, indicators: List[str]) -> Dict[str, Any]:
        """Mock momentum indicators calculation"""
        results = {}
        for indicator in indicators:
            if indicator == 'SMA':
                results[indicator] = {"value": 51234.56, "signal": "bullish", "strength": 0.7}
            elif indicator == 'EMA':
                results[indicator] = {"value": 51456.78, "signal": "bullish", "strength": 0.8}
            elif indicator == 'RSI':
                results[indicator] = {"value": 65.4, "signal": "neutral", "strength": 0.6}
            elif indicator == 'MACD':
                results[indicator] = {"macd": 245.67, "signal": 198.34, "histogram": 47.33, "signal": "bullish"}
        
        return {
            "success": True,
            "indicators": results,
            "signals": ["bullish_momentum", "overbought_warning"],
            "summary": {"overall_signal": "bullish", "confidence": 0.75},
            "mock": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def calculate_momentum_volatility(self, market_data: str = None,
                                          period: int = 14,
                                          include_bollinger: bool = True) -> Dict[str, Any]:
        """Calculate momentum and volatility indicators"""
        if not REAL_IMPLEMENTATION:
            return self._mock_momentum_volatility(period, include_bollinger)
        
        try:
            if not market_data:
                mock_data = self._generate_mock_ohlcv()
                market_data = json.dumps(mock_data)
            
            data = json.loads(market_data) if isinstance(market_data, str) else market_data
            result = await self.momentum_volatility.calculate_all(data, period, include_bollinger)
            
            return {
                "success": True,
                "volatility": result.get("volatility", {}),
                "momentum": result.get("momentum", {}),
                "bollinger_bands": result.get("bollinger_bands", {}) if include_bollinger else None,
                "signals": result.get("signals", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Momentum volatility calculation failed: {str(e)}"}
    
    def _mock_momentum_volatility(self, period: int, include_bollinger: bool) -> Dict[str, Any]:
        """Mock momentum volatility calculation"""
        result = {
            "success": True,
            "volatility": {
                "historical_volatility": 0.048,
                "realized_volatility": 0.052,
                "volatility_trend": "increasing"
            },
            "momentum": {
                "roc": 2.34,
                "momentum_oscillator": 0.67,
                "trend_strength": 0.8
            },
            "signals": ["high_volatility", "bullish_momentum"],
            "mock": True,
            "timestamp": datetime.now().isoformat()
        }
        
        if include_bollinger:
            result["bollinger_bands"] = {
                "upper": 52456.78,
                "middle": 51234.56,
                "lower": 50012.34,
                "bandwidth": 0.048,
                "percent_b": 0.65
            }
        
        return result
    
    async def analyze_volume_patterns(self, market_data: str = None,
                                    lookback_period: int = 20) -> Dict[str, Any]:
        """Analyze volume patterns and trends"""
        if not REAL_IMPLEMENTATION:
            return self._mock_volume_analysis(lookback_period)
        
        try:
            if not market_data:
                mock_data = self._generate_mock_ohlcv()
                market_data = json.dumps(mock_data)
            
            data = json.loads(market_data) if isinstance(market_data, str) else market_data
            result = await self.volume_analysis.analyze_patterns(data, lookback_period)
            
            return {
                "success": True,
                "volume_profile": result.get("volume_profile", {}),
                "patterns": result.get("patterns", []),
                "anomalies": result.get("anomalies", []),
                "trend": result.get("trend", {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Volume analysis failed: {str(e)}"}
    
    def _mock_volume_analysis(self, lookback_period: int) -> Dict[str, Any]:
        """Mock volume analysis"""
        return {
            "success": True,
            "volume_profile": {
                "average_volume": 5234567,
                "volume_trend": "increasing",
                "relative_volume": 1.23,
                "volume_sma": 4987654
            },
            "patterns": [
                {"type": "volume_surge", "strength": 0.8, "date": "2024-01-15"},
                {"type": "accumulation", "strength": 0.6, "date": "2024-01-14"}
            ],
            "anomalies": [
                {"type": "unusual_volume", "multiplier": 3.2, "date": "2024-01-13"}
            ],
            "trend": {
                "direction": "bullish",
                "strength": 0.75,
                "confidence": 0.82
            },
            "mock": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def identify_support_resistance(self, market_data: str = None,
                                        min_touches: int = 2,
                                        tolerance: float = 0.01) -> Dict[str, Any]:
        """Identify support and resistance levels"""
        if not REAL_IMPLEMENTATION:
            return self._mock_support_resistance(min_touches, tolerance)
        
        try:
            if not market_data:
                mock_data = self._generate_mock_ohlcv()
                market_data = json.dumps(mock_data)
            
            data = json.loads(market_data) if isinstance(market_data, str) else market_data
            result = await self.support_resistance.identify_levels(data, min_touches, tolerance)
            
            return {
                "success": True,
                "support_levels": result.get("support_levels", []),
                "resistance_levels": result.get("resistance_levels", []),
                "current_position": result.get("current_position", {}),
                "signals": result.get("signals", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Support resistance analysis failed: {str(e)}"}
    
    def _mock_support_resistance(self, min_touches: int, tolerance: float) -> Dict[str, Any]:
        """Mock support resistance analysis"""
        return {
            "success": True,
            "support_levels": [
                {"price": 49876.54, "strength": 0.85, "touches": 4, "age_days": 12},
                {"price": 48234.67, "strength": 0.72, "touches": 3, "age_days": 8},
                {"price": 46789.12, "strength": 0.68, "touches": 2, "age_days": 5}
            ],
            "resistance_levels": [
                {"price": 52456.78, "strength": 0.92, "touches": 5, "age_days": 15},
                {"price": 53789.45, "strength": 0.78, "touches": 3, "age_days": 10},
                {"price": 55123.67, "strength": 0.65, "touches": 2, "age_days": 3}
            ],
            "current_position": {
                "price": 51234.56,
                "nearest_support": 49876.54,
                "nearest_resistance": 52456.78,
                "position": "between_levels"
            },
            "signals": ["approaching_resistance", "strong_support_below"],
            "mock": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def detect_chart_patterns(self, market_data: str = None,
                                  pattern_types: List[str] = None,
                                  sensitivity: float = 0.5) -> Dict[str, Any]:
        """Detect advanced chart patterns"""
        if not REAL_IMPLEMENTATION:
            return self._mock_pattern_detection(pattern_types or ['head_shoulders', 'triangles', 'flags'])
        
        try:
            if not market_data:
                mock_data = self._generate_mock_ohlcv()
                market_data = json.dumps(mock_data)
            
            data = json.loads(market_data) if isinstance(market_data, str) else market_data
            result = await self.harmonic_patterns.detect_patterns(data, pattern_types, sensitivity)
            
            return {
                "success": True,
                "patterns_detected": result.get("patterns", []),
                "pattern_count": len(result.get("patterns", [])),
                "confidence_scores": result.get("confidence_scores", {}),
                "signals": result.get("signals", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Pattern detection failed: {str(e)}"}
    
    def _mock_pattern_detection(self, pattern_types: List[str]) -> Dict[str, Any]:
        """Mock pattern detection"""
        patterns = []
        for pattern_type in pattern_types[:3]:  # Limit to 3 patterns
            if pattern_type == 'head_shoulders':
                patterns.append({
                    "type": "head_and_shoulders",
                    "confidence": 0.78,
                    "completion": 0.85,
                    "target_price": 48567.89,
                    "signal": "bearish"
                })
            elif pattern_type == 'triangles':
                patterns.append({
                    "type": "ascending_triangle",
                    "confidence": 0.82,
                    "completion": 0.92,
                    "target_price": 54321.09,
                    "signal": "bullish"
                })
            elif pattern_type == 'flags':
                patterns.append({
                    "type": "bull_flag",
                    "confidence": 0.71,
                    "completion": 0.76,
                    "target_price": 53456.78,
                    "signal": "bullish"
                })
        
        return {
            "success": True,
            "patterns_detected": patterns,
            "pattern_count": len(patterns),
            "confidence_scores": {p["type"]: p["confidence"] for p in patterns},
            "signals": [p["signal"] + "_pattern_detected" for p in patterns],
            "mock": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def comprehensive_analysis(self, market_data: str = None,
                                   analysis_depth: str = "full") -> Dict[str, Any]:
        """Run comprehensive technical analysis"""
        if not REAL_IMPLEMENTATION:
            return self._mock_comprehensive_analysis(analysis_depth)
        
        try:
            if not market_data:
                mock_data = self._generate_mock_ohlcv()
                market_data = json.dumps(mock_data)
            
            data = json.loads(market_data) if isinstance(market_data, str) else market_data
            result = await self.comprehensive_system.full_analysis(data, analysis_depth)
            
            return {
                "success": True,
                "overall_signal": result.get("overall_signal", {}),
                "momentum_analysis": result.get("momentum", {}),
                "volatility_analysis": result.get("volatility", {}),
                "volume_analysis": result.get("volume", {}),
                "support_resistance": result.get("support_resistance", {}),
                "patterns": result.get("patterns", []),
                "recommendations": result.get("recommendations", []),
                "confidence": result.get("confidence", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Comprehensive analysis failed: {str(e)}"}
    
    def _mock_comprehensive_analysis(self, analysis_depth: str) -> Dict[str, Any]:
        """Mock comprehensive analysis"""
        return {
            "success": True,
            "overall_signal": {
                "direction": "bullish",
                "strength": 0.76,
                "timeframe": "medium_term"
            },
            "momentum_analysis": {
                "trend": "uptrend",
                "strength": 0.82,
                "rsi": 65.4,
                "macd_signal": "bullish"
            },
            "volatility_analysis": {
                "level": "moderate",
                "trend": "increasing",
                "bollinger_position": "upper_half"
            },
            "volume_analysis": {
                "trend": "bullish",
                "confirmation": True,
                "relative_volume": 1.23
            },
            "support_resistance": {
                "nearest_support": 49876.54,
                "nearest_resistance": 52456.78,
                "position": "approaching_resistance"
            },
            "patterns": [
                {"type": "bull_flag", "confidence": 0.78},
                {"type": "ascending_triangle", "confidence": 0.71}
            ],
            "recommendations": [
                "Consider taking partial profits near resistance",
                "Watch for breakout above 52,500",
                "Stop loss below 49,800"
            ],
            "confidence": 0.78,
            "mock": True,
            "timestamp": datetime.now().isoformat()
        }

# Global agent instance
agent = TechnicalAnalysisSkillsAgent()

def async_command(f):
    """Decorator to run async commands"""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper

@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """A2A Technical Analysis Skills CLI - Advanced TA calculations"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if not REAL_IMPLEMENTATION:
        click.echo("⚠️ Running in fallback mode - using mock analysis")

@cli.command()
@click.option('--data-file', type=click.Path(), help='JSON file with market data')
@click.option('--indicators', help='Comma-separated indicator list (SMA,EMA,RSI,MACD)')
@click.option('--periods', help='JSON string with period settings')
@click.pass_context
@async_command
async def momentum(ctx, data_file, indicators, periods):
    """Calculate momentum indicators"""
    try:
        market_data = None
        if data_file:
            with open(data_file, 'r') as f:
                market_data = f.read()
        
        indicator_list = indicators.split(',') if indicators else None
        period_dict = json.loads(periods) if periods else None
        
        result = await agent.calculate_momentum_indicators(market_data, indicator_list, period_dict)
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo("📈 Momentum Indicators Analysis")
        click.echo("=" * 50)
        
        for indicator, data in result.get('indicators', {}).items():
            click.echo(f"{indicator}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    click.echo(f"  {key}: {value}")
            else:
                click.echo(f"  Value: {data}")
            click.echo()
        
        signals = result.get('signals', [])
        if signals:
            click.echo(f"Signals: {', '.join(signals)}")
        
        summary = result.get('summary', {})
        if summary:
            click.echo(f"Overall Signal: {summary.get('overall_signal', 'N/A')}")
            click.echo(f"Confidence: {summary.get('confidence', 0):.2f}")
        
        if result.get('mock'):
            click.echo("🔄 Mock analysis - enable real implementation for live calculations")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error calculating momentum indicators: {e}", err=True)

@cli.command()
@click.option('--data-file', type=click.Path(), help='JSON file with market data')
@click.option('--period', default=14, help='Calculation period')
@click.option('--no-bollinger', is_flag=True, help='Exclude Bollinger Bands')
@click.pass_context
@async_command
async def volatility(ctx, data_file, period, no_bollinger):
    """Calculate momentum and volatility indicators"""
    try:
        market_data = None
        if data_file:
            with open(data_file, 'r') as f:
                market_data = f.read()
        
        result = await agent.calculate_momentum_volatility(
            market_data, period, not no_bollinger
        )
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo("📊 Momentum & Volatility Analysis")
        click.echo("=" * 50)
        
        volatility = result.get('volatility', {})
        if volatility:
            click.echo("Volatility Metrics:")
            for key, value in volatility.items():
                click.echo(f"  {key.replace('_', ' ').title()}: {value}")
            click.echo()
        
        momentum = result.get('momentum', {})
        if momentum:
            click.echo("Momentum Metrics:")
            for key, value in momentum.items():
                click.echo(f"  {key.replace('_', ' ').title()}: {value}")
            click.echo()
        
        bollinger = result.get('bollinger_bands')
        if bollinger:
            click.echo("Bollinger Bands:")
            click.echo(f"  Upper: {bollinger.get('upper')}")
            click.echo(f"  Middle: {bollinger.get('middle')}")
            click.echo(f"  Lower: {bollinger.get('lower')}")
            click.echo(f"  %B: {bollinger.get('percent_b'):.2f}")
            click.echo()
        
        signals = result.get('signals', [])
        if signals:
            click.echo(f"Signals: {', '.join(signals)}")
        
        if result.get('mock'):
            click.echo("🔄 Mock analysis - enable real implementation for live calculations")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error calculating volatility: {e}", err=True)

@cli.command()
@click.option('--data-file', type=click.Path(), help='JSON file with market data')
@click.option('--lookback', default=20, help='Lookback period')
@click.pass_context
@async_command
async def volume(ctx, data_file, lookback):
    """Analyze volume patterns"""
    try:
        market_data = None
        if data_file:
            with open(data_file, 'r') as f:
                market_data = f.read()
        
        result = await agent.analyze_volume_patterns(market_data, lookback)
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo("📊 Volume Pattern Analysis")
        click.echo("=" * 50)
        
        profile = result.get('volume_profile', {})
        if profile:
            click.echo("Volume Profile:")
            click.echo(f"  Average Volume: {profile.get('average_volume'):,}")
            click.echo(f"  Relative Volume: {profile.get('relative_volume'):.2f}x")
            click.echo(f"  Trend: {profile.get('volume_trend')}")
            click.echo()
        
        patterns = result.get('patterns', [])
        if patterns:
            click.echo("Volume Patterns:")
            for pattern in patterns:
                click.echo(f"  {pattern['type']} (strength: {pattern['strength']:.2f})")
            click.echo()
        
        anomalies = result.get('anomalies', [])
        if anomalies:
            click.echo("Volume Anomalies:")
            for anomaly in anomalies:
                click.echo(f"  {anomaly['type']} ({anomaly.get('multiplier', 'N/A')}x normal)")
            click.echo()
        
        trend = result.get('trend', {})
        if trend:
            click.echo(f"Volume Trend: {trend.get('direction')} (confidence: {trend.get('confidence', 0):.2f})")
        
        if result.get('mock'):
            click.echo("🔄 Mock analysis - enable real implementation for live calculations")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error analyzing volume: {e}", err=True)

@cli.command()
@click.option('--data-file', type=click.Path(), help='JSON file with market data')
@click.option('--min-touches', default=2, help='Minimum touches for level')
@click.option('--tolerance', default=0.01, help='Price tolerance (as decimal)')
@click.pass_context
@async_command
async def levels(ctx, data_file, min_touches, tolerance):
    """Identify support and resistance levels"""
    try:
        market_data = None
        if data_file:
            with open(data_file, 'r') as f:
                market_data = f.read()
        
        result = await agent.identify_support_resistance(market_data, min_touches, tolerance)
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo("📊 Support & Resistance Analysis")
        click.echo("=" * 50)
        
        resistance_levels = result.get('resistance_levels', [])
        if resistance_levels:
            click.echo("Resistance Levels:")
            for level in resistance_levels:
                click.echo(f"  ${level['price']:,.2f} (strength: {level['strength']:.2f}, touches: {level['touches']})")
            click.echo()
        
        support_levels = result.get('support_levels', [])
        if support_levels:
            click.echo("Support Levels:")
            for level in support_levels:
                click.echo(f"  ${level['price']:,.2f} (strength: {level['strength']:.2f}, touches: {level['touches']})")
            click.echo()
        
        current = result.get('current_position', {})
        if current:
            click.echo(f"Current Position: ${current.get('price', 0):,.2f}")
            click.echo(f"Nearest Support: ${current.get('nearest_support', 0):,.2f}")
            click.echo(f"Nearest Resistance: ${current.get('nearest_resistance', 0):,.2f}")
            click.echo()
        
        signals = result.get('signals', [])
        if signals:
            click.echo(f"Signals: {', '.join(signals)}")
        
        if result.get('mock'):
            click.echo("🔄 Mock analysis - enable real implementation for live calculations")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error identifying levels: {e}", err=True)

@cli.command()
@click.option('--data-file', type=click.Path(), help='JSON file with market data')
@click.option('--patterns', help='Comma-separated pattern types')
@click.option('--sensitivity', default=0.5, help='Detection sensitivity (0-1)')
@click.pass_context
@async_command
async def patterns(ctx, data_file, patterns, sensitivity):
    """Detect chart patterns"""
    try:
        market_data = None
        if data_file:
            with open(data_file, 'r') as f:
                market_data = f.read()
        
        pattern_types = patterns.split(',') if patterns else None
        
        result = await agent.detect_chart_patterns(market_data, pattern_types, sensitivity)
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo("🔍 Chart Pattern Detection")
        click.echo("=" * 50)
        
        detected = result.get('patterns_detected', [])
        click.echo(f"Patterns Found: {result.get('pattern_count', 0)}")
        click.echo()
        
        for pattern in detected:
            click.echo(f"📈 {pattern['type'].replace('_', ' ').title()}")
            click.echo(f"  Confidence: {pattern['confidence']:.2f}")
            click.echo(f"  Completion: {pattern['completion']:.1%}")
            click.echo(f"  Target Price: ${pattern.get('target_price', 0):,.2f}")
            click.echo(f"  Signal: {pattern['signal']}")
            click.echo()
        
        signals = result.get('signals', [])
        if signals:
            click.echo(f"Pattern Signals: {', '.join(signals)}")
        
        if result.get('mock'):
            click.echo("🔄 Mock analysis - enable real implementation for live calculations")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error detecting patterns: {e}", err=True)

@cli.command()
@click.option('--data-file', type=click.Path(), help='JSON file with market data')
@click.option('--depth', default='full', type=click.Choice(['basic', 'standard', 'full']),
              help='Analysis depth')
@click.pass_context
@async_command
async def comprehensive(ctx, data_file, depth):
    """Run comprehensive technical analysis"""
    try:
        market_data = None
        if data_file:
            with open(data_file, 'r') as f:
                market_data = f.read()
        
        result = await agent.comprehensive_analysis(market_data, depth)
        
        if result.get('error'):
            click.echo(f"❌ Error: {result['error']}", err=True)
            return
        
        click.echo("🎯 Comprehensive Technical Analysis")
        click.echo("=" * 60)
        
        overall = result.get('overall_signal', {})
        if overall:
            click.echo(f"Overall Signal: {overall.get('direction', 'N/A').upper()}")
            click.echo(f"Strength: {overall.get('strength', 0):.2f}")
            click.echo(f"Timeframe: {overall.get('timeframe', 'N/A').replace('_', ' ').title()}")
            click.echo(f"Confidence: {result.get('confidence', 0):.2f}")
            click.echo()
        
        # Momentum
        momentum = result.get('momentum_analysis', {})
        if momentum:
            click.echo("📈 Momentum Analysis:")
            click.echo(f"  Trend: {momentum.get('trend', 'N/A')}")
            click.echo(f"  Strength: {momentum.get('strength', 0):.2f}")
            click.echo(f"  RSI: {momentum.get('rsi', 0):.1f}")
            click.echo(f"  MACD: {momentum.get('macd_signal', 'N/A')}")
            click.echo()
        
        # Volume
        volume = result.get('volume_analysis', {})
        if volume:
            click.echo("📊 Volume Analysis:")
            click.echo(f"  Trend: {volume.get('trend', 'N/A')}")
            click.echo(f"  Confirmation: {'Yes' if volume.get('confirmation') else 'No'}")
            click.echo(f"  Relative Volume: {volume.get('relative_volume', 0):.2f}x")
            click.echo()
        
        # Support/Resistance
        sr = result.get('support_resistance', {})
        if sr:
            click.echo("📊 Support & Resistance:")
            click.echo(f"  Nearest Support: ${sr.get('nearest_support', 0):,.2f}")
            click.echo(f"  Nearest Resistance: ${sr.get('nearest_resistance', 0):,.2f}")
            click.echo(f"  Position: {sr.get('position', 'N/A').replace('_', ' ').title()}")
            click.echo()
        
        # Patterns
        patterns = result.get('patterns', [])
        if patterns:
            click.echo("🔍 Pattern Detection:")
            for pattern in patterns:
                click.echo(f"  {pattern['type'].replace('_', ' ').title()} (confidence: {pattern['confidence']:.2f})")
            click.echo()
        
        # Recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            click.echo("💡 Recommendations:")
            for rec in recommendations:
                click.echo(f"  • {rec}")
            click.echo()
        
        if result.get('mock'):
            click.echo("🔄 Mock analysis - enable real implementation for live calculations")
            
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result.get('timestamp')}")
            
    except Exception as e:
        click.echo(f"Error running comprehensive analysis: {e}", err=True)

@cli.command()
@click.pass_context
def capabilities(ctx):
    """List agent capabilities"""
    click.echo("🔧 Technical Analysis Skills Agent Capabilities:")
    click.echo()
    for i, capability in enumerate(agent.capabilities, 1):
        click.echo(f"{i:2d}. {capability.replace('_', ' ').title()}")

@cli.command()
@click.pass_context
def status(ctx):
    """Get agent status and health"""
    click.echo("🏥 Technical Analysis Skills Agent Status:")
    click.echo(f"Agent ID: {agent.agent_id}")
    click.echo(f"Capabilities: {len(agent.capabilities)}")
    click.echo(f"Implementation: {'Real' if REAL_IMPLEMENTATION else 'Fallback'}")
    click.echo("Status: ✅ ACTIVE")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")

if __name__ == '__main__':
    cli()