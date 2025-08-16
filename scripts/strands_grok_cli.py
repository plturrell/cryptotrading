#!/usr/bin/env python3
"""
Strands-Grok CLI - AI-enhanced code analysis using Strands + Glean + Grok
"""

import asyncio
import sys
import json
import logging
import os
from pathlib import Path
import click
from typing import Dict, Any, Optional

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from cryptotrading.core.agents.specialized.strands_glean_agent import (
        create_strands_glean_agent,
        StrandsGleanAgent
    )
    STRANDS_GLEAN_AVAILABLE = True
except ImportError as e:
    STRANDS_GLEAN_AVAILABLE = False
    print(f"Warning: StrandsGleanAgent not available: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GrokEnhancedAnalyzer:
    """AI-enhanced code analyzer using Grok API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROK_API_KEY')
        self.base_url = "https://api.x.ai/v1"  # Grok API endpoint
        
    async def enhance_analysis(self, glean_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Enhance Glean results with Grok AI analysis"""
        if not self.api_key:
            return {
                "status": "warning", 
                "message": "Grok API key not available",
                "original_results": glean_results
            }
        
        try:
            # Prepare context for Grok
            context = {
                "query": query,
                "glean_analysis": glean_results,
                "analysis_type": glean_results.get("type", "unknown")
            }
            
            # Create prompt for Grok
            prompt = self._create_analysis_prompt(context)
            
            # Call Grok API (mock implementation for now)
            grok_response = await self._call_grok_api(prompt)
            
            return {
                "status": "success",
                "original_glean": glean_results,
                "grok_enhancement": grok_response,
                "combined_insights": self._combine_insights(glean_results, grok_response)
            }
            
        except Exception as e:
            logger.error(f"Grok enhancement failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "original_results": glean_results
            }
    
    def _create_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Create an effective prompt for Grok analysis"""
        query = context["query"]
        glean_data = context["glean_analysis"]
        
        prompt = f"""
Analyze this code analysis request and provide enhanced insights:

QUERY: {query}

GLEAN ANALYSIS RESULTS:
{json.dumps(glean_data, indent=2)}

Please provide:
1. Summary of findings in plain English
2. Potential issues or improvements identified
3. Architectural recommendations
4. Security considerations if applicable
5. Performance implications
6. Maintainability suggestions

Focus on actionable insights that would help a developer understand and improve their code.
"""
        return prompt
    
    async def _call_grok_api(self, prompt: str) -> Dict[str, Any]:
        """Call Grok API (mock implementation)"""
        # This is a mock implementation - replace with actual Grok API call
        # For now, we'll simulate intelligent analysis
        
        await asyncio.sleep(0.1)  # Simulate API call delay
        
        # Mock intelligent response based on prompt analysis
        if "dependencies" in prompt.lower():
            return {
                "summary": "Dependencies analysis reveals potential coupling issues",
                "recommendations": [
                    "Consider using dependency injection patterns",
                    "Review circular dependencies",
                    "Implement interface segregation"
                ],
                "risk_level": "medium",
                "confidence": 0.85
            }
        elif "symbol" in prompt.lower():
            return {
                "summary": "Symbol search indicates potential naming inconsistencies",
                "recommendations": [
                    "Consider consistent naming conventions",
                    "Review public API surface",
                    "Consider symbol visibility"
                ],
                "risk_level": "low",
                "confidence": 0.90
            }
        else:
            return {
                "summary": "Code analysis completed with mixed findings",
                "recommendations": [
                    "Review code structure for clarity",
                    "Consider refactoring opportunities",
                    "Improve documentation coverage"
                ],
                "risk_level": "low",
                "confidence": 0.75
            }
    
    def _combine_insights(self, glean_results: Dict[str, Any], grok_response: Dict[str, Any]) -> Dict[str, Any]:
        """Combine Glean technical results with Grok insights"""
        return {
            "technical_details": {
                "symbols_found": len(glean_results.get("symbols", [])),
                "dependencies_count": len(glean_results.get("dependencies", [])),
                "files_analyzed": len(glean_results.get("files", []))
            },
            "ai_insights": grok_response.get("summary", ""),
            "recommendations": grok_response.get("recommendations", []),
            "risk_assessment": grok_response.get("risk_level", "unknown"),
            "confidence_score": grok_response.get("confidence", 0.0)
        }


class StrandsGrokCLI:
    """CLI for Strands-Grok enhanced code analysis"""
    
    def __init__(self):
        self.agent: Optional[StrandsGleanAgent] = None
        self.grok_analyzer = GrokEnhancedAnalyzer()
    
    async def initialize_agent(self, project_root: str) -> bool:
        """Initialize the Strands-Glean agent"""
        if not STRANDS_GLEAN_AVAILABLE:
            print("❌ StrandsGleanAgent not available")
            return False
        
        try:
            print("🚀 Initializing Strands-Glean agent...")
            self.agent = await create_strands_glean_agent(project_root=project_root)
            print("✅ Agent initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            return False
    
    async def enhanced_analysis(self, analysis_type: str, query: str) -> Dict[str, Any]:
        """Perform enhanced analysis with Grok AI"""
        if not self.agent:
            return {"status": "error", "error": "Agent not initialized"}
        
        print(f"🔍 Analyzing: {query} (type: {analysis_type})")
        
        # Get Glean analysis
        glean_results = await self.agent.analyze_code(analysis_type, query)
        print(f"📊 Glean analysis: {glean_results.get('status', 'unknown')}")
        
        # Enhance with Grok
        enhanced_results = await self.grok_analyzer.enhance_analysis(glean_results, query)
        print(f"🤖 Grok enhancement: {enhanced_results.get('status', 'unknown')}")
        
        return enhanced_results


@click.group()
@click.option('--project-root', '-p', default=str(project_root), help='Project root directory')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, project_root, verbose):
    """Strands-Grok CLI for AI-enhanced code analysis"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['project_root'] = project_root
    ctx.obj['cli'] = StrandsGrokCLI()


@cli.command()
@click.pass_context
async def init(ctx):
    """Initialize the Strands-Glean agent"""
    cli_obj = ctx.obj['cli']
    project_root = ctx.obj['project_root']
    
    success = await cli_obj.initialize_agent(project_root)
    if success:
        print("🎉 Ready for enhanced code analysis!")
    else:
        print("💥 Initialization failed")
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--type', '-t', default='symbol_search', 
              type=click.Choice(['symbol_search', 'dependency_analysis', 'architecture_analysis']),
              help='Analysis type')
@click.pass_context
async def analyze(ctx, query, type):
    """Perform AI-enhanced code analysis"""
    cli_obj = ctx.obj['cli']
    
    if not cli_obj.agent:
        await cli_obj.initialize_agent(ctx.obj['project_root'])
    
    results = await cli_obj.enhanced_analysis(type, query)
    
    print("\n" + "="*60)
    print("🧠 ENHANCED ANALYSIS RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))


@cli.command()
@click.argument('symbol')
@click.pass_context
async def dependencies(ctx, symbol):
    """Find dependencies for a symbol with AI insights"""
    cli_obj = ctx.obj['cli']
    
    if not cli_obj.agent:
        await cli_obj.initialize_agent(ctx.obj['project_root'])
    
    results = await cli_obj.enhanced_analysis('dependency_analysis', symbol)
    
    print(f"\n🔗 DEPENDENCIES FOR: {symbol}")
    print("="*50)
    
    if 'combined_insights' in results:
        insights = results['combined_insights']
        print(f"📝 AI Summary: {insights.get('ai_insights', 'N/A')}")
        print(f"⚠️  Risk Level: {insights.get('risk_assessment', 'unknown')}")
        print(f"📈 Confidence: {insights.get('confidence_score', 0.0):.2f}")
        
        recommendations = insights.get('recommendations', [])
        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")


@cli.command()
@click.argument('pattern')
@click.pass_context
async def search(ctx, pattern):
    """Search symbols with AI-enhanced results"""
    cli_obj = ctx.obj['cli']
    
    if not cli_obj.agent:
        await cli_obj.initialize_agent(ctx.obj['project_root'])
    
    results = await cli_obj.enhanced_analysis('symbol_search', pattern)
    
    print(f"\n🔍 SYMBOL SEARCH: {pattern}")
    print("="*50)
    
    # Show original Glean results
    glean_data = results.get('original_glean', {})
    symbols = glean_data.get('symbols', [])
    print(f"Found {len(symbols)} symbols")
    
    # Show AI insights
    if 'combined_insights' in results:
        insights = results['combined_insights']
        print(f"\n🤖 AI Analysis: {insights.get('ai_insights', 'N/A')}")


@cli.command()
@click.pass_context
async def status(ctx):
    """Show agent status and capabilities"""
    cli_obj = ctx.obj['cli']
    
    if cli_obj.agent:
        summary = await cli_obj.agent.get_context_summary()
        print("\n📊 AGENT STATUS")
        print("="*40)
        print(json.dumps(summary, indent=2))
    else:
        print("❌ Agent not initialized. Run 'init' first.")


def main():
    """Main entry point with async support"""
    # Convert click commands to async
    def async_wrapper(func):
        def wrapper(*args, **kwargs):
            return asyncio.run(func(*args, **kwargs))
        return wrapper
    
    # Wrap async commands
    for command in [init, analyze, dependencies, search, status]:
        command.callback = async_wrapper(command.callback)
    
    cli()


if __name__ == '__main__':
    main()