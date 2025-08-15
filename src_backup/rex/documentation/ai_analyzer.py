"""
AI-powered analysis module for Koyfin scraper
Uses Grok-4 or fallback models for intelligent content analysis
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """AI-powered content analyzer with multiple model support"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "grok-beta"):
        self.api_key = api_key or os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY')
        self.model_name = model_name
        self.client = None
        
        # Try to initialize Grok client if available
        if self.api_key:
            try:
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent.parent))
                from a2a.grok4_client import get_grok4_client
                self.client = get_grok4_client()
                logger.info("Grok-4 client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Grok-4 client: {e}")
                self.client = None
    
    async def analyze_content(self, content: str, url: str, 
                             analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze content using AI"""
        if not self.client:
            # Fallback analysis without AI
            return self._fallback_analysis(content, url, analysis_type)
        
        try:
            prompts = self._get_analysis_prompts(analysis_type)
            
            messages = [
                {"role": "system", "content": prompts['system']},
                {"role": "user", "content": prompts['user'].format(
                    url=url, 
                    content=content[:8000]  # Limit content
                )}
            ]
            
            result = await self.client.complete(
                messages=messages,
                temperature=0.3,
                max_tokens=4096
            )
            
            if result.get("success"):
                return {
                    "status": "success",
                    "url": url,
                    "analysis_type": analysis_type,
                    "analysis": result["content"],
                    "model": result.get("model", self.model_name),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"AI analysis failed: {result.get('error')}")
                return self._fallback_analysis(content, url, analysis_type)
                
        except Exception as e:
            logger.error(f"Exception during AI analysis: {e}")
            return self._fallback_analysis(content, url, analysis_type)
    
    def _get_analysis_prompts(self, analysis_type: str) -> Dict[str, str]:
        """Get prompts for different analysis types"""
        prompts = {
            "comprehensive": {
                "system": """You are an expert software analyst specializing in reverse engineering web applications and financial platforms. 
                Analyze the provided documentation content and extract:
                1. Core features and functionality
                2. Technical capabilities and requirements
                3. User interface components and patterns
                4. Data sources and integrations
                5. Competitive advantages and unique selling points
                6. Implementation recommendations for building a competitive platform
                
                Be thorough, specific, and actionable in your analysis.""",
                
                "user": """Analyze this documentation content from Koyfin ({url}):
                
                {content}
                
                Provide a comprehensive analysis including:
                - Feature identification and categorization
                - Technical architecture insights
                - User experience patterns
                - Data requirements
                - Competitive analysis
                - Implementation roadmap suggestions
                
                Format your response with clear sections and bullet points."""
            },
            
            "features": {
                "system": """You are a product manager analyzing competitor features. Extract and categorize all features mentioned in the documentation. Focus on functionality, not marketing language.""",
                
                "user": """Extract all features from this Koyfin documentation ({url}):
                
                {content}
                
                List features in these categories:
                - Data & Analytics
                - Visualization & Charting
                - Portfolio Management
                - Screening & Discovery
                - Alerts & Notifications
                - Integration & API
                - User Interface
                - Other Features"""
            },
            
            "technical": {
                "system": """You are a technical architect analyzing system requirements. Focus on technical capabilities, data flows, performance requirements, and implementation details.""",
                
                "user": """Analyze technical requirements from this Koyfin documentation ({url}):
                
                {content}
                
                Focus on:
                - Data architecture requirements
                - Performance and scalability needs
                - Integration points and APIs
                - Security considerations
                - Technology stack implications"""
            }
        }
        
        return prompts.get(analysis_type, prompts["comprehensive"])
    
    def _fallback_analysis(self, content: str, url: str, analysis_type: str) -> Dict[str, Any]:
        """Fallback analysis when AI is not available"""
        logger.info("Using fallback analysis (no AI)")
        
        # Basic keyword extraction
        keywords = {
            'data_features': ['real-time', 'historical', 'data', 'feed', 'api', 'streaming'],
            'viz_features': ['chart', 'graph', 'visualization', 'dashboard', 'widget'],
            'analysis_features': ['screen', 'filter', 'analysis', 'metric', 'indicator'],
            'portfolio_features': ['portfolio', 'watchlist', 'holdings', 'performance'],
            'alert_features': ['alert', 'notification', 'trigger', 'condition']
        }
        
        found_features = {}
        for category, terms in keywords.items():
            found = []
            for term in terms:
                if term.lower() in content.lower():
                    # Find context
                    pos = content.lower().find(term.lower())
                    if pos >= 0:
                        start = max(0, pos - 50)
                        end = min(len(content), pos + len(term) + 50)
                        context = content[start:end].strip()
                        found.append({
                            'term': term,
                            'context': context
                        })
            if found:
                found_features[category] = found
        
        analysis = f"""# Fallback Analysis for {url}

## Summary
Analyzed {len(content)} characters of content. Found features in {len(found_features)} categories.

## Identified Features
"""
        
        for category, features in found_features.items():
            analysis += f"\n### {category.replace('_', ' ').title()}\n"
            for feature in features[:5]:  # Limit to 5 per category
                analysis += f"- **{feature['term']}**: {feature['context']}\n"
        
        analysis += """
## Recommendations
Based on keyword analysis, this page contains information about:
"""
        
        if 'data_features' in found_features:
            analysis += "- Data infrastructure and real-time capabilities\n"
        if 'viz_features' in found_features:
            analysis += "- Visualization and charting components\n"
        if 'analysis_features' in found_features:
            analysis += "- Analysis and screening tools\n"
        if 'portfolio_features' in found_features:
            analysis += "- Portfolio management functionality\n"
        if 'alert_features' in found_features:
            analysis += "- Alert and notification systems\n"
        
        return {
            "status": "success",
            "url": url,
            "analysis_type": f"{analysis_type}_fallback",
            "analysis": analysis,
            "model": "fallback_analyzer",
            "features_found": len(found_features),
            "timestamp": datetime.now().isoformat()
        }
    
    async def batch_analyze(self, pages: List[Dict[str, Any]], 
                           analysis_type: str = "comprehensive",
                           max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Analyze multiple pages concurrently"""
        logger.info(f"Batch analyzing {len(pages)} pages")
        
        results = []
        
        # Create tasks in batches to avoid overwhelming the API
        for i in range(0, len(pages), max_concurrent):
            batch = pages[i:i + max_concurrent]
            tasks = []
            
            for page in batch:
                if isinstance(page, dict):
                    content = page.get('content', '')
                    url = page.get('url', '')
                else:
                    # Handle PageContent objects
                    content = getattr(page, 'content', '')
                    url = getattr(page, 'url', '')
                
                if content and url:
                    task = self.analyze_content(content, url, analysis_type)
                    tasks.append(task)
            
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Analysis task failed: {result}")
                        results.append({
                            "status": "error",
                            "error": str(result),
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        results.append(result)
        
        return results
    
    def generate_insights_report(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights report from multiple analyses"""
        logger.info(f"Generating insights from {len(analyses)} analyses")
        
        insights = {
            "summary": {
                "total_analyses": len(analyses),
                "successful_analyses": len([a for a in analyses if a.get("status") == "success"]),
                "ai_analyses": len([a for a in analyses if "fallback" not in a.get("model", "")]),
                "timestamp": datetime.now().isoformat()
            },
            "key_features": {},
            "technical_insights": [],
            "implementation_priorities": [],
            "data_requirements": [],
            "competitive_insights": []
        }
        
        # Extract insights from each analysis
        for analysis in analyses:
            if analysis.get("status") != "success":
                continue
            
            content = analysis.get("analysis", "")
            
            # Extract features (basic keyword matching)
            feature_keywords = [
                'portfolio', 'chart', 'screening', 'alert', 'api', 
                'real-time', 'historical', 'fundamental', 'technical'
            ]
            
            for keyword in feature_keywords:
                if keyword in content.lower():
                    if keyword not in insights["key_features"]:
                        insights["key_features"][keyword] = 0
                    insights["key_features"][keyword] += 1
        
        # Sort features by frequency
        insights["key_features"] = dict(
            sorted(insights["key_features"].items(), 
                   key=lambda x: x[1], reverse=True)
        )
        
        # Generate recommendations based on findings
        if insights["key_features"]:
            top_features = list(insights["key_features"].keys())[:5]
            insights["implementation_priorities"] = [
                f"Implement {feature} functionality" for feature in top_features
            ]
        
        return insights


def create_ai_analyzer(config: Optional[Dict[str, Any]] = None) -> AIAnalyzer:
    """Factory function to create AI analyzer"""
    if config:
        return AIAnalyzer(
            api_key=config.get('api_key'),
            model_name=config.get('model_name', 'grok-beta')
        )
    return AIAnalyzer()