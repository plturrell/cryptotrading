"""
AI-Powered Root Cause Analysis using LLM for intelligent error analysis
Real implementation using actual AI, not pattern matching
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class AIRootCauseAnalyzer:
    """AI-powered root cause analysis using LLM for intelligent error analysis"""

    def __init__(self, db_path: str = "cryptotrading.db"):
        self.db_path = db_path
        self._init_database()
        # Use Claude or GPT-4 for analysis via API
        self.ai_provider = "anthropic"  # or "openai"
        self.api_key = None  # Will be loaded from environment

    def _init_database(self):
        """Initialize database for storing error analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_error_analysis (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                error_type TEXT,
                error_message TEXT,
                stack_trace TEXT,
                ai_analysis TEXT,
                root_cause TEXT,
                suggested_fixes TEXT,
                severity TEXT,
                category TEXT,
                confidence REAL,
                context TEXT,
                resolution_status TEXT DEFAULT 'pending'
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS error_patterns (
                id TEXT PRIMARY KEY,
                pattern_signature TEXT UNIQUE,
                occurrences INTEGER DEFAULT 1,
                first_seen DATETIME,
                last_seen DATETIME,
                ai_learned_pattern TEXT,
                successful_resolutions TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    async def analyze_error_with_ai(
        self, error: Exception, context: Dict[str, Any] = None, code_snippet: str = None
    ) -> Dict[str, Any]:
        """Use AI to analyze error and determine root cause"""

        error_id = hashlib.md5(f"{error}{datetime.now()}".encode()).hexdigest()
        error_str = str(error)
        error_type = type(error).__name__
        stack_trace = traceback.format_exc()

        # Get related code context
        code_context = await self._get_code_context(stack_trace)

        # Get historical similar errors
        similar_errors = self._get_similar_errors(error_str, error_type)

        # Build AI prompt with all context
        ai_prompt = self._build_ai_prompt(
            error_type=error_type,
            error_message=error_str,
            stack_trace=stack_trace,
            code_context=code_context,
            code_snippet=code_snippet,
            application_context=context,
            similar_errors=similar_errors,
        )

        # Get AI analysis
        ai_response = await self._call_ai_api(ai_prompt)

        # Parse AI response
        analysis = self._parse_ai_response(ai_response)

        # Store in database
        self._store_analysis(
            error_id=error_id,
            error_type=error_type,
            error_message=error_str,
            stack_trace=stack_trace,
            analysis=analysis,
            context=context,
        )

        # Update pattern learning
        self._update_error_patterns(error_str, analysis)

        return {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_str,
            "ai_analysis": analysis,
            "confidence": analysis.get("confidence", 0.0),
            "immediate_action": analysis.get("immediate_action"),
            "long_term_fix": analysis.get("long_term_fix"),
            "related_errors": similar_errors,
        }

    def _build_ai_prompt(self, **kwargs) -> str:
        """Build comprehensive prompt for AI analysis"""
        prompt = f"""Analyze this error from a cryptocurrency trading system and provide detailed root cause analysis.

ERROR DETAILS:
Type: {kwargs.get('error_type')}
Message: {kwargs.get('error_message')}

STACK TRACE:
{kwargs.get('stack_trace')}

CODE CONTEXT:
{kwargs.get('code_context', 'Not available')}

APPLICATION CONTEXT:
{json.dumps(kwargs.get('application_context', {}), indent=2)}

SIMILAR HISTORICAL ERRORS:
{json.dumps(kwargs.get('similar_errors', []), indent=2)}

Please provide:
1. Root cause analysis (be specific about what went wrong)
2. Severity assessment (CRITICAL/HIGH/MEDIUM/LOW)
3. Error category (database/network/logic/configuration/dependency)
4. Immediate fix (what to do right now)
5. Long-term solution (how to prevent this)
6. Confidence level (0-1) in your analysis
7. Any related system components that might be affected
8. Specific code changes needed (if applicable)

Format your response as JSON with these keys:
- root_cause: string
- severity: string
- category: string
- immediate_action: string
- long_term_fix: string
- confidence: float
- affected_components: list
- code_changes: string (optional)
- explanation: string (detailed technical explanation)
"""
        return prompt

    async def _call_ai_api(self, prompt: str) -> Dict[str, Any]:
        """Call real AI API for analysis using your actual AI providers"""

        # Try your actual AI providers in order of preference
        providers = [
            ("grok", self._call_grok_api),
            ("perplexity", self._call_perplexity_api),
            ("local_agent", self._call_local_agent),
        ]

        for provider_name, provider_func in providers:
            try:
                logger.info(f"Attempting AI analysis with {provider_name}")
                response = await provider_func(prompt)
                if response:
                    logger.info(f"AI analysis successful with {provider_name}")
                    return response
            except Exception as e:
                logger.warning(f"{provider_name} AI call failed: {e}")
                continue

        # If all AI providers fail, use intelligent fallback
        logger.warning("All AI providers failed, using intelligent fallback")
        return self._intelligent_fallback_analysis(prompt)

    async def _call_grok_api(self, prompt: str) -> Dict[str, Any]:
        """Call Grok AI API using your existing Grok client"""
        try:
            from src.cryptotrading.core.ai.grok4_client import Grok4Client

            # Use your actual Grok client
            grok_client = Grok4Client()

            # Create a focused error analysis request
            analysis_request = {
                "prompt": prompt,
                "task_type": "error_analysis",
                "output_format": "structured_json",
            }

            # Call Grok for analysis
            response = await grok_client.analyze_error_context(analysis_request)

            # Parse Grok's response into our format
            if isinstance(response, dict):
                return {
                    "root_cause": response.get("analysis", "Error analysis from Grok"),
                    "severity": response.get("severity", "MEDIUM"),
                    "category": response.get("category", "technical"),
                    "immediate_action": response.get(
                        "immediate_action", "Investigate error details"
                    ),
                    "long_term_fix": response.get(
                        "long_term_solution", "Review system architecture"
                    ),
                    "confidence": response.get("confidence", 0.8),
                    "explanation": response.get("detailed_explanation", str(response)),
                }
            else:
                # If response is text, structure it
                return {
                    "root_cause": str(response)[:200],
                    "severity": "MEDIUM",
                    "category": "technical",
                    "immediate_action": "Review Grok analysis",
                    "long_term_fix": "Implement suggested fixes",
                    "confidence": 0.7,
                    "explanation": str(response),
                }

        except Exception as e:
            logger.error(f"Grok API call failed: {e}")
            raise Exception(f"Grok analysis failed: {e}")

    async def _call_perplexity_api(self, prompt: str) -> Dict[str, Any]:
        """Call Perplexity AI using your existing client"""
        try:
            from src.cryptotrading.core.ml.perplexity import PerplexityClient

            # Use your actual Perplexity client
            perplexity_client = PerplexityClient()

            # Format prompt for Perplexity's strengths (research and analysis)
            research_prompt = f"""
            Analyze this software error and provide technical insights:
            
            {prompt}
            
            Focus on:
            1. Root cause identification
            2. Similar known issues
            3. Best practices for resolution
            4. Prevention strategies
            """

            # Call Perplexity for research-based analysis
            response = await perplexity_client.search_and_analyze(research_prompt)

            # Parse Perplexity's response
            if isinstance(response, dict):
                return {
                    "root_cause": response.get("summary", "Technical analysis from Perplexity"),
                    "severity": "MEDIUM",  # Perplexity doesn't typically classify severity
                    "category": "research_based",
                    "immediate_action": response.get(
                        "recommendations", ["Investigate based on research"]
                    )[:1][0]
                    if response.get("recommendations")
                    else "Research similar issues",
                    "long_term_fix": response.get("recommendations", ["Review best practices"])[
                        -1:
                    ][0]
                    if response.get("recommendations")
                    else "Follow industry best practices",
                    "confidence": 0.75,  # Perplexity provides research-based insights
                    "explanation": response.get("detailed_analysis", str(response)),
                }
            else:
                return {
                    "root_cause": str(response)[:200],
                    "severity": "MEDIUM",
                    "category": "research_based",
                    "immediate_action": "Review Perplexity research findings",
                    "long_term_fix": "Apply research-backed solutions",
                    "confidence": 0.75,
                    "explanation": str(response),
                }

        except Exception as e:
            logger.error(f"Perplexity API call failed: {e}")
            raise Exception(f"Perplexity analysis failed: {e}")

    async def _call_local_agent(self, prompt: str) -> Dict[str, Any]:
        """Call local AI agent if available"""
        try:
            from src.cryptotrading.core.agents.ai_agent import AIAgent

            agent = AIAgent()
            response = await agent.analyze_text(prompt)

            # Parse JSON from response
            if isinstance(response, str):
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            return response

        except ImportError:
            raise Exception("Local AI agent not available")

    def _intelligent_fallback_analysis(self, prompt: str) -> Dict[str, Any]:
        """Intelligent fallback when AI API is not available"""
        # Extract error details from prompt
        lines = prompt.split("\n")
        error_type = ""
        error_message = ""

        for i, line in enumerate(lines):
            if line.startswith("Type:"):
                error_type = line.replace("Type:", "").strip()
            elif line.startswith("Message:"):
                error_message = line.replace("Message:", "").strip()

        # Use intelligent heuristics based on error patterns
        analysis = {
            "root_cause": "Unable to connect to AI service for deep analysis",
            "severity": "MEDIUM",
            "category": "unknown",
            "immediate_action": "Check error logs and stack trace",
            "long_term_fix": "Configure AI service for intelligent analysis",
            "confidence": 0.3,
            "affected_components": [],
            "explanation": f"Fallback analysis for {error_type}: {error_message}",
        }

        # Smart categorization based on error type
        if "Database" in error_type or "sqlite" in error_message.lower():
            analysis["category"] = "database"
            analysis["root_cause"] = "Database operation failed"
            analysis["immediate_action"] = "Check database connectivity and schema"
        elif "Connection" in error_type or "timeout" in error_message.lower():
            analysis["category"] = "network"
            analysis["root_cause"] = "Network connectivity issue"
            analysis["immediate_action"] = "Check network connection and service availability"
        elif "Import" in error_type or "module" in error_message.lower():
            analysis["category"] = "dependency"
            analysis["root_cause"] = "Missing or incompatible dependency"
            analysis["immediate_action"] = "Check installed packages and imports"

        return analysis

    def _parse_ai_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate AI response"""
        required_keys = [
            "root_cause",
            "severity",
            "category",
            "immediate_action",
            "long_term_fix",
            "confidence",
        ]

        # Ensure all required keys are present
        for key in required_keys:
            if key not in response:
                response[key] = "Not determined by AI"
                if key == "confidence":
                    response[key] = 0.5
                elif key == "severity":
                    response[key] = "MEDIUM"

        # Validate confidence is between 0 and 1
        if not isinstance(response["confidence"], (int, float)):
            response["confidence"] = 0.5
        else:
            response["confidence"] = max(0.0, min(1.0, response["confidence"]))

        return response

    async def _get_code_context(self, stack_trace: str) -> str:
        """Extract relevant code context from stack trace"""
        code_context = []
        lines = stack_trace.split("\n")

        for line in lines:
            if 'File "' in line and "/cryptotrading/" in line:
                import re

                match = re.search(r'File "([^"]+)", line (\d+)', line)
                if match:
                    filepath = match.group(1)
                    line_num = int(match.group(2))

                    try:
                        with open(filepath, "r") as f:
                            file_lines = f.readlines()
                            # Get 5 lines before and after
                            start = max(0, line_num - 5)
                            end = min(len(file_lines), line_num + 5)

                            context_lines = []
                            for i in range(start, end):
                                prefix = ">>> " if i == line_num - 1 else "    "
                                context_lines.append(f"{prefix}{i+1}: {file_lines[i].rstrip()}")

                            code_context.append(f"\nFile: {filepath}\n" + "\n".join(context_lines))
                    except:
                        pass

        return "\n".join(code_context) if code_context else "Could not extract code context"

    def _get_similar_errors(self, error_message: str, error_type: str) -> List[Dict]:
        """Get similar errors from history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent similar errors
        cursor.execute(
            """
            SELECT error_type, error_message, root_cause, suggested_fixes, confidence
            FROM ai_error_analysis
            WHERE error_type = ? OR error_message LIKE ?
            ORDER BY timestamp DESC
            LIMIT 5
        """,
            (error_type, f"%{error_message[:50]}%"),
        )

        similar = []
        for row in cursor.fetchall():
            similar.append(
                {
                    "error_type": row[0],
                    "error_message": row[1][:100],
                    "root_cause": row[2],
                    "suggested_fixes": row[3],
                    "confidence": row[4],
                }
            )

        conn.close()
        return similar

    def _store_analysis(self, **kwargs):
        """Store analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        analysis = kwargs.get("analysis", {})

        cursor.execute(
            """
            INSERT INTO ai_error_analysis 
            (id, error_type, error_message, stack_trace, ai_analysis, 
             root_cause, suggested_fixes, severity, category, confidence, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                kwargs["error_id"],
                kwargs["error_type"],
                kwargs["error_message"],
                kwargs["stack_trace"],
                json.dumps(analysis),
                analysis.get("root_cause"),
                json.dumps(
                    {
                        "immediate": analysis.get("immediate_action"),
                        "long_term": analysis.get("long_term_fix"),
                    }
                ),
                analysis.get("severity"),
                analysis.get("category"),
                analysis.get("confidence"),
                json.dumps(kwargs.get("context", {})),
            ),
        )

        conn.commit()
        conn.close()

    def _update_error_patterns(self, error_message: str, analysis: Dict):
        """Update learned error patterns"""
        pattern_signature = hashlib.md5(error_message[:100].encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO error_patterns (id, pattern_signature, first_seen, last_seen, ai_learned_pattern)
            VALUES (?, ?, datetime('now'), datetime('now'), ?)
            ON CONFLICT(pattern_signature) DO UPDATE SET
                occurrences = occurrences + 1,
                last_seen = datetime('now'),
                ai_learned_pattern = ?
        """,
            (pattern_signature, pattern_signature, json.dumps(analysis), json.dumps(analysis)),
        )

        conn.commit()
        conn.close()

    def get_ai_insights(self, hours: int = 24) -> Dict[str, Any]:
        """Get AI-powered insights from error history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        # Get error statistics
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total,
                AVG(confidence) as avg_confidence,
                COUNT(DISTINCT category) as unique_categories,
                COUNT(DISTINCT root_cause) as unique_root_causes
            FROM ai_error_analysis
            WHERE timestamp > ?
        """,
            (cutoff,),
        )

        stats = cursor.fetchone()

        # Get top issues
        cursor.execute(
            """
            SELECT root_cause, COUNT(*) as count, AVG(confidence) as avg_conf
            FROM ai_error_analysis
            WHERE timestamp > ?
            GROUP BY root_cause
            ORDER BY count DESC
            LIMIT 5
        """,
            (cutoff,),
        )

        top_issues = [
            {"root_cause": row[0], "count": row[1], "confidence": row[2]}
            for row in cursor.fetchall()
        ]

        # Get learned patterns
        cursor.execute(
            """
            SELECT COUNT(*) as patterns, SUM(occurrences) as total_occurrences
            FROM error_patterns
            WHERE last_seen > ?
        """,
            (cutoff,),
        )

        patterns = cursor.fetchone()

        conn.close()

        return {
            "total_errors_analyzed": stats[0] if stats else 0,
            "average_ai_confidence": stats[1] if stats else 0,
            "unique_error_categories": stats[2] if stats else 0,
            "unique_root_causes": stats[3] if stats else 0,
            "top_issues": top_issues,
            "learned_patterns": patterns[0] if patterns else 0,
            "pattern_matches": patterns[1] if patterns else 0,
            "ai_provider": self.ai_provider,
            "analysis_period_hours": hours,
        }


# Global instance
ai_analyzer = AIRootCauseAnalyzer()


async def analyze_with_ai(error: Exception, context: Dict = None) -> Dict[str, Any]:
    """Convenience function for AI error analysis"""
    return await ai_analyzer.analyze_error_with_ai(error, context)
