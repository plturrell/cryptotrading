"""
Grok API Client for AI-Enhanced Code Analysis
Integrates with xAI's Grok API for intelligent code insights
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiohttp

logger = logging.getLogger(__name__)


class GrokAPIError(Exception):
    """Grok API specific error"""

    pass


class GrokClient:
    """Client for xAI's Grok API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Grok API key required. Set GROK_API_KEY environment variable or pass api_key parameter."
            )

        self.base_url = "https://api.x.ai/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 1000
        self.rate_limit_reset = time.time() + 3600

    async def __aenter__(self):
        """Async context manager entry"""
        # Create SSL context that's more permissive for testing
        import ssl

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)

        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=60),
            connector=connector,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated request to Grok API"""
        if not self.session:
            raise GrokAPIError("Client not initialized. Use async context manager.")

        # Check rate limits
        if self.rate_limit_remaining < 10 and time.time() < self.rate_limit_reset:
            wait_time = self.rate_limit_reset - time.time()
            logger.warning(f"Rate limit low, waiting {wait_time:.1f}s")
            await asyncio.sleep(min(wait_time, 60))

        url = f"{self.base_url}/{endpoint}"

        try:
            async with self.session.post(url, json=payload) as response:
                # Update rate limit info from headers
                self.rate_limit_remaining = int(
                    response.headers.get("x-ratelimit-remaining", self.rate_limit_remaining)
                )
                self.rate_limit_reset = int(
                    response.headers.get("x-ratelimit-reset", self.rate_limit_reset)
                )

                if response.status == 429:
                    retry_after = int(response.headers.get("retry-after", 60))
                    raise GrokAPIError(f"Rate limited. Retry after {retry_after}s")

                if response.status != 200:
                    error_text = await response.text()
                    raise GrokAPIError(f"API error {response.status}: {error_text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise GrokAPIError(f"Network error: {str(e)}")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "grok-4-latest",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Create chat completion using Grok"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        return await self._make_request("chat/completions", payload)

    async def analyze_code_structure(
        self, glean_data: Dict[str, Any], focus: str = "architecture"
    ) -> Dict[str, Any]:
        """Analyze code structure using Grok"""

        # Create focused prompt based on Glean data
        symbols_summary = self._summarize_symbols(glean_data.get("symbols", []))
        dependencies_summary = self._summarize_dependencies(glean_data.get("dependencies", []))

        system_prompt = f"""You are an expert software architect and code analyst. Analyze the provided codebase structure and provide insights focused on {focus}.

Key areas to analyze:
- Architecture patterns and design quality
- Code organization and modularity  
- Dependency relationships and coupling
- Potential issues and improvement opportunities
- Refactoring recommendations

Provide specific, actionable insights based on the structural analysis."""

        user_prompt = f"""Analyze this codebase structure:

SYMBOLS FOUND:
{symbols_summary}

DEPENDENCIES:
{dependencies_summary}

ANALYSIS FOCUS: {focus}

Please provide:
1. Architecture assessment (1-2 sentences)
2. Key strengths (2-3 bullet points)
3. Areas for improvement (2-3 bullet points)  
4. Specific recommendations (2-3 actionable items)
5. Risk assessment (low/medium/high with brief explanation)

Keep the analysis concise but insightful."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.chat_completion(messages, temperature=0.3, max_tokens=1024)

            # Extract content from response
            if "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]

                return {
                    "status": "success",
                    "analysis": content,
                    "focus": focus,
                    "model": response.get("model", "grok-4-latest"),
                    "usage": response.get("usage", {}),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                raise GrokAPIError("Invalid response format")

        except Exception as e:
            logger.error(f"Grok analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "focus": focus,
                "timestamp": datetime.now().isoformat(),
            }

    async def generate_code_review(
        self, files_data: List[Dict[str, Any]], review_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Generate AI code review using Grok"""

        files_summary = self._summarize_files(files_data)

        system_prompt = f"""You are an experienced code reviewer conducting a {review_type} code review. 
Provide constructive feedback focusing on:
- Code quality and best practices
- Security considerations  
- Performance implications
- Maintainability and readability
- Testing and error handling

Be specific and actionable in your recommendations."""

        user_prompt = f"""Please review these files for a {review_type} code review:

{files_summary}

Provide:
1. Overall assessment score (1-10)
2. Critical issues (if any)
3. Improvement suggestions (top 3)
4. Positive aspects (what's done well)
5. Security considerations
6. Next steps for the development team

Format as structured feedback that can guide development decisions."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.chat_completion(messages, temperature=0.2, max_tokens=1536)

            if "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]

                return {
                    "status": "success",
                    "review": content,
                    "review_type": review_type,
                    "files_count": len(files_data),
                    "model": response.get("model", "grok-4-latest"),
                    "usage": response.get("usage", {}),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                raise GrokAPIError("Invalid response format")

        except Exception as e:
            logger.error(f"Grok code review failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "review_type": review_type,
                "timestamp": datetime.now().isoformat(),
            }

    async def explain_code_component(
        self, symbol_data: Dict[str, Any], context_level: str = "detailed"
    ) -> Dict[str, Any]:
        """Explain a specific code component using Grok"""

        symbol_name = symbol_data.get("name", "Unknown")
        symbol_type = symbol_data.get("kind", "symbol")
        file_path = symbol_data.get("file", "unknown")

        context_instructions = {
            "brief": "Provide a concise 1-2 sentence explanation",
            "detailed": "Provide a comprehensive explanation with context and usage",
            "tutorial": "Explain as if teaching someone new to the codebase",
        }

        system_prompt = f"""You are a senior developer explaining code to a team member. 
{context_instructions.get(context_level, context_instructions['detailed'])}

Focus on:
- What the component does and why it exists
- How it fits into the larger architecture  
- Key implementation details
- Usage patterns and examples
- Relationships with other components"""

        user_prompt = f"""Explain this code component:

COMPONENT: {symbol_name}
TYPE: {symbol_type}  
FILE: {file_path}
CONTEXT_LEVEL: {context_level}

Additional context:
{json.dumps(symbol_data, indent=2)}

Please explain this component clearly and helpfully."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            max_tokens = {"brief": 256, "detailed": 1024, "tutorial": 1536}.get(context_level, 1024)
            response = await self.chat_completion(messages, temperature=0.4, max_tokens=max_tokens)

            if "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]

                return {
                    "status": "success",
                    "explanation": content,
                    "symbol": symbol_name,
                    "context_level": context_level,
                    "model": response.get("model", "grok-4-latest"),
                    "usage": response.get("usage", {}),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                raise GrokAPIError("Invalid response format")

        except Exception as e:
            logger.error(f"Grok explanation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol_name,
                "timestamp": datetime.now().isoformat(),
            }

    async def generate_refactoring_suggestions(
        self, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate refactoring suggestions using Grok"""

        complexity_issues = analysis_data.get("complexity_issues", [])
        coupling_issues = analysis_data.get("coupling_issues", [])
        code_smells = analysis_data.get("code_smells", [])

        system_prompt = """You are a refactoring expert. Analyze the code issues and provide specific, prioritized refactoring suggestions.

Focus on:
- High-impact, low-risk refactoring opportunities
- Step-by-step refactoring plans
- Estimated effort and benefits
- Potential risks and mitigation strategies
- Order of operations for safe refactoring"""

        user_prompt = f"""Based on this code analysis, suggest refactoring improvements:

COMPLEXITY ISSUES:
{json.dumps(complexity_issues, indent=2)}

COUPLING ISSUES:  
{json.dumps(coupling_issues, indent=2)}

CODE SMELLS:
{json.dumps(code_smells, indent=2)}

Provide:
1. Top 3 refactoring priorities (with rationale)
2. Step-by-step plan for each priority
3. Estimated effort (hours/days)
4. Expected benefits
5. Risks and mitigation strategies
6. Testing recommendations

Make suggestions specific and actionable."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.chat_completion(messages, temperature=0.3, max_tokens=2048)

            if "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]

                return {
                    "status": "success",
                    "suggestions": content,
                    "issues_analyzed": len(complexity_issues)
                    + len(coupling_issues)
                    + len(code_smells),
                    "model": response.get("model", "grok-4-latest"),
                    "usage": response.get("usage", {}),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                raise GrokAPIError("Invalid response format")

        except Exception as e:
            logger.error(f"Grok refactoring suggestions failed: {e}")
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

    def _summarize_symbols(self, symbols: List[Dict[str, Any]]) -> str:
        """Create a concise summary of symbols for Grok analysis"""
        if not symbols:
            return "No symbols found"

        by_type = {}
        for symbol in symbols:
            symbol_type = symbol.get("kind", "unknown")
            if symbol_type not in by_type:
                by_type[symbol_type] = []
            by_type[symbol_type].append(symbol.get("name", "unnamed"))

        summary_lines = []
        for symbol_type, names in by_type.items():
            count = len(names)
            examples = names[:3]  # Show first 3 examples
            examples_str = ", ".join(examples)
            if count > 3:
                examples_str += f" (and {count-3} more)"
            summary_lines.append(f"- {symbol_type.title()}s ({count}): {examples_str}")

        return "\n".join(summary_lines)

    def _summarize_dependencies(self, dependencies: List[Dict[str, Any]]) -> str:
        """Create a concise summary of dependencies for Grok analysis"""
        if not dependencies:
            return "No dependencies analyzed"

        dep_summary = []
        for dep in dependencies[:10]:  # Limit to first 10
            source = dep.get("source", "unknown")
            target = dep.get("target", "unknown")
            dep_type = dep.get("type", "import")
            dep_summary.append(f"- {source} → {target} ({dep_type})")

        if len(dependencies) > 10:
            dep_summary.append(f"... and {len(dependencies) - 10} more dependencies")

        return "\n".join(dep_summary)

    def _summarize_files(self, files_data: List[Dict[str, Any]]) -> str:
        """Create a summary of files for code review"""
        if not files_data:
            return "No files provided"

        file_summary = []
        for file_data in files_data:
            file_path = file_data.get("path", "unknown")
            symbols = file_data.get("symbols", [])
            complexity = file_data.get("complexity", "unknown")

            file_summary.append(
                f"""
FILE: {file_path}
- Symbols: {len(symbols)} ({', '.join([s.get('name', 'unknown') for s in symbols[:3]])})
- Complexity: {complexity}
"""
            )

        return "\n".join(file_summary)

    async def get_model_info(self) -> Dict[str, Any]:
        """Get available models information"""
        try:
            return await self._make_request("models", {})
        except Exception as e:
            return {"error": str(e)}


# Factory function for easy usage
async def create_grok_client(api_key: Optional[str] = None) -> GrokClient:
    """Create and return a Grok client"""
    return GrokClient(api_key)


# Test function
async def test_grok_integration():
    """Test the Grok integration"""
    try:
        async with GrokClient() as grok:
            # Test simple chat
            test_messages = [
                {
                    "role": "user",
                    "content": "Explain what makes good software architecture in one sentence.",
                }
            ]

            response = await grok.chat_completion(test_messages, max_tokens=100)
            print("✅ Grok API connection successful")
            print(
                f"Response: {response.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}"
            )

            return True
    except Exception as e:
        print(f"❌ Grok test failed: {e}")
        return False


if __name__ == "__main__":
    # Set the API key for testing
    import os

    os.environ[
        "GROK_API_KEY"
    ] = "YOUR_XAI_API_KEY_HEREm6U8QovWNoUphU8Ax8dUMAbh2I3nlgCRNAwYc8yUMnMUtCbYPo44bJBxX8BoKw3EdkAXOp7TJJFQIT7b"

    # Test the integration
    asyncio.run(test_grok_integration())
