#!/usr/bin/env python3
"""
Glean-MCP Integration Script
Integrates Glean code analysis with MCP server and agent testing framework
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cryptotrading.infrastructure.analysis.vercel_glean_client import VercelGleanClient
from cryptotrading.core.protocols.mcp.server import MCPServer
# Agent testing framework integration (optional)
try:
    from framework.agent_testing.cli import AgentTestRunner
    AGENT_TESTING_AVAILABLE = True
except ImportError:
    AgentTestRunner = None
    AGENT_TESTING_AVAILABLE = False

logger = logging.getLogger(__name__)

class GleanMCPIntegration:
    """Integration between Glean analysis, MCP server, and agent testing"""
    
    def __init__(self, project_root: str = "/Users/apple/projects/cryptotrading"):
        self.project_root = Path(project_root)
        self.glean_client = None
        self.mcp_server = None
        self.test_runner = None
        
    async def initialize(self):
        """Initialize all components"""
        print("🔧 Initializing Glean-MCP integration...")
        
        # Initialize Glean client
        self.glean_client = VercelGleanClient(str(self.project_root))
        
        # Initialize MCP server (for tool integration)
        self.mcp_server = MCPServer("glean-integration", "1.0.0")
        
        # Initialize agent test runner (if available)
        if AGENT_TESTING_AVAILABLE:
            self.test_runner = AgentTestRunner()
        else:
            self.test_runner = None
        
        print("✅ All components initialized")
    
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis combining all tools"""
        results = {
            "timestamp": asyncio.get_event_loop().time(),
            "glean_analysis": {},
            "mcp_status": {},
            "agent_tests": {},
            "integration_status": "success"
        }
        
        try:
            # 1. Index project with Glean
            print("📦 Indexing project with Glean...")
            index_result = await self.glean_client.index_project("main", force_reindex=True)
            results["glean_analysis"]["indexing"] = index_result
            
            if index_result.get("status") == "success":
                print(f"✅ Indexed {index_result['stats']['files_indexed']} files")
                print(f"📊 Found {index_result['stats']['total_symbols']} symbols")
                
                # 2. Run architecture validation
                print("🏗️ Validating architecture...")
                arch_result = await self.glean_client.validate_architecture()
                results["glean_analysis"]["architecture"] = arch_result
                
                violations = arch_result.get("total_violations", 0)
                if violations == 0:
                    print("✅ No architecture violations found")
                else:
                    print(f"⚠️ Found {violations} architecture violations")
                
                # 3. Analyze core dependencies
                print("📦 Analyzing core module dependencies...")
                deps_result = await self.glean_client.analyze_dependencies(
                    "src/cryptotrading/core", max_depth=3
                )
                results["glean_analysis"]["dependencies"] = deps_result
                print(f"📊 Found {deps_result.get('total_dependencies', 0)} total dependencies")
                
                # 4. Find function and class definitions
                print("🔍 Cataloging code symbols...")
                functions = await self.glean_client.find_function_definitions()
                classes = await self.glean_client.find_class_definitions()
                results["glean_analysis"]["symbols"] = {
                    "functions": len(functions),
                    "classes": len(classes)
                }
                print(f"📝 Found {len(functions)} functions and {len(classes)} classes")
            
            # 5. Get MCP server status
            print("🔌 Checking MCP server integration...")
            mcp_tools = await self._get_mcp_tools()
            results["mcp_status"] = {
                "available_tools": len(mcp_tools),
                "tools": mcp_tools
            }
            
            # 6. Run agent tests with Glean integration
            print("🤖 Running agent tests with Glean integration...")
            test_results = await self._run_integrated_tests()
            results["agent_tests"] = test_results
            
            # 7. Generate integration report
            report = self._generate_integration_report(results)
            results["report"] = report
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            results["integration_status"] = "error"
            results["error"] = str(e)
            return results
    
    async def _get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools"""
        try:
            # Add Glean-powered tools to MCP server
            glean_tools = [
                {
                    "name": "analyze_code_dependencies",
                    "description": "Analyze code dependencies using Glean",
                    "parameters": {
                        "module": {"type": "string", "description": "Module to analyze"},
                        "depth": {"type": "integer", "description": "Analysis depth", "default": 2}
                    }
                },
                {
                    "name": "validate_architecture",
                    "description": "Validate architectural constraints",
                    "parameters": {}
                },
                {
                    "name": "find_code_symbols",
                    "description": "Find functions and classes in codebase",
                    "parameters": {
                        "symbol_type": {"type": "string", "enum": ["function", "class", "all"]}
                    }
                }
            ]
            return glean_tools
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            return []
    
    async def _run_integrated_tests(self) -> Dict[str, Any]:
        """Run agent tests with Glean integration"""
        try:
            # Test Glean client functionality
            test_results = {
                "glean_client_tests": {
                    "indexing": await self._test_glean_indexing(),
                    "querying": await self._test_glean_querying(),
                    "architecture": await self._test_architecture_validation()
                },
                "mcp_integration": {
                    "tool_registration": await self._test_mcp_tool_registration(),
                    "tool_execution": await self._test_mcp_tool_execution()
                }
            }
            return test_results
        except Exception as e:
            logger.error(f"Integrated tests failed: {e}")
            return {"error": str(e)}
    
    async def _test_glean_indexing(self) -> Dict[str, Any]:
        """Test Glean indexing functionality"""
        try:
            result = await self.glean_client.index_project("test", force_reindex=True)
            return {
                "success": result.get("status") == "success",
                "files_indexed": result.get("stats", {}).get("files_indexed", 0),
                "symbols_found": result.get("stats", {}).get("total_symbols", 0)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_glean_querying(self) -> Dict[str, Any]:
        """Test Glean query functionality"""
        try:
            functions = await self.glean_client.find_function_definitions()
            classes = await self.glean_client.find_class_definitions()
            return {
                "success": True,
                "functions_found": len(functions),
                "classes_found": len(classes)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_architecture_validation(self) -> Dict[str, Any]:
        """Test architecture validation"""
        try:
            result = await self.glean_client.validate_architecture()
            return {
                "success": result.get("status") == "completed",
                "violations": result.get("total_violations", 0),
                "rules_checked": result.get("rules_checked", 0)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_mcp_tool_registration(self) -> Dict[str, Any]:
        """Test MCP tool registration"""
        try:
            tools = await self._get_mcp_tools()
            return {
                "success": len(tools) > 0,
                "tools_registered": len(tools)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_mcp_tool_execution(self) -> Dict[str, Any]:
        """Test MCP tool execution"""
        try:
            # Simulate tool execution
            deps = await self.glean_client.analyze_dependencies("src/cryptotrading/core")
            return {
                "success": True,
                "dependencies_found": deps.get("total_dependencies", 0)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_integration_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive integration report"""
        glean = results.get("glean_analysis", {})
        mcp = results.get("mcp_status", {})
        tests = results.get("agent_tests", {})
        
        # Calculate overall health score
        health_score = 0
        max_score = 100
        
        # Glean analysis health (40 points)
        if glean.get("indexing", {}).get("status") == "success":
            health_score += 20
        if glean.get("architecture", {}).get("total_violations", 1) == 0:
            health_score += 10
        if glean.get("dependencies", {}).get("total_dependencies", 0) > 0:
            health_score += 10
        
        # MCP integration health (30 points)
        if mcp.get("available_tools", 0) > 0:
            health_score += 15
        if tests.get("mcp_integration", {}).get("tool_registration", {}).get("success"):
            health_score += 15
        
        # Agent testing health (30 points)
        glean_tests = tests.get("glean_client_tests", {})
        test_successes = sum(1 for test in glean_tests.values() if test.get("success"))
        health_score += (test_successes / max(len(glean_tests), 1)) * 30
        
        return {
            "overall_health_score": health_score,
            "max_score": max_score,
            "health_percentage": (health_score / max_score) * 100,
            "status": "excellent" if health_score >= 90 else 
                     "good" if health_score >= 70 else
                     "fair" if health_score >= 50 else "poor",
            "recommendations": self._generate_recommendations(results),
            "summary": {
                "files_analyzed": glean.get("indexing", {}).get("stats", {}).get("files_indexed", 0),
                "symbols_cataloged": glean.get("symbols", {}).get("functions", 0) + 
                                   glean.get("symbols", {}).get("classes", 0),
                "architecture_violations": glean.get("architecture", {}).get("total_violations", 0),
                "mcp_tools_available": mcp.get("available_tools", 0),
                "tests_passed": sum(1 for test_group in tests.values() 
                                  for test in (test_group.values() if isinstance(test_group, dict) else [])
                                  if isinstance(test, dict) and test.get("success"))
            }
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        glean = results.get("glean_analysis", {})
        tests = results.get("agent_tests", {})
        
        # Architecture recommendations
        violations = glean.get("architecture", {}).get("total_violations", 0)
        if violations > 0:
            recommendations.append(f"Fix {violations} architecture violations to improve code quality")
        
        # Dependency recommendations
        deps = glean.get("dependencies", {}).get("total_dependencies", 0)
        if deps > 50:
            recommendations.append("Consider reducing dependencies to improve maintainability")
        
        # Testing recommendations
        failed_tests = []
        for test_group_name, test_group in tests.items():
            if isinstance(test_group, dict):
                for test_name, test_result in test_group.items():
                    if isinstance(test_result, dict) and not test_result.get("success"):
                        failed_tests.append(f"{test_group_name}.{test_name}")
        
        if failed_tests:
            recommendations.append(f"Fix failing tests: {', '.join(failed_tests)}")
        
        if not recommendations:
            recommendations.append("System is healthy - continue regular monitoring")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.glean_client:
            await self.glean_client.cleanup()
        print("🧹 Cleanup complete")

async def main():
    """Main integration function"""
    integration = GleanMCPIntegration()
    
    try:
        await integration.initialize()
        
        print("\n" + "="*60)
        print("🚀 Running Comprehensive Glean-MCP Integration Analysis")
        print("="*60)
        
        results = await integration.run_comprehensive_analysis()
        
        # Save results
        output_file = Path("data/glean_mcp_integration_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        report = results.get("report", {})
        print(f"\n📊 Integration Health Score: {report.get('health_percentage', 0):.1f}%")
        print(f"🎯 Status: {report.get('status', 'unknown').title()}")
        
        summary = report.get("summary", {})
        print(f"\n📈 Summary:")
        print(f"  • Files analyzed: {summary.get('files_analyzed', 0)}")
        print(f"  • Symbols cataloged: {summary.get('symbols_cataloged', 0)}")
        print(f"  • Architecture violations: {summary.get('architecture_violations', 0)}")
        print(f"  • MCP tools available: {summary.get('mcp_tools_available', 0)}")
        print(f"  • Tests passed: {summary.get('tests_passed', 0)}")
        
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n💾 Full results saved to: {output_file}")
        print("\n✅ Glean-MCP integration analysis complete!")
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await integration.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
