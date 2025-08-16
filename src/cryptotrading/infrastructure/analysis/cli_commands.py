"""
CLI Commands for Glean Integration - Production implementation
Extends the existing CLI framework with Glean-powered code analysis commands
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import click
from datetime import datetime

from .glean_client import GleanClient
from .code_analyzer import CodeAnalyzer
from .impact_analyzer import ImpactAnalyzer
from .architecture_validator import ArchitectureValidator
from .realtime_analyzer import RealtimeCodeAnalyzer, AnalysisType

logger = logging.getLogger(__name__)

class GleanCLI:
    """CLI interface for Glean-powered code analysis"""
    
    def __init__(self, project_root: str = "/Users/apple/projects/cryptotrading"):
        self.project_root = project_root
        self.glean_client = None
        self.code_analyzer = None
        self.impact_analyzer = None
        self.architecture_validator = None
        self.realtime_analyzer = None
    
    async def _ensure_initialized(self):
        """Ensure all analyzers are initialized"""
        if not self.glean_client:
            self.glean_client = GleanClient(project_root=self.project_root)
            self.code_analyzer = CodeAnalyzer(project_root=self.project_root)
            self.impact_analyzer = ImpactAnalyzer(self.glean_client)
            self.architecture_validator = ArchitectureValidator(self.glean_client)
            self.realtime_analyzer = RealtimeCodeAnalyzer(self.project_root, self.glean_client)
            
            # Initialize code analyzer with glean client
            await self.code_analyzer.initialize()
            self.code_analyzer.glean_client = self.glean_client
    
    async def analyze_dependencies(self, module: str, depth: int = 2, output_format: str = "json") -> Dict[str, Any]:
        """Analyze dependencies for a specific module"""
        await self._ensure_initialized()
        
        try:
            click.echo(f"🔍 Analyzing dependencies for module: {module}")
            
            # Get dependency information
            deps = await self.code_analyzer.analyze_dependencies(module, depth=depth)
            
            if output_format == "json":
                return deps
            elif output_format == "tree":
                return self._format_dependency_tree(deps)
            elif output_format == "graph":
                return await self._generate_dependency_graph(deps)
            
        except Exception as e:
            logger.error("Dependency analysis failed: %s", e)
            return {"error": str(e), "success": False}
    
    def _format_dependency_tree(self, deps: Dict[str, Any]) -> str:
        """Format dependencies as a tree structure"""
        lines = []
        lines.append(f"📦 Dependencies for {deps.get('module', 'Unknown')}")
        lines.append("=" * 50)
        
        def format_deps(dep_list: List[str], prefix: str = ""):
            for i, dep in enumerate(dep_list):
                is_last = i == len(dep_list) - 1
                current_prefix = "└── " if is_last else "├── "
                lines.append(f"{prefix}{current_prefix}{dep}")
        
        if deps.get("direct"):
            lines.append("\n🔗 Direct Dependencies:")
            format_deps(deps["direct"])
        
        if deps.get("transitive"):
            lines.append("\n🔄 Transitive Dependencies:")
            format_deps(deps["transitive"][:10])  # Limit for readability
            if len(deps["transitive"]) > 10:
                lines.append(f"    ... and {len(deps['transitive']) - 10} more")
        
        return "\n".join(lines)
    
    async def _generate_dependency_graph(self, deps: Dict[str, Any]) -> str:
        """Generate a visual dependency graph"""
        # This would integrate with graphviz or similar
        # For now, return a simple representation
        return f"Dependency graph for {deps.get('module')} (visual graph generation not implemented)"
    
    async def analyze_impact(self, files: List[str], output_format: str = "json") -> Dict[str, Any]:
        """Analyze impact of changes to specified files"""
        await self._ensure_initialized()
        
        try:
            click.echo(f"💥 Analyzing impact of changes to {len(files)} files")
            
            changes = [{"file": f, "type": "modified"} for f in files]
            impact = await self.impact_analyzer.analyze_change_impact(changes)
            
            if output_format == "json":
                return impact
            elif output_format == "summary":
                return self._format_impact_summary(impact)
            elif output_format == "report":
                return self._generate_impact_report(impact)
            
        except Exception as e:
            logger.error("Impact analysis failed: %s", e)
            return {"error": str(e), "success": False}
    
    def _format_impact_summary(self, impact: Dict[str, Any]) -> str:
        """Format impact analysis as a summary"""
        lines = []
        lines.append("💥 Change Impact Summary")
        lines.append("=" * 30)
        
        if impact.get("affected_modules"):
            lines.append(f"\n📦 Affected Modules: {len(impact['affected_modules'])}")
            for module in impact["affected_modules"][:5]:
                lines.append(f"  • {module}")
            if len(impact["affected_modules"]) > 5:
                lines.append(f"  ... and {len(impact['affected_modules']) - 5} more")
        
        if impact.get("affected_tests"):
            lines.append(f"\n🧪 Affected Tests: {len(impact['affected_tests'])}")
            for test in impact["affected_tests"][:3]:
                lines.append(f"  • {test}")
        
        if impact.get("risk_score"):
            risk = impact["risk_score"]
            risk_level = "🔴 HIGH" if risk > 0.7 else "🟡 MEDIUM" if risk > 0.3 else "🟢 LOW"
            lines.append(f"\n⚠️ Risk Level: {risk_level} ({risk:.2f})")
        
        return "\n".join(lines)
    
    async def validate_architecture(self, output_format: str = "json") -> Dict[str, Any]:
        """Validate architectural constraints"""
        await self._ensure_initialized()
        
        try:
            click.echo("🏗️ Validating architecture constraints...")
            
            await self.architecture_validator.validate_architecture()
            report = self.architecture_validator.generate_violation_report()
            
            if output_format == "json":
                return report
            elif output_format == "summary":
                return self._format_architecture_summary(report)
            elif output_format == "detailed":
                return self._format_architecture_detailed(report)
            
        except Exception as e:
            logger.error("Architecture validation failed: %s", e)
            return {"error": str(e), "success": False}
    
    def _format_architecture_summary(self, report: Dict[str, Any]) -> str:
        """Format architecture validation as summary"""
        lines = []
        lines.append("🏗️ Architecture Validation Summary")
        lines.append("=" * 40)
        
        status = report.get("status", "unknown")
        message = report.get("message", "No message")
        
        status_icon = {
            "clean": "✅",
            "minor": "ℹ️",
            "warning": "⚠️", 
            "critical": "🚨"
        }.get(status, "❓")
        
        lines.append(f"\n{status_icon} Status: {message}")
        
        by_severity = report.get("by_severity", {})
        if any(by_severity.values()):
            lines.append("\n📊 Violations by Severity:")
            for severity, count in by_severity.items():
                if count > 0:
                    icon = {"critical": "🚨", "high": "⚠️", "medium": "🟡", "low": "ℹ️"}.get(severity, "•")
                    lines.append(f"  {icon} {severity.title()}: {count}")
        
        by_type = report.get("by_type", {})
        if any(by_type.values()):
            lines.append("\n🔍 Violations by Type:")
            for vtype, count in by_type.items():
                if count > 0:
                    lines.append(f"  • {vtype.replace('_', ' ').title()}: {count}")
        
        return "\n".join(lines)
    
    def _format_architecture_detailed(self, report: Dict[str, Any]) -> str:
        """Format detailed architecture validation report"""
        lines = []
        lines.append("🏗️ Detailed Architecture Validation Report")
        lines.append("=" * 50)
        
        # Add summary
        lines.append(self._format_architecture_summary(report))
        
        # Add detailed violations
        violations = report.get("violations", [])
        if violations:
            lines.append("\n🔍 Detailed Violations:")
            lines.append("-" * 30)
            
            for i, violation in enumerate(violations[:10], 1):  # Limit for readability
                severity_icon = {
                    "critical": "🚨",
                    "high": "⚠️", 
                    "medium": "🟡",
                    "low": "ℹ️"
                }.get(violation.get("severity"), "•")
                
                lines.append(f"\n{i}. {severity_icon} {violation.get('description')}")
                if violation.get("source"):
                    lines.append(f"   📍 Source: {violation['source']}")
                if violation.get("recommendation"):
                    lines.append(f"   💡 Fix: {violation['recommendation']}")
            
            if len(violations) > 10:
                lines.append(f"\n... and {len(violations) - 10} more violations")
        
        return "\n".join(lines)
    
    async def start_realtime_monitoring(self, analyses: Optional[List[str]] = None) -> Dict[str, Any]:
        """Start real-time code monitoring"""
        await self._ensure_initialized()
        
        try:
            # Configure analyses
            if analyses:
                enabled_analyses = set()
                for analysis in analyses:
                    try:
                        enabled_analyses.add(AnalysisType(analysis))
                    except ValueError:
                        click.echo(f"⚠️ Unknown analysis type: {analysis}")
                
                self.realtime_analyzer.configure_analyses(enabled_analyses)
            
            # Start monitoring
            success = await self.realtime_analyzer.start_watching()
            
            if success:
                click.echo("🔄 Real-time code monitoring started successfully!")
                click.echo("📁 Watching for changes in src/ directory")
                
                # Add a simple callback to show analysis results
                def result_callback(result):
                    timestamp = datetime.fromtimestamp(result.timestamp).strftime("%H:%M:%S")
                    status = "✅" if result.success else "❌"
                    click.echo(f"{timestamp} {status} {result.analysis_type.value}: {Path(result.file_path).name}")
                
                self.realtime_analyzer.add_analysis_callback(result_callback)
                
                return {
                    "success": True,
                    "message": "Real-time monitoring started",
                    "enabled_analyses": [a.value for a in self.realtime_analyzer.enabled_analyses]
                }
            else:
                return {"success": False, "error": "Failed to start monitoring"}
                
        except Exception as e:
            logger.error("Failed to start real-time monitoring: %s", e)
            return {"success": False, "error": str(e)}
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get real-time monitoring status"""
        if not self.realtime_analyzer:
            return {"monitoring": False, "message": "Monitoring not initialized"}
        
        stats = self.realtime_analyzer.get_analysis_stats()
        recent_results = self.realtime_analyzer.get_recent_results(limit=5)
        
        return {
            "monitoring": stats["is_watching"],
            "stats": stats,
            "recent_results": [
                {
                    "timestamp": datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                    "file": Path(r.file_path).name,
                    "analysis": r.analysis_type.value,
                    "success": r.success,
                    "duration_ms": r.duration_ms
                }
                for r in recent_results
            ]
        }
    
    async def run_full_analysis(self, output_format: str = "json") -> Dict[str, Any]:
        """Run comprehensive analysis on entire codebase"""
        await self._ensure_initialized()
        
        try:
            click.echo("🔍 Running comprehensive codebase analysis...")
            click.echo("This may take several minutes...")
            
            # Configure all analyses
            all_analyses = {
                AnalysisType.DEPENDENCY_ANALYSIS,
                AnalysisType.IMPACT_ANALYSIS,
                AnalysisType.ARCHITECTURE_VALIDATION,
                AnalysisType.COMPLEXITY_ANALYSIS,
                AnalysisType.DEAD_CODE_DETECTION
            }
            self.realtime_analyzer.configure_analyses(all_analyses)
            
            # Run full analysis
            results = await self.realtime_analyzer.run_full_analysis()
            
            if output_format == "json":
                return results
            elif output_format == "summary":
                return self._format_full_analysis_summary(results)
            elif output_format == "report":
                return self._generate_full_analysis_report(results)
            
        except Exception as e:
            logger.error("Full analysis failed: %s", e)
            return {"error": str(e), "success": False}
    
    def _format_full_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Format full analysis results as summary"""
        lines = []
        lines.append("🔍 Comprehensive Codebase Analysis Summary")
        lines.append("=" * 50)
        
        summary = results.get("analysis_summary", {})
        if summary.get("success"):
            lines.append(f"✅ Analysis completed in {summary.get('duration_seconds', 0):.1f} seconds")
        else:
            lines.append(f"❌ Analysis failed: {summary.get('error', 'Unknown error')}")
            return "\n".join(lines)
        
        # Dependencies
        if "dependencies" in results:
            deps = results["dependencies"]
            lines.append(f"\n📦 Dependencies: {len(deps.get('modules', []))} modules analyzed")
        
        # Architecture
        if "architecture" in results:
            arch = results["architecture"]
            total_violations = arch.get("total_violations", 0)
            status_icon = "✅" if total_violations == 0 else "⚠️"
            lines.append(f"\n🏗️ Architecture: {status_icon} {total_violations} violations found")
        
        # Complexity
        if "complexity" in results:
            complexity = results["complexity"]
            avg_complexity = complexity.get("average_complexity", 0)
            lines.append(f"\n🧮 Complexity: Average {avg_complexity:.1f}")
        
        # Dead code
        if "dead_code" in results:
            dead_code = results["dead_code"]
            dead_functions = len(dead_code.get("unused_functions", []))
            lines.append(f"\n💀 Dead Code: {dead_functions} unused functions detected")
        
        return "\n".join(lines)
    
    def _generate_impact_report(self, impact: Dict[str, Any]) -> str:
        """Generate detailed impact analysis report"""
        lines = []
        lines.append("💥 Detailed Change Impact Report")
        lines.append("=" * 40)
        
        # Add summary
        lines.append(self._format_impact_summary(impact))
        
        # Add detailed analysis
        if impact.get("affected_files"):
            lines.append("\n📁 Affected Files:")
            for file_info in impact["affected_files"][:10]:
                lines.append(f"  • {file_info}")
        
        if impact.get("recommendations"):
            lines.append("\n💡 Recommendations:")
            for rec in impact["recommendations"][:5]:
                lines.append(f"  • {rec}")
        
        return "\n".join(lines)
    
    def _generate_full_analysis_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive full analysis report"""
        lines = []
        lines.append("🔍 Comprehensive Analysis Report")
        lines.append("=" * 40)
        
        # Add summary
        lines.append(self._format_full_analysis_summary(results))
        
        # Add detailed sections
        if "architecture" in results:
            lines.append("\n🏗️ Architecture Details:")
            arch = results["architecture"]
            if arch.get("violations"):
                for violation in arch["violations"][:5]:
                    lines.append(f"  • {violation.get('description', 'Unknown violation')}")
        
        if "dependencies" in results:
            lines.append("\n📦 Dependency Analysis:")
            deps = results["dependencies"]
            if deps.get("critical_modules"):
                lines.append("  Critical modules:")
                for module in deps["critical_modules"][:3]:
                    lines.append(f"    - {module.get('module', 'Unknown')}")
        
        return "\n".join(lines)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.realtime_analyzer:
            await self.realtime_analyzer.cleanup()
        if self.glean_client:
            await self.glean_client.cleanup()

# CLI command functions for integration with existing framework
async def glean_analyze_dependencies(module: str, depth: int = 2, output_format: str = "json"):
    """CLI command for dependency analysis"""
    cli = GleanCLI()
    try:
        result = await cli.analyze_dependencies(module, depth, output_format)
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(result)
    finally:
        await cli.cleanup()

async def glean_analyze_impact(files: List[str], output_format: str = "summary"):
    """CLI command for impact analysis"""
    cli = GleanCLI()
    try:
        result = await cli.analyze_impact(files, output_format)
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(result)
    finally:
        await cli.cleanup()

async def glean_validate_architecture(output_format: str = "summary"):
    """CLI command for architecture validation"""
    cli = GleanCLI()
    try:
        result = await cli.validate_architecture(output_format)
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(result)
    finally:
        await cli.cleanup()

async def glean_start_monitoring(analyses: Optional[List[str]] = None):
    """CLI command to start real-time monitoring"""
    cli = GleanCLI()
    try:
        result = await cli.start_realtime_monitoring(analyses)
        click.echo(json.dumps(result, indent=2))
        
        if result.get("success"):
            click.echo("\n🔄 Monitoring active. Press Ctrl+C to stop.")
            try:
                # Keep running until interrupted
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                click.echo("\n🛑 Stopping monitoring...")
                cli.realtime_analyzer.stop_watching()
    finally:
        await cli.cleanup()

async def glean_full_analysis(output_format: str = "summary"):
    """CLI command for full analysis"""
    cli = GleanCLI()
    try:
        result = await cli.run_full_analysis(output_format)
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(result)
    finally:
        await cli.cleanup()

async def glean_status():
    """CLI command to get monitoring status"""
    cli = GleanCLI()
    try:
        result = await cli.get_monitoring_status()
        click.echo(json.dumps(result, indent=2))
    finally:
        await cli.cleanup()

# CLRS+Tree Analysis CLI Commands
async def clrs_analyze_code(file_path: str, algorithm: str = "all", output_format: str = "json"):
    """CLI command for CLRS algorithmic analysis of code"""
    try:
        from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import CLRSAnalysisTool
        
        tool = CLRSAnalysisTool()
        result = await tool.execute({
            "file_path": file_path,
            "algorithm": algorithm
        })
        
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"CLRS Analysis Results for {file_path}:")
            if "complexity" in result:
                click.echo(f"  Time Complexity: {result['complexity'].get('time', 'N/A')}")
                click.echo(f"  Space Complexity: {result['complexity'].get('space', 'N/A')}")
            if "recommendations" in result:
                click.echo("  Optimization Recommendations:")
                for rec in result["recommendations"][:3]:
                    click.echo(f"    • {rec}")
                    
    except ImportError:
        click.echo("❌ CLRS analysis tools not available")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

async def tree_analyze_structure(file_path: str, operation: str = "analyze", output_format: str = "json"):
    """CLI command for Tree library structure analysis"""
    try:
        from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import HierarchicalIndexingTool
        
        tool = HierarchicalIndexingTool()
        result = await tool.execute({
            "file_path": file_path,
            "operation": operation
        })
        
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Tree Structure Analysis for {file_path}:")
            if "hierarchy" in result:
                click.echo(f"  Depth: {result['hierarchy'].get('depth', 'N/A')}")
                click.echo(f"  Nodes: {result['hierarchy'].get('node_count', 'N/A')}")
                click.echo(f"  Leaves: {result['hierarchy'].get('leaf_count', 'N/A')}")
            if "structure" in result:
                click.echo("  Structure Overview:")
                for item in result["structure"][:5]:
                    click.echo(f"    • {item}")
                    
    except ImportError:
        click.echo("❌ Tree analysis tools not available")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

async def dependency_graph_analysis(project_path: str = ".", algorithm: str = "dfs", output_format: str = "json"):
    """CLI command for dependency graph analysis using CLRS algorithms"""
    try:
        from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import DependencyGraphTool
        
        tool = DependencyGraphTool()
        result = await tool.execute({
            "project_path": project_path,
            "algorithm": algorithm
        })
        
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Dependency Graph Analysis ({algorithm.upper()}):")
            if "graph_metrics" in result:
                metrics = result["graph_metrics"]
                click.echo(f"  Nodes: {metrics.get('node_count', 'N/A')}")
                click.echo(f"  Edges: {metrics.get('edge_count', 'N/A')}")
                click.echo(f"  Cycles: {metrics.get('cycle_count', 'N/A')}")
            if "critical_paths" in result:
                click.echo("  Critical Dependencies:")
                for path in result["critical_paths"][:3]:
                    click.echo(f"    • {' → '.join(path)}")
                    
    except ImportError:
        click.echo("❌ Dependency graph tools not available")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

async def code_similarity_analysis(file1: str, file2: str, algorithm: str = "lcs", output_format: str = "json"):
    """CLI command for code similarity analysis using CLRS string algorithms"""
    try:
        from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import CodeSimilarityTool
        
        tool = CodeSimilarityTool()
        result = await tool.execute({
            "file1": file1,
            "file2": file2,
            "algorithm": algorithm
        })
        
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Code Similarity Analysis ({algorithm.upper()}):")
            if "similarity_score" in result:
                score = result["similarity_score"]
                click.echo(f"  Similarity Score: {score:.2%}")
            if "common_patterns" in result:
                click.echo("  Common Patterns:")
                for pattern in result["common_patterns"][:3]:
                    click.echo(f"    • {pattern}")
            if "differences" in result:
                click.echo(f"  Differences Found: {len(result['differences'])}")
                    
    except ImportError:
        click.echo("❌ Code similarity tools not available")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

async def optimization_recommendations(project_path: str = ".", focus: str = "performance", output_format: str = "json"):
    """CLI command for optimization recommendations using CLRS+Tree analysis"""
    try:
        from cryptotrading.infrastructure.analysis.clrs_tree_mcp_tools import OptimizationRecommendationTool
        
        tool = OptimizationRecommendationTool()
        result = await tool.execute({
            "project_path": project_path,
            "focus": focus
        })
        
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Optimization Recommendations ({focus}):")
            if "recommendations" in result:
                for i, rec in enumerate(result["recommendations"][:5], 1):
                    click.echo(f"  {i}. {rec.get('title', 'Optimization')}")
                    click.echo(f"     Impact: {rec.get('impact', 'Unknown')}")
                    click.echo(f"     Effort: {rec.get('effort', 'Unknown')}")
            if "metrics" in result:
                click.echo(f"  Current Performance Score: {result['metrics'].get('score', 'N/A')}")
                    
    except ImportError:
        click.echo("❌ Optimization tools not available")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
    finally:
        await cli.cleanup()
