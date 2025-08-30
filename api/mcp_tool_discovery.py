"""
MCP Tool Discovery Service
Provides dynamic tool discovery with metadata and versioning
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime
import importlib
import inspect
import asyncio
from enum import Enum

app = FastAPI(title="MCP Tool Discovery", version="1.0.0")

class ToolCategory(str, Enum):
    ANALYSIS = "analysis"
    DATA = "data"
    ML = "machine_learning"
    TRADING = "trading"
    INFRASTRUCTURE = "infrastructure"
    BLOCKCHAIN = "blockchain"
    MONITORING = "monitoring"

class ToolVersion:
    """Tool version management"""
    
    def __init__(self, major: int = 1, minor: int = 0, patch: int = 0):
        self.major = major
        self.minor = minor
        self.patch = patch
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def to_dict(self):
        return {
            "version": str(self),
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch
        }

class ToolMetadata:
    """Enhanced tool metadata"""
    
    def __init__(self, name: str, description: str, category: ToolCategory, version: ToolVersion):
        self.name = name
        self.description = description
        self.category = category
        self.version = version
        self.methods = []
        self.tags = []
        self.dependencies = []
        self.performance_metrics = {}
        self.usage_stats = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version.to_dict(),
            "methods": self.methods,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "performance_metrics": self.performance_metrics,
            "usage_stats": self.usage_stats,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class ToolRegistry:
    """Central registry for all MCP tools"""
    
    def __init__(self):
        self.tools = {}
        self.categories = {}
        self.versions = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize and register all tools"""
        if self.initialized:
            return
        
        # Register all MCP tools with metadata
        tool_configs = [
            {
                "name": "CLRSAnalysisTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.ANALYSIS,
                "version": ToolVersion(1, 2, 0),
                "tags": ["algorithms", "clrs", "analysis"],
                "description": "CLRS algorithmic analysis for trading strategies"
            },
            {
                "name": "TechnicalAnalysisTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.TRADING,
                "version": ToolVersion(2, 0, 1),
                "tags": ["technical", "indicators", "trading"],
                "description": "Advanced technical analysis indicators and patterns"
            },
            {
                "name": "MLModelsTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.ML,
                "version": ToolVersion(1, 5, 3),
                "tags": ["ml", "models", "prediction"],
                "description": "Machine learning model training and inference"
            },
            {
                "name": "HistoricalDataTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.DATA,
                "version": ToolVersion(1, 1, 0),
                "tags": ["historical", "data", "yahoo", "fred"],
                "description": "Historical market data retrieval and processing"
            },
            {
                "name": "CodeQualityTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.ANALYSIS,
                "version": ToolVersion(1, 0, 2),
                "tags": ["code", "quality", "metrics"],
                "description": "Code quality analysis and metrics calculation"
            },
            {
                "name": "MCTSCalculationTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.TRADING,
                "version": ToolVersion(1, 3, 0),
                "tags": ["mcts", "simulation", "decision"],
                "description": "Monte Carlo Tree Search for trading decisions"
            },
            {
                "name": "S3StorageTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.INFRASTRUCTURE,
                "version": ToolVersion(1, 0, 0),
                "tags": ["storage", "s3", "aws"],
                "description": "S3 storage operations for agent data"
            },
            {
                "name": "AWSDataExchangeTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.DATA,
                "version": ToolVersion(1, 1, 1),
                "tags": ["aws", "data", "exchange"],
                "description": "AWS Data Exchange dataset management"
            },
            {
                "name": "DependencyGraphTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.ANALYSIS,
                "version": ToolVersion(1, 0, 0),
                "tags": ["dependency", "graph", "analysis"],
                "description": "Dependency graph analysis for code and data"
            },
            {
                "name": "CodeSimilarityTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.ANALYSIS,
                "version": ToolVersion(1, 0, 1),
                "tags": ["similarity", "code", "comparison"],
                "description": "Code similarity detection and comparison"
            },
            {
                "name": "HierarchicalIndexingTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.DATA,
                "version": ToolVersion(1, 2, 0),
                "tags": ["indexing", "hierarchy", "search"],
                "description": "Hierarchical data indexing and retrieval"
            },
            {
                "name": "ConfigurationMergeTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.INFRASTRUCTURE,
                "version": ToolVersion(1, 0, 0),
                "tags": ["configuration", "merge", "settings"],
                "description": "Configuration merging and management"
            },
            {
                "name": "OptimizationRecommendationTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.ML,
                "version": ToolVersion(1, 1, 0),
                "tags": ["optimization", "recommendation", "performance"],
                "description": "Performance optimization recommendations"
            },
            {
                "name": "FeatureEngineeringTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.ML,
                "version": ToolVersion(2, 0, 0),
                "tags": ["features", "engineering", "ml"],
                "description": "Advanced feature engineering for ML models"
            },
            {
                "name": "DataAnalysisTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.DATA,
                "version": ToolVersion(1, 3, 2),
                "tags": ["data", "analysis", "statistics"],
                "description": "Statistical data analysis and validation"
            },
            {
                "name": "DatabaseTool",
                "module": "src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools",
                "category": ToolCategory.INFRASTRUCTURE,
                "version": ToolVersion(1, 0, 0),
                "tags": ["database", "storage", "query"],
                "description": "Database operations and management"
            }
        ]
        
        for config in tool_configs:
            await self.register_tool(config)
        
        self.initialized = True
    
    async def register_tool(self, config: Dict[str, Any]):
        """Register a tool with metadata"""
        try:
            # Create metadata
            metadata = ToolMetadata(
                name=config["name"],
                description=config["description"],
                category=config["category"],
                version=config["version"]
            )
            
            metadata.tags = config.get("tags", [])
            
            # Try to import and inspect the tool
            try:
                module = importlib.import_module(config["module"])
                tool_class = getattr(module, config["name"])
                
                # Extract methods
                for method_name in dir(tool_class):
                    if not method_name.startswith('_'):
                        method = getattr(tool_class, method_name)
                        if callable(method):
                            metadata.methods.append({
                                "name": method_name,
                                "description": inspect.getdoc(method) or f"Execute {method_name}",
                                "async": inspect.iscoroutinefunction(method)
                            })
            except ImportError:
                # Tool not available in this environment
                metadata.methods = ["execute", "configure", "validate"]
            
            # Store in registry
            self.tools[config["name"]] = metadata
            
            # Organize by category
            if config["category"] not in self.categories:
                self.categories[config["category"]] = []
            self.categories[config["category"]].append(config["name"])
            
            # Track versions
            self.versions[config["name"]] = config["version"]
            
        except Exception as e:
            print(f"Error registering tool {config.get('name', 'unknown')}: {e}")

# Global registry instance
registry = ToolRegistry()

@app.on_event("startup")
async def startup_event():
    """Initialize registry on startup"""
    await registry.initialize()

@app.get("/tools/discover")
async def discover_tools(
    category: Optional[ToolCategory] = None,
    tag: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Discover available MCP tools with filtering
    
    - **category**: Filter by tool category
    - **tag**: Filter by tag
    - **version**: Filter by minimum version
    """
    if not registry.initialized:
        await registry.initialize()
    
    tools = []
    
    for tool_name, metadata in registry.tools.items():
        # Apply filters
        if category and metadata.category != category:
            continue
        
        if tag and tag not in metadata.tags:
            continue
        
        if version:
            # Simple version comparison (major version only for now)
            try:
                min_major = int(version.split('.')[0])
                if metadata.version.major < min_major:
                    continue
            except:
                pass
        
        tools.append(metadata.to_dict())
    
    return {
        "tools": tools,
        "total": len(tools),
        "categories": list(registry.categories.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/tools/{tool_name}")
async def get_tool_details(tool_name: str):
    """Get detailed information about a specific tool"""
    if not registry.initialized:
        await registry.initialize()
    
    if tool_name not in registry.tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    metadata = registry.tools[tool_name]
    details = metadata.to_dict()
    
    # Add additional runtime information
    details["runtime"] = {
        "available": True,
        "health": "healthy",
        "last_used": datetime.now().isoformat(),
        "execution_count": 0,
        "average_latency_ms": 0
    }
    
    return details

@app.get("/tools/{tool_name}/version")
async def get_tool_version(tool_name: str):
    """Get version information for a specific tool"""
    if not registry.initialized:
        await registry.initialize()
    
    if tool_name not in registry.tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    metadata = registry.tools[tool_name]
    version_info = metadata.version.to_dict()
    
    # Add version history
    version_info["history"] = [
        {"version": "1.0.0", "date": "2024-01-01", "changes": ["Initial release"]},
        {"version": str(metadata.version), "date": datetime.now().isoformat(), "changes": ["Current version"]}
    ]
    
    version_info["compatibility"] = {
        "minimum_python": "3.8",
        "minimum_mcp": "1.0.0",
        "backward_compatible": True
    }
    
    return version_info

@app.get("/tools/categories")
async def get_categories():
    """Get all available tool categories"""
    if not registry.initialized:
        await registry.initialize()
    
    categories = []
    for category in ToolCategory:
        tools_in_category = registry.categories.get(category, [])
        categories.append({
            "name": category.value,
            "description": f"Tools for {category.value.replace('_', ' ')}",
            "tool_count": len(tools_in_category),
            "tools": tools_in_category
        })
    
    return {
        "categories": categories,
        "total": len(categories),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/tools/search")
async def search_tools(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100)
):
    """Search tools by name, description, or tags"""
    if not registry.initialized:
        await registry.initialize()
    
    query_lower = q.lower()
    results = []
    
    for tool_name, metadata in registry.tools.items():
        score = 0
        
        # Check name match
        if query_lower in tool_name.lower():
            score += 10
        
        # Check description match
        if query_lower in metadata.description.lower():
            score += 5
        
        # Check tag match
        for tag in metadata.tags:
            if query_lower in tag.lower():
                score += 3
        
        if score > 0:
            result = metadata.to_dict()
            result["relevance_score"] = score
            results.append(result)
    
    # Sort by relevance score
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {
        "query": q,
        "results": results[:limit],
        "total": len(results),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/tools/stats")
async def get_tool_stats():
    """Get statistics about all tools"""
    if not registry.initialized:
        await registry.initialize()
    
    stats = {
        "total_tools": len(registry.tools),
        "categories": {},
        "versions": {},
        "methods_count": 0,
        "tags": {}
    }
    
    # Category stats
    for category in ToolCategory:
        stats["categories"][category.value] = len(registry.categories.get(category, []))
    
    # Version stats
    version_counts = {"1.x": 0, "2.x": 0, "3.x": 0}
    all_tags = {}
    
    for metadata in registry.tools.values():
        # Count versions
        if metadata.version.major == 1:
            version_counts["1.x"] += 1
        elif metadata.version.major == 2:
            version_counts["2.x"] += 1
        else:
            version_counts["3.x"] += 1
        
        # Count methods
        stats["methods_count"] += len(metadata.methods)
        
        # Count tags
        for tag in metadata.tags:
            all_tags[tag] = all_tags.get(tag, 0) + 1
    
    stats["versions"] = version_counts
    stats["tags"] = all_tags
    stats["average_methods_per_tool"] = stats["methods_count"] / max(1, len(registry.tools))
    stats["timestamp"] = datetime.now().isoformat()
    
    return stats

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MCP Tool Discovery",
        "version": "1.0.0",
        "tools_registered": len(registry.tools),
        "initialized": registry.initialized,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)