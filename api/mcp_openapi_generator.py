"""
MCP OpenAPI Specification Generator
Generates OpenAPI 3.0 specifications from MCP tool definitions
"""

import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
import inspect
import re

class MCPOpenAPIGenerator:
    """Generate OpenAPI specifications from MCP tools"""
    
    def __init__(self):
        self.openapi_version = "3.0.3"
        self.base_spec = {
            "openapi": self.openapi_version,
            "info": {
                "title": "MCP Tools API",
                "description": "Model Context Protocol Tools for A2A Crypto Trading Platform",
                "version": "1.0.0",
                "contact": {
                    "name": "A2A Platform Team",
                    "email": "support@rex.com"
                },
                "license": {
                    "name": "MIT"
                }
            },
            "servers": [
                {
                    "url": "https://api.rex.com/mcp",
                    "description": "Production server"
                },
                {
                    "url": "http://localhost:8000/mcp",
                    "description": "Development server"
                }
            ],
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    },
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                },
                "schemas": {},
                "responses": {
                    "UnauthorizedError": {
                        "description": "Authentication information is missing or invalid",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            },
            "paths": {},
            "security": [
                {"ApiKeyAuth": []},
                {"BearerAuth": []}
            ]
        }
        
    def extract_tool_metadata(self, tool_class) -> Dict[str, Any]:
        """Extract metadata from a tool class"""
        metadata = {
            "name": tool_class.__name__,
            "description": inspect.getdoc(tool_class) or "No description available",
            "methods": []
        }
        
        # Extract methods
        for name, method in inspect.getmembers(tool_class, predicate=inspect.ismethod):
            if not name.startswith('_'):
                method_info = {
                    "name": name,
                    "description": inspect.getdoc(method) or f"Execute {name} operation",
                    "parameters": self._extract_parameters(method)
                }
                metadata["methods"].append(method_info)
                
        return metadata
    
    def _extract_parameters(self, method) -> List[Dict[str, Any]]:
        """Extract parameters from a method signature"""
        params = []
        sig = inspect.signature(method)
        
        for param_name, param in sig.parameters.items():
            if param_name not in ['self', 'cls']:
                param_info = {
                    "name": param_name,
                    "in": "query",
                    "required": param.default == inspect.Parameter.empty,
                    "schema": {"type": "string"}  # Default type
                }
                
                # Try to infer type from annotation
                if param.annotation != inspect.Parameter.empty:
                    param_info["schema"] = self._python_type_to_openapi(param.annotation)
                    
                params.append(param_info)
                
        return params
    
    def _python_type_to_openapi(self, py_type) -> Dict[str, Any]:
        """Convert Python type to OpenAPI schema type"""
        type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array", "items": {"type": "string"}},
            dict: {"type": "object"},
            datetime: {"type": "string", "format": "date-time"}
        }
        
        # Handle Optional types
        if hasattr(py_type, '__origin__'):
            if py_type.__origin__ is list:
                return {"type": "array", "items": {"type": "string"}}
            elif py_type.__origin__ is dict:
                return {"type": "object"}
                
        return type_mapping.get(py_type, {"type": "string"})
    
    def generate_tool_path(self, tool_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate OpenAPI path for a tool"""
        tool_name = tool_metadata["name"].lower().replace("tool", "")
        base_path = f"/tools/{tool_name}"
        
        paths = {}
        
        for method in tool_metadata["methods"]:
            method_path = f"{base_path}/{method['name']}"
            
            paths[method_path] = {
                "post": {
                    "summary": f"Execute {method['name']} on {tool_metadata['name']}",
                    "description": method["description"],
                    "operationId": f"{tool_name}_{method['name']}",
                    "tags": [tool_metadata["name"]],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        param["name"]: param["schema"]
                                        for param in method["parameters"]
                                    },
                                    "required": [
                                        param["name"] 
                                        for param in method["parameters"] 
                                        if param["required"]
                                    ]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful operation",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": f"#/components/schemas/{tool_metadata['name']}Response"
                                    }
                                }
                            }
                        },
                        "401": {
                            "$ref": "#/components/responses/UnauthorizedError"
                        }
                    }
                }
            }
            
        return paths
    
    def generate_schemas(self, tool_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response schemas for a tool"""
        schemas = {}
        
        # Generate response schema
        response_schema = {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {"type": "object"},
                "timestamp": {"type": "string", "format": "date-time"},
                "tool": {"type": "string"},
                "method": {"type": "string"}
            },
            "required": ["success", "data", "timestamp"]
        }
        
        schemas[f"{tool_metadata['name']}Response"] = response_schema
        
        return schemas
    
    def add_discovery_endpoint(self):
        """Add tool discovery endpoint to OpenAPI spec"""
        discovery_path = {
            "/tools/discover": {
                "get": {
                    "summary": "Discover available MCP tools",
                    "description": "Returns a list of all available MCP tools with their metadata",
                    "operationId": "discoverTools",
                    "tags": ["Discovery"],
                    "responses": {
                        "200": {
                            "description": "List of available tools",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "tools": {
                                                "type": "array",
                                                "items": {
                                                    "$ref": "#/components/schemas/ToolMetadata"
                                                }
                                            },
                                            "version": {"type": "string"},
                                            "timestamp": {"type": "string", "format": "date-time"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Add ToolMetadata schema
        tool_metadata_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "version": {"type": "string"},
                "category": {"type": "string"},
                "methods": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {"type": "array"}
                        }
                    }
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        self.base_spec["paths"].update(discovery_path)
        self.base_spec["components"]["schemas"]["ToolMetadata"] = tool_metadata_schema
        
        # Add Error schema
        self.base_spec["components"]["schemas"]["Error"] = {
            "type": "object",
            "properties": {
                "error": {"type": "string"},
                "message": {"type": "string"},
                "code": {"type": "integer"}
            }
        }
    
    def generate_from_tools(self, tools: List[Any]) -> Dict[str, Any]:
        """Generate complete OpenAPI spec from list of tools"""
        spec = self.base_spec.copy()
        
        # Add discovery endpoint
        self.add_discovery_endpoint()
        
        # Process each tool
        for tool in tools:
            try:
                metadata = self.extract_tool_metadata(tool)
                
                # Generate paths
                paths = self.generate_tool_path(metadata)
                spec["paths"].update(paths)
                
                # Generate schemas
                schemas = self.generate_schemas(metadata)
                spec["components"]["schemas"].update(schemas)
                
            except Exception as e:
                print(f"Error processing tool {tool}: {e}")
                continue
        
        # Add metadata
        spec["info"]["x-generated-at"] = datetime.now().isoformat()
        spec["info"]["x-tool-count"] = len(tools)
        
        return spec
    
    def save_spec(self, spec: Dict[str, Any], format: str = "yaml", filepath: str = None):
        """Save OpenAPI specification to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"openapi_spec_{timestamp}.{format}"
        
        if format == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(spec, f, indent=2)
        
        return filepath
    
    def validate_spec(self, spec: Dict[str, Any]) -> List[str]:
        """Basic validation of OpenAPI spec"""
        errors = []
        
        # Check required fields
        if "openapi" not in spec:
            errors.append("Missing 'openapi' field")
        if "info" not in spec:
            errors.append("Missing 'info' field")
        if "paths" not in spec:
            errors.append("Missing 'paths' field")
            
        # Check paths
        if len(spec.get("paths", {})) == 0:
            errors.append("No paths defined")
            
        # Check components
        if "components" in spec:
            if "schemas" in spec["components"]:
                if len(spec["components"]["schemas"]) == 0:
                    errors.append("No schemas defined")
                    
        return errors

def generate_openapi_for_all_mcp_tools():
    """Generate OpenAPI spec for all MCP tools in the system"""
    generator = MCPOpenAPIGenerator()
    
    # Import all MCP tools
    from src.cryptotrading.infrastructure.analysis.all_segregated_mcp_tools import (
        CLRSAnalysisTool, DependencyGraphTool, CodeSimilarityTool,
        HierarchicalIndexingTool, ConfigurationMergeTool, OptimizationRecommendationTool,
        TechnicalAnalysisTool, HistoricalDataTool, MLModelsTool,
        CodeQualityTool, MCTSCalculationTool, S3StorageTool,
        AWSDataExchangeTool, FeatureEngineeringTool, DataAnalysisTool,
        DatabaseTool
    )
    
    # List of all tools
    tools = [
        CLRSAnalysisTool,
        DependencyGraphTool,
        CodeSimilarityTool,
        HierarchicalIndexingTool,
        ConfigurationMergeTool,
        OptimizationRecommendationTool,
        TechnicalAnalysisTool,
        HistoricalDataTool,
        MLModelsTool,
        CodeQualityTool,
        MCTSCalculationTool,
        S3StorageTool,
        AWSDataExchangeTool,
        FeatureEngineeringTool,
        DataAnalysisTool,
        DatabaseTool
    ]
    
    # Generate specification
    spec = generator.generate_from_tools(tools)
    
    # Validate
    errors = generator.validate_spec(spec)
    if errors:
        print(f"Validation errors: {errors}")
    
    # Save in both formats
    yaml_file = generator.save_spec(spec, "yaml", "api/openapi_mcp_tools.yaml")
    json_file = generator.save_spec(spec, "json", "api/openapi_mcp_tools.json")
    
    print(f"OpenAPI spec generated: {yaml_file}, {json_file}")
    
    return spec

if __name__ == "__main__":
    spec = generate_openapi_for_all_mcp_tools()
    print(f"Generated OpenAPI spec with {len(spec['paths'])} endpoints")