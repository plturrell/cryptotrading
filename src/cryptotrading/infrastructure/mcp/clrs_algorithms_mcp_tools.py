"""
CLRS Algorithms MCP Tools - All search, sort, and algorithmic calculations
"""
import logging
from typing import Dict, Any, List, Optional, Callable
import json
from datetime import datetime

from ...infrastructure.analysis.clrs_algorithms import CLRSSearchAlgorithms, CLRSSortingAlgorithms

logger = logging.getLogger(__name__)


class CLRSAlgorithmsMCPTools:
    """MCP tools for CLRS algorithms and calculations"""
    
    def __init__(self):
        self.search_algorithms = CLRSSearchAlgorithms()
        self.sorting_algorithms = CLRSSortingAlgorithms()
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create MCP tool definitions for CLRS algorithms"""
        return [
            {
                "name": "binary_search",
                "description": "Perform binary search on sorted array - O(log n)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "array": {"type": "array", "description": "Sorted array to search"},
                        "target": {"description": "Target value to find"},
                        "data_type": {"type": "string", "enum": ["number", "string"], "default": "number"}
                    },
                    "required": ["array", "target"]
                }
            },
            {
                "name": "linear_search",
                "description": "Perform linear search on array - O(n)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "array": {"type": "array", "description": "Array to search"},
                        "target": {"description": "Target value to find"},
                        "data_type": {"type": "string", "enum": ["number", "string"], "default": "number"}
                    },
                    "required": ["array", "target"]
                }
            },
            {
                "name": "quick_select",
                "description": "Find k-th smallest element - O(n) average case",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "array": {"type": "array", "description": "Array to search"},
                        "k": {"type": "integer", "description": "Position (1-indexed) of element to find"},
                        "data_type": {"type": "string", "enum": ["number", "string"], "default": "number"}
                    },
                    "required": ["array", "k"]
                }
            },
            {
                "name": "insertion_sort",
                "description": "Sort array using insertion sort - O(n²) time, O(1) space",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "array": {"type": "array", "description": "Array to sort"},
                        "ascending": {"type": "boolean", "default": True, "description": "Sort in ascending order"},
                        "data_type": {"type": "string", "enum": ["number", "string"], "default": "number"}
                    },
                    "required": ["array"]
                }
            },
            {
                "name": "merge_sort",
                "description": "Sort array using merge sort - O(n log n) time, O(n) space",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "array": {"type": "array", "description": "Array to sort"},
                        "ascending": {"type": "boolean", "default": True, "description": "Sort in ascending order"},
                        "data_type": {"type": "string", "enum": ["number", "string"], "default": "number"}
                    },
                    "required": ["array"]
                }
            },
            {
                "name": "quick_sort",
                "description": "Sort array using quicksort - O(n log n) average time",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "array": {"type": "array", "description": "Array to sort"},
                        "ascending": {"type": "boolean", "default": True, "description": "Sort in ascending order"},
                        "data_type": {"type": "string", "enum": ["number", "string"], "default": "number"}
                    },
                    "required": ["array"]
                }
            }
        ]
    
    def register_tools(self, server):
        """Register all CLRS algorithm tools with MCP server"""
        for tool_def in self.tools:
            tool_name = tool_def["name"]
            
            @server.call_tool()
            async def handle_tool(name: str, arguments: dict) -> dict:
                if name == tool_name:
                    return await self.handle_tool_call(tool_name, arguments)
                return {"error": f"Unknown tool: {name}"}
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls for CLRS algorithms"""
        try:
            if tool_name == "binary_search":
                return await self._handle_binary_search(arguments)
            elif tool_name == "linear_search":
                return await self._handle_linear_search(arguments)
            elif tool_name == "quick_select":
                return await self._handle_quick_select(arguments)
            elif tool_name == "insertion_sort":
                return await self._handle_insertion_sort(arguments)
            elif tool_name == "merge_sort":
                return await self._handle_merge_sort(arguments)
            elif tool_name == "quick_sort":
                return await self._handle_quick_sort(arguments)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Error in CLRS algorithm tool {tool_name}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_binary_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle binary search requests"""
        try:
            array = args["array"]
            target = args["target"]
            data_type = args.get("data_type", "number")
            
            # Convert data types
            if data_type == "number":
                array = [float(x) for x in array]
                target = float(target)
            
            # Create comparison function
            compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)
            
            result_index = self.search_algorithms.binary_search(array, target, compare_fn)
            
            return {
                "success": True,
                "result": {
                    "index": result_index,
                    "found": result_index != -1,
                    "value": array[result_index] if result_index != -1 else None,
                    "algorithm": "binary_search",
                    "complexity": "O(log n)"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_linear_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle linear search requests"""
        try:
            array = args["array"]
            target = args["target"]
            data_type = args.get("data_type", "number")
            
            # Convert data types
            if data_type == "number":
                array = [float(x) for x in array]
                target = float(target)
            
            result_index = self.search_algorithms.linear_search(array, target)
            
            return {
                "success": True,
                "result": {
                    "index": result_index,
                    "found": result_index != -1,
                    "value": array[result_index] if result_index != -1 else None,
                    "algorithm": "linear_search",
                    "complexity": "O(n)"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_quick_select(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quick select requests"""
        try:
            array = args["array"]
            k = args["k"]
            data_type = args.get("data_type", "number")
            
            # Convert data types
            if data_type == "number":
                array = [float(x) for x in array]
            
            # Create comparison function
            compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)
            
            result = self.search_algorithms.quick_select(array, k, compare_fn)
            
            return {
                "success": True,
                "result": {
                    "kth_element": result,
                    "k": k,
                    "algorithm": "quick_select",
                    "complexity": "O(n) average"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_insertion_sort(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle insertion sort requests"""
        try:
            array = args["array"]
            ascending = args.get("ascending", True)
            data_type = args.get("data_type", "number")
            
            # Convert data types
            if data_type == "number":
                array = [float(x) for x in array]
            
            # Create comparison function
            if ascending:
                compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)
            else:
                compare_fn = lambda a, b: 1 if a < b else (-1 if a > b else 0)
            
            sorted_array = self.sorting_algorithms.insertion_sort(array, compare_fn)
            
            return {
                "success": True,
                "result": {
                    "sorted_array": sorted_array,
                    "original_length": len(array),
                    "algorithm": "insertion_sort",
                    "complexity": "O(n²) time, O(1) space",
                    "ascending": ascending
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_merge_sort(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle merge sort requests"""
        try:
            array = args["array"]
            ascending = args.get("ascending", True)
            data_type = args.get("data_type", "number")
            
            # Convert data types
            if data_type == "number":
                array = [float(x) for x in array]
            
            # Create comparison function
            if ascending:
                compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)
            else:
                compare_fn = lambda a, b: 1 if a < b else (-1 if a > b else 0)
            
            sorted_array = self.sorting_algorithms.merge_sort(array, compare_fn)
            
            return {
                "success": True,
                "result": {
                    "sorted_array": sorted_array,
                    "original_length": len(array),
                    "algorithm": "merge_sort",
                    "complexity": "O(n log n) time, O(n) space",
                    "ascending": ascending
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_quick_sort(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quicksort requests"""
        try:
            array = args["array"]
            ascending = args.get("ascending", True)
            data_type = args.get("data_type", "number")
            
            # Convert data types
            if data_type == "number":
                array = [float(x) for x in array]
            
            # Create comparison function
            if ascending:
                compare_fn = lambda a, b: -1 if a < b else (1 if a > b else 0)
            else:
                compare_fn = lambda a, b: 1 if a < b else (-1 if a > b else 0)
            
            sorted_array = self.sorting_algorithms.quick_sort(array, compare_fn)
            
            return {
                "success": True,
                "result": {
                    "sorted_array": sorted_array,
                    "original_length": len(array),
                    "algorithm": "quick_sort",
                    "complexity": "O(n log n) average time",
                    "ascending": ascending
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
