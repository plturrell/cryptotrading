#!/usr/bin/env python3
"""
A2A CLRS Algorithms Agent CLI
Provides command-line interface for algorithmic calculations and CLRS operations
"""

import os
import sys
import asyncio
import click
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set environment variables for CLI
os.environ['ENVIRONMENT'] = 'development'
os.environ['SKIP_DB_INIT'] = 'true'

try:
    from cryptotrading.core.protocols.a2a.a2a_protocol import A2A_CAPABILITIES
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback minimal CLRS Algorithms agent for CLI testing...")
    
    class FallbackCLRSAlgorithmsAgent:
        """Minimal CLRS Algorithms agent for CLI testing when imports fail"""
        def __init__(self):
            self.agent_id = "clrs_algorithms_agent"
            self.capabilities = [
                'binary_search', 'linear_search', 'quick_select', 'find_minimum', 'find_maximum',
                'insertion_sort', 'merge_sort', 'quick_sort', 'algorithmic_calculations',
                'search_algorithms', 'sorting_algorithms', 'clrs_algorithms'
            ]
            
        async def binary_search(self, array, target):
            """Mock binary search"""
            # Simulate binary search
            if not array:
                return {"found": False, "index": -1, "comparisons": 0}
            
            left, right = 0, len(array) - 1
            comparisons = 0
            
            while left <= right:
                comparisons += 1
                mid = (left + right) // 2
                if array[mid] == target:
                    return {
                        "found": True,
                        "index": mid,
                        "value": target,
                        "comparisons": comparisons,
                        "complexity": f"O(log n) - {comparisons} comparisons",
                        "timestamp": datetime.now().isoformat()
                    }
                elif array[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return {
                "found": False,
                "index": -1,
                "comparisons": comparisons,
                "complexity": f"O(log n) - {comparisons} comparisons",
                "timestamp": datetime.now().isoformat()
            }
            
        async def quick_sort(self, array):
            """Mock quick sort"""
            original = array.copy()
            sorted_array = sorted(array)
            
            return {
                "original": original,
                "sorted": sorted_array,
                "pivots": len(array) // 2,
                "partitions": len(array) - 1,
                "complexity": "O(n log n) average",
                "timestamp": datetime.now().isoformat()
            }
            
        async def find_minimum_maximum(self, array):
            """Mock min/max finding"""
            if not array:
                return {"error": "Empty array"}
            
            min_val = min(array)
            max_val = max(array)
            min_idx = array.index(min_val)
            max_idx = array.index(max_val)
            
            return {
                "array_size": len(array),
                "minimum": {"value": min_val, "index": min_idx},
                "maximum": {"value": max_val, "index": max_idx},
                "comparisons": len(array) - 1,
                "complexity": "O(n)",
                "timestamp": datetime.now().isoformat()
            }
            
        async def linear_search(self, array, target):
            """Mock linear search"""
            try:
                index = array.index(target)
                found = True
            except ValueError:
                index = -1
                found = False
            
            return {
                "array_size": len(array),
                "target": target,
                "found": found,
                "index": index,
                "comparisons": index + 1 if found else len(array),
                "complexity": "O(n)",
                "timestamp": datetime.now().isoformat()
            }
            
        async def quick_select(self, array, k):
            """Mock quick select"""
            sorted_array = sorted(array)
            kth_element = sorted_array[k-1] if k <= len(array) else None
            
            return {
                "array_size": len(array),
                "k": k,
                "kth_element": kth_element,
                "partitions": 3,
                "complexity": "O(n) average",
                "timestamp": datetime.now().isoformat()
            }
            
        async def insertion_sort(self, array):
            """Mock insertion sort"""
            original = array.copy()
            sorted_array = sorted(array)
            
            return {
                "original": original,
                "sorted": sorted_array,
                "comparisons": len(array) * (len(array) - 1) // 2,
                "swaps": len(array) // 2,
                "complexity": "O(n²)",
                "timestamp": datetime.now().isoformat()
            }
            
        async def merge_sort(self, array):
            """Mock merge sort"""
            original = array.copy()
            sorted_array = sorted(array)
            
            return {
                "original": original,
                "sorted": sorted_array,
                "merges": len(array) - 1,
                "complexity": "O(n log n)",
                "timestamp": datetime.now().isoformat()
            }
    
    CLRSAlgorithmsAgent = FallbackCLRSAlgorithmsAgent

# Global agent instance
agent = None

def get_agent():
    """Get or create agent instance"""
    global agent
    if agent is None:
        agent = CLRSAlgorithmsAgent()
    return agent

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
    """A2A CLRS Algorithms Agent CLI - Classic algorithms and data structures"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

@cli.command()
@click.argument('target', type=int)
@click.argument('array', nargs=-1, type=int, required=True)
@click.pass_context
@async_command
async def search(ctx, target, array):
    """Perform binary search on sorted array"""
    agent = get_agent()
    
    try:
        sorted_array = sorted(array)
        result = await agent.binary_search(sorted_array, target)
        
        click.echo(f"🔍 Binary Search for {target}")
        click.echo("=" * 50)
        click.echo(f"Array: {sorted_array}")
        click.echo(f"Target: {target}")
        click.echo()
        
        if result['found']:
            click.echo(f"✅ Found at index {result['index']}")
            click.echo(f"Value: {result['value']}")
        else:
            click.echo("❌ Not found")
        
        click.echo(f"Comparisons: {result['comparisons']}")
        click.echo(f"Complexity: {result['complexity']}")
        
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error performing binary search: {e}", err=True)

@cli.command()
@click.argument('array', nargs=-1, type=int, required=True)
@click.pass_context
@async_command
async def quick_sort(ctx, array):
    """Sort array using quicksort algorithm"""
    agent = get_agent()
    
    try:
        result = await agent.quick_sort(list(array))
        
        click.echo("⚡ Quick Sort")
        click.echo("=" * 50)
        click.echo(f"Original: {result['original']}")
        click.echo(f"Sorted:   {result['sorted']}")
        click.echo()
        click.echo(f"Pivots: {result['pivots']}")
        click.echo(f"Partitions: {result['partitions']}")
        click.echo(f"Complexity: {result['complexity']}")
        
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error performing quick sort: {e}", err=True)

@cli.command()
@click.argument('array', nargs=-1, type=int, required=True)
@click.pass_context
@async_command
async def minmax(ctx, array):
    """Find minimum and maximum values"""
    agent = get_agent()
    
    try:
        result = await agent.find_minimum_maximum(list(array))
        
        if 'error' in result:
            click.echo(f"❌ {result['error']}", err=True)
            return
        
        click.echo("📊 Min/Max Finding")
        click.echo("=" * 50)
        click.echo(f"Array Size: {result['array_size']}")
        click.echo()
        click.echo(f"Minimum: {result['minimum']['value']} at index {result['minimum']['index']}")
        click.echo(f"Maximum: {result['maximum']['value']} at index {result['maximum']['index']}")
        click.echo()
        click.echo(f"Comparisons: {result['comparisons']}")
        click.echo(f"Complexity: {result['complexity']}")
        
        if ctx.obj['verbose']:
            click.echo(f"\nTimestamp: {result['timestamp']}")
            
    except Exception as e:
        click.echo(f"Error finding min/max: {e}", err=True)

@cli.command()
@click.pass_context
def algorithms(ctx):
    """List available algorithms"""
    agent = get_agent()
    
    algorithms_list = [
        ("Binary Search", "Search sorted array in O(log n)"),
        ("Linear Search", "Search unsorted array in O(n)"),
        ("Quick Sort", "Sort array in O(n log n) average"),
        ("Merge Sort", "Sort array in O(n log n) guaranteed"),
        ("Insertion Sort", "Sort array in O(n²) worst case"),
        ("Min/Max Finding", "Find extremes in O(n)"),
        ("Quick Select", "Find kth element in O(n) average")
    ]
    
    click.echo("🧮 Available CLRS Algorithms:")
    click.echo()
    for i, (name, desc) in enumerate(algorithms_list, 1):
        click.echo(f"{i:2d}. {name}")
        click.echo(f"    {desc}")
        click.echo()

@cli.command()
@click.pass_context
def capabilities(ctx):
    """List agent capabilities"""
    agent = get_agent()
    
    click.echo("🔧 CLRS Algorithms Agent Capabilities:")
    click.echo()
    for i, capability in enumerate(agent.capabilities, 1):
        click.echo(f"{i:2d}. {capability.replace('_', ' ').title()}")

@cli.command()
@click.pass_context
def status(ctx):
    """Get agent status and health"""
    agent = get_agent()
    
    click.echo("🏥 CLRS Algorithms Agent Status:")
    click.echo(f"Agent ID: {agent.agent_id}")
    click.echo(f"Capabilities: {len(agent.capabilities)}")
    click.echo("Status: ✅ ACTIVE")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")

if __name__ == '__main__':
    cli()
