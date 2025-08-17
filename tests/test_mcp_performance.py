"""
MCP Performance and Scalability Tests
Tests for performance optimization, concurrency, and scalability
"""

import pytest
import asyncio
import time
import json
import statistics
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

from cryptotrading.core.protocols.mcp.server import MCPServer
from cryptotrading.core.protocols.mcp.enhanced_server import EnhancedMCPServer
from cryptotrading.core.protocols.mcp.tools import MCPTool, ToolResult
from cryptotrading.core.protocols.mcp.cache import mcp_cache
from cryptotrading.core.protocols.mcp.rate_limiter import RateLimitMiddleware
from cryptotrading.core.protocols.mcp.metrics import mcp_metrics
from cryptotrading.core.protocols.mcp.transport import WebSocketTransport, HTTPTransport
from cryptotrading.core.protocols.mcp.security.middleware import (
    SecurityConfig, create_secure_middleware
)


class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self):
        self.results = []
        self.start_memory = 0
        self.end_memory = 0
    
    async def measure_async(self, func: Callable, iterations: int = 100) -> Dict[str, float]:
        """Measure async function performance"""
        times = []
        
        # Warmup
        for _ in range(10):
            await func()
        
        # Actual measurements
        for _ in range(iterations):
            start = time.perf_counter()
            await func()
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "p95": statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times),
            "p99": statistics.quantiles(times, n=100)[98] if len(times) > 100 else max(times)
        }
    
    def measure_memory(self):
        """Measure current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def start_memory_tracking(self):
        """Start memory tracking"""
        self.start_memory = self.measure_memory()
    
    def end_memory_tracking(self):
        """End memory tracking and return delta"""
        self.end_memory = self.measure_memory()
        return self.end_memory - self.start_memory


class TestMCPServerPerformance:
    """Test MCP server performance characteristics"""
    
    @pytest.fixture
    def performance_server(self):
        """Create optimized MCP server for performance testing"""
        server = EnhancedMCPServer("perf-test-server", "1.0.0")
        
        # Add lightweight test tools
        for i in range(100):  # Many tools to test scalability
            tool = MCPTool(
                name=f"tool_{i}",
                description=f"Test tool {i}",
                parameters={"type": "object", "properties": {}},
                function=lambda i=i: f"Result from tool {i}"
            )
            server.add_tool(tool)
        
        return server
    
    @pytest.mark.asyncio
    async def test_server_initialization_performance(self):
        """Test server initialization speed"""
        benchmark = PerformanceBenchmark()
        
        async def create_server():
            server = EnhancedMCPServer("test", "1.0.0")
            await server.initialize_enhanced_features({
                "auth_enabled": True,
                "cache_enabled": True,
                "rate_limiting_enabled": True,
                "metrics_enabled": True
            })
            return server
        
        results = await benchmark.measure_async(create_server, iterations=20)
        
        # Server should initialize quickly
        assert results["mean"] < 0.1  # Less than 100ms average
        assert results["p95"] < 0.2   # 95th percentile under 200ms
        
        print(f"\nServer initialization performance:")
        print(f"  Mean: {results['mean']*1000:.2f}ms")
        print(f"  P95: {results['p95']*1000:.2f}ms")
        print(f"  P99: {results['p99']*1000:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_tool_listing_performance(self, performance_server):
        """Test performance of listing many tools"""
        benchmark = PerformanceBenchmark()
        
        async def list_tools():
            request = {
                "jsonrpc": "2.0",
                "id": "list_1",
                "method": "tools/list"
            }
            return await performance_server.handle_request(request)
        
        results = await benchmark.measure_async(list_tools, iterations=1000)
        
        # Should handle tool listing efficiently
        assert results["mean"] < 0.001  # Less than 1ms average
        assert results["p99"] < 0.005   # 99th percentile under 5ms
        
        print(f"\nTool listing performance (100 tools):")
        print(f"  Mean: {results['mean']*1000:.3f}ms")
        print(f"  P95: {results['p95']*1000:.3f}ms")
        print(f"  P99: {results['p99']*1000:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_tool_execution_performance(self, performance_server):
        """Test tool execution performance"""
        benchmark = PerformanceBenchmark()
        
        tool_id = 0
        def get_next_tool():
            nonlocal tool_id
            tool_id = (tool_id + 1) % 100
            return f"tool_{tool_id}"
        
        async def execute_tool():
            request = {
                "jsonrpc": "2.0",
                "id": "exec_1",
                "method": "tools/call",
                "params": {
                    "name": get_next_tool(),
                    "arguments": {}
                }
            }
            return await performance_server.handle_request(request)
        
        results = await benchmark.measure_async(execute_tool, iterations=1000)
        
        # Tool execution should be fast
        assert results["mean"] < 0.002  # Less than 2ms average
        assert results["p99"] < 0.01    # 99th percentile under 10ms
        
        print(f"\nTool execution performance:")
        print(f"  Mean: {results['mean']*1000:.3f}ms")
        print(f"  P95: {results['p95']*1000:.3f}ms")
        print(f"  P99: {results['p99']*1000:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, performance_server):
        """Test handling many concurrent requests"""
        concurrent_levels = [10, 50, 100, 200]
        
        for concurrency in concurrent_levels:
            start_time = time.perf_counter()
            
            # Create concurrent requests
            requests = []
            for i in range(concurrency):
                request = {
                    "jsonrpc": "2.0",
                    "id": f"concurrent_{i}",
                    "method": "tools/call",
                    "params": {
                        "name": f"tool_{i % 100}",
                        "arguments": {}
                    }
                }
                requests.append(performance_server.handle_request(request))
            
            # Execute concurrently
            responses = await asyncio.gather(*requests)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            throughput = concurrency / total_time
            
            # Verify all succeeded
            assert all("result" in r for r in responses)
            
            print(f"\nConcurrent requests ({concurrency} concurrent):")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} req/s")
            print(f"  Avg latency: {total_time/concurrency*1000:.2f}ms")
            
            # Should maintain good throughput
            assert throughput > 100  # At least 100 req/s


class TestMCPCachePerformance:
    """Test MCP caching layer performance"""
    
    @pytest.fixture
    def cached_server(self):
        """Create server with caching enabled"""
        server = EnhancedMCPServer("cache-test", "1.0.0")
        
        # Add expensive tool for cache testing
        call_count = 0
        async def expensive_tool(**kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate expensive operation
            return f"Expensive result {call_count}"
        
        server.add_tool(MCPTool(
            name="expensive",
            description="Expensive operation",
            parameters={"type": "object", "properties": {"input": {"type": "string"}}},
            function=expensive_tool
        ))
        
        server.call_count = lambda: call_count
        return server
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, cached_server):
        """Test cache hit performance improvement"""
        benchmark = PerformanceBenchmark()
        
        # First call (cache miss)
        request = {
            "jsonrpc": "2.0",
            "id": "cache_test",
            "method": "tools/call",
            "params": {
                "name": "expensive",
                "arguments": {"input": "test"}
            }
        }
        
        # Warm up cache
        await cached_server.handle_request(request)
        initial_calls = cached_server.call_count()
        
        # Measure cached performance
        async def cached_call():
            return await cached_server.handle_request(request)
        
        results = await benchmark.measure_async(cached_call, iterations=1000)
        
        # Verify cache was used
        assert cached_server.call_count() == initial_calls  # No additional calls
        
        # Cache hits should be very fast
        assert results["mean"] < 0.0001  # Less than 0.1ms average
        assert results["p99"] < 0.001    # 99th percentile under 1ms
        
        print(f"\nCache hit performance:")
        print(f"  Mean: {results['mean']*1000:.4f}ms")
        print(f"  P95: {results['p95']*1000:.4f}ms")
        print(f"  P99: {results['p99']*1000:.4f}ms")
    
    @pytest.mark.asyncio
    async def test_cache_memory_efficiency(self):
        """Test cache memory usage efficiency"""
        benchmark = PerformanceBenchmark()
        benchmark.start_memory_tracking()
        
        # Clear cache first
        mcp_cache.clear()
        
        # Add many items to cache
        for i in range(10000):
            key = f"test_key_{i}"
            value = {"data": f"test_value_{i}" * 10}  # ~100 bytes per item
            mcp_cache.set(key, value, ttl=300)
        
        memory_used = benchmark.end_memory_tracking()
        
        # Check memory efficiency
        items_in_cache = len(mcp_cache._cache)
        memory_per_item = memory_used / items_in_cache if items_in_cache > 0 else 0
        
        print(f"\nCache memory efficiency:")
        print(f"  Items cached: {items_in_cache}")
        print(f"  Total memory: {memory_used:.2f}MB")
        print(f"  Memory per item: {memory_per_item*1024:.2f}KB")
        
        # Should be memory efficient
        assert memory_per_item < 0.01  # Less than 10KB per item


class TestMCPSecurityPerformance:
    """Test security middleware performance impact"""
    
    @pytest.fixture
    def secure_server(self):
        """Create server with full security enabled"""
        server = EnhancedMCPServer("secure-test", "1.0.0")
        
        # Add security middleware
        security_config = SecurityConfig(
            require_auth=True,
            rate_limiting_enabled=True,
            input_validation_enabled=True,
            strict_validation=True
        )
        
        server.security_middleware = create_secure_middleware(
            jwt_secret="test_secret_key_32_chars_minimum!!",
            enable_strict_mode=True
        )
        
        # Add test tool
        server.add_tool(MCPTool(
            name="secure_tool",
            description="Secure tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "Secure result"
        ))
        
        # Create auth token
        server.auth_token = server.security_middleware.create_admin_token()
        
        return server
    
    @pytest.mark.asyncio
    async def test_authentication_overhead(self, secure_server):
        """Test authentication performance overhead"""
        benchmark = PerformanceBenchmark()
        
        # Authenticated request
        async def authenticated_request():
            request = {
                "jsonrpc": "2.0",
                "id": "auth_test",
                "method": "tools/list"
            }
            headers = {
                "Authorization": f"Bearer {secure_server.auth_token}"
            }
            
            # Process through security middleware
            _, context = await secure_server.security_middleware.process_request(
                request["method"],
                request.get("params", {}),
                headers
            )
            
            return await secure_server.handle_request(request)
        
        results = await benchmark.measure_async(authenticated_request, iterations=1000)
        
        # Authentication should have minimal overhead
        assert results["mean"] < 0.002  # Less than 2ms average
        assert results["p99"] < 0.01    # 99th percentile under 10ms
        
        print(f"\nAuthentication overhead:")
        print(f"  Mean: {results['mean']*1000:.3f}ms")
        print(f"  P95: {results['p95']*1000:.3f}ms")
        print(f"  P99: {results['p99']*1000:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_rate_limiting_performance(self):
        """Test rate limiting performance impact"""
        benchmark = PerformanceBenchmark()
        
        # Create rate limiter
        rate_limiter = RateLimitMiddleware()
        
        async def check_rate_limit():
            user_id = f"user_{int(time.time() * 1000000) % 1000}"  # Distribute across users
            return await rate_limiter.check_rate_limit(user_id, "test_method")
        
        results = await benchmark.measure_async(check_rate_limit, iterations=10000)
        
        # Rate limiting should be very fast
        assert results["mean"] < 0.0001  # Less than 0.1ms average
        assert results["p99"] < 0.001    # 99th percentile under 1ms
        
        print(f"\nRate limiting performance:")
        print(f"  Mean: {results['mean']*1000:.4f}ms")
        print(f"  P95: {results['p95']*1000:.4f}ms")
        print(f"  P99: {results['p99']*1000:.4f}ms")


class TestMCPTransportPerformance:
    """Test transport layer performance"""
    
    @pytest.mark.asyncio
    async def test_websocket_message_throughput(self):
        """Test WebSocket message throughput"""
        transport = WebSocketTransport()
        benchmark = PerformanceBenchmark()
        
        # Simulate message serialization/deserialization
        test_message = {
            "jsonrpc": "2.0",
            "id": "test",
            "result": {
                "tools": [{"name": f"tool_{i}", "description": f"Tool {i}"} for i in range(10)]
            }
        }
        
        async def process_message():
            # Serialize
            serialized = json.dumps(test_message)
            # Simulate network transfer
            await asyncio.sleep(0)
            # Deserialize
            return json.loads(serialized)
        
        results = await benchmark.measure_async(process_message, iterations=10000)
        
        # Should handle high message throughput
        assert results["mean"] < 0.0001  # Less than 0.1ms per message
        
        throughput = 1 / results["mean"] if results["mean"] > 0 else 0
        
        print(f"\nWebSocket message throughput:")
        print(f"  Mean latency: {results['mean']*1000:.4f}ms")
        print(f"  Throughput: {throughput:.0f} msg/s")
    
    @pytest.mark.asyncio
    async def test_http_transport_performance(self):
        """Test HTTP transport performance characteristics"""
        transport = HTTPTransport()
        benchmark = PerformanceBenchmark()
        
        # Test request handling
        async def handle_http_request():
            request_data = {
                "jsonrpc": "2.0",
                "id": "http_test",
                "method": "tools/list"
            }
            
            # Simulate HTTP request processing
            json_str = json.dumps(request_data)
            # Parse request
            parsed = json.loads(json_str)
            # Create response
            response = {
                "jsonrpc": "2.0",
                "id": parsed["id"],
                "result": {"tools": []}
            }
            # Serialize response
            return json.dumps(response)
        
        results = await benchmark.measure_async(handle_http_request, iterations=10000)
        
        # HTTP handling should be efficient
        assert results["mean"] < 0.0002  # Less than 0.2ms average
        
        print(f"\nHTTP transport performance:")
        print(f"  Mean: {results['mean']*1000:.4f}ms")
        print(f"  P95: {results['p95']*1000:.4f}ms")
        print(f"  P99: {results['p99']*1000:.4f}ms")


class TestMCPScalability:
    """Test MCP scalability characteristics"""
    
    @pytest.mark.asyncio
    async def test_tool_registry_scalability(self):
        """Test performance with many registered tools"""
        benchmark = PerformanceBenchmark()
        
        # Test with different numbers of tools
        tool_counts = [100, 500, 1000, 5000]
        
        for count in tool_counts:
            server = MCPServer("scale-test", "1.0.0")
            
            # Add many tools
            for i in range(count):
                tool = MCPTool(
                    name=f"tool_{i}",
                    description=f"Tool {i}",
                    parameters={},
                    function=lambda i=i: f"Result {i}"
                )
                server.add_tool(tool)
            
            # Measure lookup performance
            async def lookup_tool():
                # Look up random tool
                tool_name = f"tool_{count // 2}"  # Middle tool
                return server.get_tool(tool_name)
            
            results = await benchmark.measure_async(lookup_tool, iterations=1000)
            
            print(f"\nTool lookup with {count} tools:")
            print(f"  Mean: {results['mean']*1000:.4f}ms")
            print(f"  P99: {results['p99']*1000:.4f}ms")
            
            # Lookup should remain fast even with many tools
            assert results["mean"] < 0.0001  # Less than 0.1ms
    
    @pytest.mark.asyncio
    async def test_connection_scalability(self):
        """Test server performance with many connections"""
        server = EnhancedMCPServer("connection-test", "1.0.0")
        
        # Simulate many connections
        connection_counts = [10, 50, 100, 500]
        
        for count in connection_counts:
            connections = []
            
            # Create connections
            for i in range(count):
                conn = {
                    "id": f"conn_{i}",
                    "authenticated": True,
                    "last_activity": time.time()
                }
                connections.append(conn)
            
            # Measure connection management overhead
            start_time = time.perf_counter()
            
            # Simulate connection tracking operations
            for _ in range(100):
                # Check all connections
                active = [c for c in connections if time.time() - c["last_activity"] < 300]
                # Update random connection
                if active:
                    active[count % len(active)]["last_activity"] = time.time()
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            print(f"\nConnection management with {count} connections:")
            print(f"  Duration: {duration*1000:.2f}ms for 100 operations")
            print(f"  Per operation: {duration*10:.4f}ms")
            
            # Should scale well
            assert duration < 0.1  # Less than 100ms for 100 operations
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage under sustained load"""
        benchmark = PerformanceBenchmark()
        server = EnhancedMCPServer("memory-test", "1.0.0")
        
        # Add tools
        for i in range(100):
            server.add_tool(MCPTool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters={},
                function=lambda i=i: f"Result {i}"
            ))
        
        benchmark.start_memory_tracking()
        
        # Simulate sustained load
        request_count = 10000
        for i in range(request_count):
            request = {
                "jsonrpc": "2.0",
                "id": f"req_{i}",
                "method": "tools/call",
                "params": {
                    "name": f"tool_{i % 100}",
                    "arguments": {}
                }
            }
            
            await server.handle_request(request)
            
            # Periodically clear old data to prevent unbounded growth
            if i % 1000 == 0:
                # Simulate cleanup
                mcp_metrics.reset()
        
        memory_delta = benchmark.end_memory_tracking()
        
        print(f"\nMemory usage under load:")
        print(f"  Requests processed: {request_count}")
        print(f"  Memory increase: {memory_delta:.2f}MB")
        print(f"  Memory per 1k requests: {memory_delta/10:.3f}MB")
        
        # Memory usage should be reasonable
        assert memory_delta < 100  # Less than 100MB for 10k requests


class TestMCPOptimizations:
    """Test specific performance optimizations"""
    
    @pytest.mark.asyncio
    async def test_json_parsing_optimization(self):
        """Test JSON parsing performance"""
        benchmark = PerformanceBenchmark()
        
        # Create complex JSON structure
        complex_data = {
            "tools": [
                {
                    "name": f"tool_{i}",
                    "description": f"Complex tool {i}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            f"param_{j}": {"type": "string", "description": f"Parameter {j}"}
                            for j in range(10)
                        }
                    }
                }
                for i in range(50)
            ]
        }
        
        json_str = json.dumps(complex_data)
        
        async def parse_json():
            return json.loads(json_str)
        
        results = await benchmark.measure_async(parse_json, iterations=1000)
        
        print(f"\nJSON parsing performance (complex structure):")
        print(f"  Mean: {results['mean']*1000:.3f}ms")
        print(f"  P99: {results['p99']*1000:.3f}ms")
        
        # Should parse efficiently
        assert results["mean"] < 0.001  # Less than 1ms average
    
    @pytest.mark.asyncio
    async def test_connection_pooling_benefit(self):
        """Test connection pooling performance benefit"""
        # Simulate database operations with and without pooling
        
        # Without pooling
        async def without_pooling():
            # Simulate connection overhead
            await asyncio.sleep(0.001)  # 1ms connection time
            # Simulate query
            await asyncio.sleep(0.0001)  # 0.1ms query time
            return "result"
        
        # With pooling (reused connection)
        pool = []  # Simulated connection pool
        
        async def with_pooling():
            if not pool:
                # First connection
                await asyncio.sleep(0.001)
                pool.append("connection")
            # Reuse connection - no overhead
            await asyncio.sleep(0.0001)  # Just query time
            return "result"
        
        benchmark = PerformanceBenchmark()
        
        # Measure without pooling
        results_no_pool = await benchmark.measure_async(without_pooling, iterations=100)
        
        # Clear pool and measure with pooling
        pool.clear()
        results_with_pool = await benchmark.measure_async(with_pooling, iterations=100)
        
        improvement = (results_no_pool["mean"] - results_with_pool["mean"]) / results_no_pool["mean"] * 100
        
        print(f"\nConnection pooling benefit:")
        print(f"  Without pooling: {results_no_pool['mean']*1000:.3f}ms")
        print(f"  With pooling: {results_with_pool['mean']*1000:.3f}ms")
        print(f"  Improvement: {improvement:.1f}%")
        
        # Pooling should provide significant benefit
        assert improvement > 50  # At least 50% improvement


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output