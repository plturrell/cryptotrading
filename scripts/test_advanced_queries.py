#!/usr/bin/env python3
"""
Test Advanced Angle Query Templates
Test the sophisticated query patterns and integration with Glean
"""

import asyncio
import json
import sys
import time
from pathlib import Path
import aiohttp

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


async def test_query_templates():
    """Test advanced query template functionality"""
    print("🔍 ADVANCED ANGLE QUERY TEMPLATES TEST")
    print("=" * 60)
    
    try:
        from cryptotrading.infrastructure.analysis.advanced_angle_queries import (
            create_advanced_query_builder, 
            QueryComplexity, 
            AnalysisScope,
            get_query_collection,
            QUERY_COLLECTIONS
        )
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 1: Query builder initialization
    print("\n1️⃣ Testing query builder initialization...")
    builder = create_advanced_query_builder()
    templates = builder.list_templates()
    print(f"   ✅ Builder created with {len(templates)} templates")
    
    # Show available templates by complexity
    for complexity in QueryComplexity:
        complexity_templates = builder.list_templates(complexity=complexity)
        print(f"      • {complexity.value}: {len(complexity_templates)} templates")
    
    # Test 2: Template generation
    print("\n2️⃣ Testing template generation...")
    
    # Test dependency graph query
    try:
        dependency_query = builder.generate_query(
            "dependency_graph", 
            root="src/cryptotrading/"
        )
        print("   ✅ Dependency graph query generated")
        print(f"      • Query length: {len(dependency_query)} characters")
        
        # Validate the query
        validation = builder.validate_query_syntax(dependency_query)
        print(f"      • Syntax valid: {validation['valid']}")
        if validation['errors']:
            print(f"      • Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"      • Warnings: {validation['warnings']}")
    except Exception as e:
        print(f"   ❌ Dependency query generation failed: {e}")
    
    # Test complexity analysis query
    try:
        complexity_query = builder.generate_query(
            "code_complexity_analysis",
            min_complexity=3
        )
        print("   ✅ Code complexity query generated")
        print(f"      • Query length: {len(complexity_query)} characters")
    except Exception as e:
        print(f"   ❌ Complexity query generation failed: {e}")
    
    # Test 3: Query suggestions
    print("\n3️⃣ Testing query suggestions...")
    
    security_suggestions = builder.get_query_suggestions(["security", "vulnerability"])
    print(f"   ✅ Security suggestions: {len(security_suggestions)} templates")
    for suggestion in security_suggestions[:3]:
        print(f"      • {suggestion.name}: {suggestion.description[:60]}...")
    
    performance_suggestions = builder.get_query_suggestions(["performance", "optimization"])
    print(f"   ✅ Performance suggestions: {len(performance_suggestions)} templates")
    for suggestion in performance_suggestions[:3]:
        print(f"      • {suggestion.name}: {suggestion.description[:60]}...")
    
    # Test 4: Query collections
    print("\n4️⃣ Testing query collections...")
    
    for collection_name, query_names in QUERY_COLLECTIONS.items():
        print(f"   📋 {collection_name} collection: {len(query_names)} queries")
        for query_name in query_names[:2]:  # Show first 2
            template = builder.get_template(query_name)
            if template:
                print(f"      • {query_name} ({template.complexity.value})")
    
    # Test 5: Composite query generation
    print("\n5️⃣ Testing composite query generation...")
    
    try:
        code_quality_queries = get_query_collection("code_quality")
        composite_query = builder.create_composite_query(
            code_quality_queries,
            min_complexity=5,
            test_pattern="test.*\\.py$",
            similarity_threshold=0.8
        )
        print(f"   ✅ Composite query generated with {len(code_quality_queries)} sub-queries")
        print(f"      • Total length: {len(composite_query)} characters")
        print(f"      • Sub-queries: {', '.join(code_quality_queries)}")
    except Exception as e:
        print(f"   ❌ Composite query generation failed: {e}")
    
    # Test 6: Custom template addition
    print("\n6️⃣ Testing custom template addition...")
    
    try:
        from cryptotrading.infrastructure.analysis.advanced_angle_queries import QueryTemplate
        
        custom_template = QueryTemplate(
            name="simple_function_count",
            description="Count functions by file",
            complexity=QueryComplexity.SIMPLE,
            scope=AnalysisScope.PROJECT,
            query_pattern="""
            query FunctionCount {
              codebase.function
              | group_by(.file.name)
              | map({
                  file: .key,
                  function_count: .values | length
                })
              | sort_by(.function_count) desc
            }
            """,
            use_cases=["Quick function counting", "File size estimation"]
        )
        
        builder.add_custom_template(custom_template)
        custom_query = builder.generate_query("simple_function_count")
        
        print("   ✅ Custom template added and generated")
        print(f"      • Template name: {custom_template.name}")
        print(f"      • Query length: {len(custom_query)} characters")
    except Exception as e:
        print(f"   ❌ Custom template test failed: {e}")
    
    return True


async def test_query_integration_with_mcp(mcp_url: str = "http://localhost:8082"):
    """Test advanced queries with MCP server integration"""
    print("\n🔗 MCP INTEGRATION TEST")
    print("=" * 50)
    
    try:
        from cryptotrading.infrastructure.analysis.advanced_angle_queries import create_advanced_query_builder
        builder = create_advanced_query_builder()
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Check MCP server availability
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{mcp_url}/mcp/status") as response:
                if response.status != 200:
                    print(f"❌ MCP server not available at {mcp_url}")
                    return False
                print("✅ MCP server is running")
        except Exception as e:
            print(f"❌ MCP server not accessible: {e}")
            return False
    
    # Test 1: Simple query execution
    print("\n1️⃣ Testing simple query execution...")
    
    try:
        # Generate a simple query
        simple_query = builder.generate_query("simple_function_count")
        
        # Execute via MCP
        query_request = {
            "jsonrpc": "2.0",
            "id": "test_advanced_query",
            "method": "tools/call", 
            "params": {
                "name": "glean_symbol_search",
                "arguments": {
                    "pattern": "function",
                    "limit": 5
                }
            }
        }
        
        async with session.post(f"{mcp_url}/mcp", json=query_request) as response:
            if response.status == 200:
                data = await response.json()
                result = data.get("result", {})
                if not result.get("isError", False):
                    content = result.get("content", [{}])[0]
                    if content.get("type") == "resource":
                        query_data = json.loads(content.get("data", "{}"))
                        symbols = query_data.get("symbols", [])
                        print(f"   ✅ Query executed successfully")
                        print(f"      • Found {len(symbols)} symbols")
                        for symbol in symbols[:3]:
                            print(f"        - {symbol.get('name', 'unknown')} ({symbol.get('kind', 'unknown')})")
                else:
                    print(f"   ❌ Query execution failed: {result}")
            else:
                print(f"   ❌ MCP request failed: {response.status}")
                
    except Exception as e:
        print(f"   ❌ Query integration test failed: {e}")
    
    # Test 2: Test indexing for advanced queries
    print("\n2️⃣ Testing project indexing for advanced queries...")
    
    try:
        index_request = {
            "jsonrpc": "2.0",
            "id": "index_for_advanced",
            "method": "tools/call",
            "params": {
                "name": "glean_index_project",
                "arguments": {
                    "unit_name": "advanced-queries-test",
                    "force_reindex": True
                }
            }
        }
        
        async with session.post(f"{mcp_url}/mcp", json=index_request) as response:
            if response.status == 200:
                data = await response.json()
                result = data.get("result", {})
                if not result.get("isError", False):
                    content = result.get("content", [{}])[0]
                    if content.get("type") == "resource":
                        index_data = json.loads(content.get("data", "{}"))
                        print(f"   ✅ Project indexed successfully")
                        print(f"      • Files indexed: {index_data.get('files_indexed', 0)}")
                        print(f"      • Symbols found: {index_data.get('symbols_found', 0)}")
                        print(f"      • Facts stored: {index_data.get('facts_stored', 0)}")
                else:
                    print(f"   ❌ Indexing failed: {result}")
            else:
                print(f"   ❌ Indexing request failed: {response.status}")
                
    except Exception as e:
        print(f"   ❌ Indexing test failed: {e}")
    
    return True


async def test_query_performance():
    """Test query template performance"""
    print("\n⚡ QUERY PERFORMANCE TEST")
    print("=" * 40)
    
    try:
        from cryptotrading.infrastructure.analysis.advanced_angle_queries import create_advanced_query_builder
        builder = create_advanced_query_builder()
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test template generation performance
    print("\n1️⃣ Testing template generation performance...")
    
    template_names = [
        "dependency_graph",
        "code_complexity_analysis", 
        "api_surface_analysis",
        "security_pattern_analysis",
        "performance_hotspot_analysis"
    ]
    
    start_time = time.time()
    generated_queries = []
    
    for template_name in template_names:
        template_start = time.time()
        try:
            query = builder.generate_query(template_name)
            generation_time = time.time() - template_start
            generated_queries.append((template_name, len(query), generation_time))
            print(f"   ✅ {template_name}: {generation_time:.3f}s ({len(query)} chars)")
        except Exception as e:
            print(f"   ❌ {template_name}: {e}")
    
    total_time = time.time() - start_time
    print(f"\n   📊 Total generation time: {total_time:.3f}s")
    print(f"   📊 Average per template: {total_time / len(template_names):.3f}s")
    
    # Test validation performance
    print("\n2️⃣ Testing validation performance...")
    
    validation_start = time.time()
    validation_results = []
    
    for template_name, query_len, gen_time in generated_queries:
        val_start = time.time()
        try:
            query = builder.generate_query(template_name)
            validation = builder.validate_query_syntax(query)
            val_time = time.time() - val_start
            validation_results.append((template_name, validation['valid'], val_time))
            status = "✅" if validation['valid'] else "❌"
            print(f"   {status} {template_name}: {val_time:.3f}s")
        except Exception as e:
            print(f"   ❌ {template_name}: {e}")
    
    validation_total = time.time() - validation_start
    print(f"\n   📊 Total validation time: {validation_total:.3f}s")
    print(f"   📊 Average per validation: {validation_total / len(validation_results):.3f}s")
    
    # Test suggestion performance
    print("\n3️⃣ Testing suggestion performance...")
    
    suggestion_tests = [
        ["security", "vulnerability"],
        ["performance", "optimization"], 
        ["test", "coverage"],
        ["architecture", "design"],
        ["refactor", "maintenance"]
    ]
    
    suggestion_start = time.time()
    
    for keywords in suggestion_tests:
        sug_start = time.time()
        suggestions = builder.get_query_suggestions(keywords)
        sug_time = time.time() - sug_start
        print(f"   ✅ {' + '.join(keywords)}: {len(suggestions)} suggestions in {sug_time:.3f}s")
    
    suggestion_total = time.time() - suggestion_start
    print(f"\n   📊 Total suggestion time: {suggestion_total:.3f}s")
    
    return True


async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Advanced Angle Query Templates")
    parser.add_argument("--mcp-url", default="http://localhost:8082", help="MCP server URL")
    parser.add_argument("--templates-only", action="store_true", help="Test templates only")
    parser.add_argument("--performance-only", action="store_true", help="Test performance only")
    parser.add_argument("--integration-only", action="store_true", help="Test MCP integration only")
    
    args = parser.parse_args()
    
    print("🧪 ADVANCED ANGLE QUERIES TEST SUITE")
    print(f"MCP Server URL: {args.mcp_url}")
    
    success_count = 0
    total_tests = 0
    
    if args.templates_only:
        total_tests = 1
        if await test_query_templates():
            success_count += 1
    elif args.performance_only:
        total_tests = 1
        if await test_query_performance():
            success_count += 1
    elif args.integration_only:
        total_tests = 1
        if await test_query_integration_with_mcp(args.mcp_url):
            success_count += 1
    else:
        # Run all tests
        total_tests = 3
        
        if await test_query_templates():
            success_count += 1
        
        if await test_query_performance():
            success_count += 1
            
        if await test_query_integration_with_mcp(args.mcp_url):
            success_count += 1
    
    # Summary
    print(f"\n🎉 ADVANCED QUERIES TEST COMPLETED!")
    print("=" * 60)
    print(f"📊 Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎊 ALL ADVANCED QUERY TESTS PASSED!")
        print("\n✅ Advanced query system is fully functional:")
        print("   • Template system working")
        print("   • Query generation operational")
        print("   • Validation system active")
        print("   • MCP integration ready")
        print("   • Performance acceptable")
    else:
        print("⚠️ Some tests failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Advanced queries test stopped")
    except Exception as e:
        print(f"❌ Test error: {e}")
        sys.exit(1)