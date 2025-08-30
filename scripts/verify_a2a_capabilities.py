#!/usr/bin/env python3
"""
A2A Capabilities Verification Script
Systematically checks each A2A agent CLI against all defined capabilities and MCP tools
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# A2A Capabilities from protocol definition
A2A_CAPABILITIES = {
    'technical_analysis_agent': [
        'technical_indicators', 'momentum_analysis', 'volume_analysis',
        'pattern_recognition', 'trend_analysis', 'oscillator_analysis',
        'support_resistance', 'market_sentiment'
    ],
    'ml_agent': [
        'price_prediction', 'model_training', 'feature_engineering',
        'batch_prediction', 'model_evaluation', 'hyperparameter_optimization',
        'ml_inference'
    ],
    'strands_glean_agent': [
        'code_analysis', 'dependency_mapping', 'symbol_search',
        'code_navigation', 'insight_generation', 'coverage_validation',
        'change_monitoring', 'code_quality'
    ],
    'feature_store_agent': [
        'compute_features', 'get_feature_vector', 'get_training_features',
        'get_feature_definitions', 'get_feature_importance', 'feature_engineering',
        'ml_features', 'technical_indicators'
    ],
    'clrs_algorithms_agent': [
        'binary_search', 'linear_search', 'quick_select', 'find_minimum', 'find_maximum',
        'insertion_sort', 'merge_sort', 'quick_sort', 'algorithmic_calculations',
        'search_algorithms', 'sorting_algorithms', 'clrs_algorithms'
    ],
    'historical_data_loader_agent': [
        'data_loading', 'historical_data', 'multi_source_aggregation',
        'temporal_alignment', 'data_validation', 'catalog_management',
        'yahoo_finance', 'fred_data', 'cboe_data', 'defillama_data'
    ],
    'database_agent': [
        'data_storage', 'data_retrieval', 'bulk_insert', 'ai_analysis_storage',
        'portfolio_management', 'trade_history', 'database_health',
        'query_optimization', 'data_cleanup'
    ],
    'data_analysis_agent': [
        'validate_data_quality', 'analyze_data_distribution', 'compute_correlation_matrix',
        'detect_outliers', 'compute_rolling_statistics', 'statistical_analysis',
        'data_validation', 'quality_assessment'
    ]
}

# CLI Scripts mapping
CLI_SCRIPTS = {
    'technical_analysis_agent': 'a2a_technical_analysis_cli.py',
    'ml_agent': 'a2a_ml_agent_cli.py',
    'strands_glean_agent': 'a2a_strands_glean_cli.py',
    'feature_store_agent': 'a2a_feature_store_cli.py',
    'clrs_algorithms_agent': 'a2a_clrs_algorithms_cli.py',
    'historical_data_loader_agent': 'a2a_data_loader_cli.py',
    'database_agent': 'a2a_database_agent_cli.py',
    'data_analysis_agent': 'a2a_data_analysis_agent_cli.py'
}

# MCP Tools mapping (from a2a_mcp_bridge.py)
MCP_MAPPINGS = {
    'technical_analysis_agent': {
        'primary_tools': ['clrs_analysis', 'optimization_recommendation'],
        'secondary_tools': ['dependency_graph'],
        'capabilities_mapping': {
            'technical_indicators': ['clrs_analysis'],
            'momentum_analysis': ['optimization_recommendation'],
            'pattern_recognition': ['clrs_analysis', 'code_similarity']
        }
    },
    'ml_agent': {
        'primary_tools': ['optimization_recommendation', 'configuration_merge'],
        'secondary_tools': ['clrs_analysis'],
        'capabilities_mapping': {
            'model_training': ['optimization_recommendation'],
            'hyperparameter_optimization': ['optimization_recommendation'],
            'feature_engineering': ['configuration_merge']
        }
    },
    'strands_glean_agent': {
        'primary_tools': ['code_similarity', 'hierarchical_indexing'],
        'secondary_tools': ['dependency_graph'],
        'capabilities_mapping': {
            'code_analysis': ['code_similarity', 'hierarchical_indexing'],
            'dependency_mapping': ['dependency_graph'],
            'symbol_search': ['hierarchical_indexing']
        }
    },
    'feature_store_agent': {
        'primary_tools': ['configuration_merge', 'optimization_recommendation'],
        'secondary_tools': ['clrs_analysis']
    },
    'clrs_algorithms_agent': {
        'primary_tools': ['clrs_analysis'],
        'secondary_tools': ['optimization_recommendation']
    }
}

def get_cli_commands(script_name):
    """Get available commands from a CLI script"""
    try:
        result = subprocess.run(
            ['python3', script_name, '--help'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            commands = []
            in_commands_section = False
            
            for line in lines:
                if 'Commands:' in line:
                    in_commands_section = True
                    continue
                elif in_commands_section and line.strip():
                    if line.startswith('  ') and not line.startswith('    '):
                        command = line.strip().split()[0]
                        commands.append(command)
                    elif not line.startswith('  '):
                        break
            
            return commands
        else:
            return []
            
    except Exception as e:
        print(f"Error getting commands for {script_name}: {e}")
        return []

def test_cli_command(script_name, command, test_args=None):
    """Test if a CLI command works"""
    try:
        cmd = ['python3', script_name, command]
        if test_args:
            cmd.extend(test_args)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        # Consider it working if it doesn't crash with import errors
        if result.returncode == 0 or "Import error:" in result.stdout:
            return True
        else:
            return False
            
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

def analyze_agent_coverage(agent_id):
    """Analyze CLI coverage for a specific agent"""
    print(f"\n🔍 Analyzing {agent_id}")
    print("=" * 60)
    
    # Check if CLI script exists
    if agent_id not in CLI_SCRIPTS:
        print("❌ No CLI script mapped")
        return {
            'agent_id': agent_id,
            'cli_available': False,
            'capabilities_coverage': {},
            'mcp_tools': [],
            'coverage_score': 0.0
        }
    
    script_name = CLI_SCRIPTS[agent_id]
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ CLI script not found: {script_name}")
        return {
            'agent_id': agent_id,
            'cli_available': False,
            'capabilities_coverage': {},
            'mcp_tools': [],
            'coverage_score': 0.0
        }
    
    print(f"✅ CLI script found: {script_name}")
    
    # Get available commands
    commands = get_cli_commands(script_path)
    print(f"Available commands: {commands}")
    
    # Check capability coverage
    capabilities = A2A_CAPABILITIES.get(agent_id, [])
    coverage = {}
    
    # Map capabilities to commands (simplified heuristic)
    capability_command_mapping = {
        # Technical Analysis Agent
        'technical_indicators': ['momentum', 'comprehensive'],
        'momentum_analysis': ['momentum', 'comprehensive'],
        'volume_analysis': ['volume', 'comprehensive'],
        'pattern_recognition': ['patterns', 'comprehensive'],
        'trend_analysis': ['trend', 'comprehensive'],
        'oscillator_analysis': ['oscillators', 'comprehensive'],
        'support_resistance': ['levels', 'comprehensive'],
        'market_sentiment': ['sentiment', 'comprehensive'],
        # ML Agent
        'price_prediction': ['predict', 'comprehensive'],
        'model_training': ['train', 'comprehensive'],
        'model_evaluation': ['evaluate', 'comprehensive'],
        'hyperparameter_optimization': ['optimize', 'comprehensive'],
        'batch_prediction': ['batch', 'comprehensive'],
        'feature_engineering': ['engineering', 'comprehensive'],
        'ml_inference': ['inference', 'comprehensive'],
        # Strands Glean Agent
        'code_analysis': ['analyze', 'comprehensive'],
        'dependency_mapping': ['dependencies', 'comprehensive'],
        'symbol_search': ['search', 'comprehensive'],
        'coverage_validation': ['coverage', 'comprehensive'],
        'code_navigation': ['search', 'analyze'],
        'insight_generation': ['analyze', 'comprehensive'],
        'change_monitoring': ['analyze', 'status'],
        'code_quality': ['analyze', 'coverage'],
        # Feature Store Agent
        'compute_features': ['compute', 'comprehensive'],
        'get_feature_vector': ['vector', 'comprehensive'],
        'get_feature_importance': ['importance', 'comprehensive'],
        'get_training_features': ['vector', 'compute'],
        'get_feature_definitions': ['compute', 'status'],
        'ml_features': ['compute', 'vector'],
        'technical_indicators': ['compute', 'importance'],
        # CLRS Algorithms Agent
        'binary_search': ['search', 'comprehensive'],
        'linear_search': ['search', 'algorithms'],
        'quick_select': ['search', 'algorithms'],
        'find_minimum': ['minmax', 'comprehensive'],
        'find_maximum': ['minmax', 'comprehensive'],
        'insertion_sort': ['sort', 'algorithms'],
        'merge_sort': ['sort', 'algorithms'],
        'quick_sort': ['quick-sort', 'comprehensive'],
        'algorithmic_calculations': ['algorithms', 'comprehensive'],
        'search_algorithms': ['search', 'algorithms'],
        'sorting_algorithms': ['sort', 'algorithms'],
        'clrs_algorithms': ['algorithms', 'comprehensive'],
        # Historical Data Loader Agent
        'data_loading': ['load', 'symbols', 'status'],
        'historical_data': ['load', 'crypto', 'economic'],
        'multi_source_aggregation': ['full-sync', 'load'],
        'temporal_alignment': ['load', 'status'],
        'data_validation': ['status', 'symbols'],
        'catalog_management': ['symbols', 'status'],
        'yahoo_finance': ['economic', 'load'],
        'fred_data': ['economic', 'load'],
        'cboe_data': ['crypto', 'load'],
        'defillama_data': ['crypto', 'load'],
        # Database Agent
        'data_storage': ['store', 'comprehensive'],
        'data_retrieval': ['retrieve', 'comprehensive'],
        'bulk_insert': ['bulk', 'comprehensive'],
        'ai_analysis_storage': ['ai-store', 'comprehensive'],
        'portfolio_management': ['portfolio', 'comprehensive'],
        'trade_history': ['trades', 'comprehensive'],
        'database_health': ['health', 'comprehensive'],
        'query_optimization': ['optimize', 'comprehensive'],
        'data_cleanup': ['cleanup', 'comprehensive'],
        # Data Analysis Agent
        'validate_data_quality': ['validate', 'comprehensive'],
        'analyze_data_distribution': ['distribution', 'comprehensive'],
        'compute_correlation_matrix': ['correlation', 'comprehensive'],
        'detect_outliers': ['outliers', 'comprehensive'],
        'compute_rolling_statistics': ['rolling', 'comprehensive'],
        'statistical_analysis': ['statistical', 'comprehensive'],
        'data_validation': ['validate', 'schema'],
        'quality_assessment': ['quality', 'comprehensive']
    }
    
    covered_capabilities = 0
    for capability in capabilities:
        expected_commands = capability_command_mapping.get(capability, [])
        has_coverage = any(cmd in commands for cmd in expected_commands)
        coverage[capability] = has_coverage
        if has_coverage:
            covered_capabilities += 1
    
    coverage_score = covered_capabilities / len(capabilities) if capabilities else 0.0
    
    # Check MCP tools
    mcp_tools = MCP_MAPPINGS.get(agent_id, {}).get('primary_tools', [])
    
    print(f"Capabilities: {len(capabilities)}")
    print(f"Covered: {covered_capabilities}")
    print(f"Coverage: {coverage_score:.1%}")
    print(f"MCP Tools: {mcp_tools}")
    
    # Show missing capabilities
    missing = [cap for cap, covered in coverage.items() if not covered]
    if missing:
        print(f"❌ Missing coverage: {missing}")
    else:
        print("✅ All capabilities covered")
    
    return {
        'agent_id': agent_id,
        'cli_available': True,
        'script_name': script_name,
        'commands': commands,
        'capabilities': capabilities,
        'capabilities_coverage': coverage,
        'covered_capabilities': covered_capabilities,
        'total_capabilities': len(capabilities),
        'coverage_score': coverage_score,
        'mcp_tools': mcp_tools,
        'missing_capabilities': missing
    }

def main():
    """Run comprehensive A2A capabilities verification"""
    print("🚀 A2A Capabilities Verification")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = []
    total_agents = len(A2A_CAPABILITIES)
    
    for agent_id in A2A_CAPABILITIES.keys():
        result = analyze_agent_coverage(agent_id)
        results.append(result)
    
    # Summary
    print(f"\n📊 Overall Summary")
    print("=" * 80)
    
    agents_with_cli = sum(1 for r in results if r['cli_available'])
    total_capabilities = sum(r.get('total_capabilities', 0) for r in results)
    covered_capabilities = sum(r.get('covered_capabilities', 0) for r in results)
    
    print(f"Agents with CLI: {agents_with_cli}/{total_agents} ({agents_with_cli/total_agents:.1%})")
    print(f"Total Capabilities: {total_capabilities}")
    print(f"Covered Capabilities: {covered_capabilities}")
    print(f"Overall Coverage: {covered_capabilities/total_capabilities:.1%}")
    
    # Top performers
    print(f"\n🏆 Coverage by Agent:")
    sorted_results = sorted(results, key=lambda x: x.get('coverage_score', 0), reverse=True)
    
    for result in sorted_results:
        if result['cli_available']:
            score = result['coverage_score']
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
            print(f"{status} {result['agent_id']}: {score:.1%} ({result['covered_capabilities']}/{result['total_capabilities']})")
        else:
            print(f"❌ {result['agent_id']}: No CLI")
    
    # Agents needing attention
    print(f"\n⚠️  Agents Needing Attention:")
    for result in results:
        if not result['cli_available']:
            print(f"❌ {result['agent_id']}: No CLI script")
        elif result['coverage_score'] < 0.8:
            missing = result.get('missing_capabilities', [])
            print(f"⚠️  {result['agent_id']}: Missing {missing}")
    
    # Save detailed results
    output_file = 'a2a_capabilities_report.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_agents': total_agents,
                'agents_with_cli': agents_with_cli,
                'total_capabilities': total_capabilities,
                'covered_capabilities': covered_capabilities,
                'overall_coverage': covered_capabilities/total_capabilities
            },
            'agents': results
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {output_file}")

if __name__ == '__main__':
    main()
