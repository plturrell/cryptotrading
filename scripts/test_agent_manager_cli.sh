#!/bin/bash
# Test script for Agent Manager CLI
# Demonstrates proper A2A agent registration through MCP tools

echo "=========================================="
echo "Agent Manager CLI Test"
echo "=========================================="
echo ""

# Set Python path
export PYTHONPATH=/Users/apple/projects/cryptotrading:$PYTHONPATH

# CLI path
CLI="python3 src/cryptotrading/core/agents/cli/agent_manager_cli.py"

echo "1. Registering Trading Algorithm Agent..."
echo "-----------------------------------------"
$CLI register trading_algorithm_agent trading_algorithm \
    -c grid_trading \
    -c dollar_cost_averaging \
    -c arbitrage_detection \
    -c momentum_trading \
    -c mean_reversion \
    -c signal_generation \
    -c strategy_analysis \
    -c backtesting \
    -t grid_create \
    -t grid_rebalance \
    -t dca_execute \
    -t arbitrage_scan \
    -t risk_calculate

echo ""
echo "2. Registering Data Analysis Agent..."
echo "--------------------------------------"
$CLI register data_analysis_agent data_analysis \
    -c data_processing \
    -c statistical_analysis \
    -c pattern_recognition \
    -t analyze_data \
    -t generate_report

echo ""
echo "3. Registering Feature Store Agent..."
echo "--------------------------------------"
$CLI register feature_store_agent feature_store \
    -c feature_storage \
    -c feature_retrieval \
    -c feature_versioning \
    -t store_feature \
    -t retrieve_feature

echo ""
echo "4. Listing all registered agents..."
echo "------------------------------------"
$CLI list

echo ""
echo "5. Checking compliance audit..."
echo "--------------------------------"
$CLI audit

echo ""
echo "6. Monitoring agent health..."
echo "------------------------------"
$CLI health

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="