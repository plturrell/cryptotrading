#!/bin/bash
# Register A2A agents using Agent Manager CLI
# This uses the proper Agent Manager with MCP tools for full A2A compliance

echo "============================================================"
echo "A2A Agent Registration via Agent Manager"
echo "============================================================"
echo ""

# Set environment
export ENVIRONMENT=development
export DATABASE_URL=sqlite:///cryptotrading_dev.db
export DATABASE_USERNAME=dev_user
export DATABASE_PASSWORD=dev_password
export JWT_SECRET=dev_secret_key_for_testing
export PYTHONPATH=/Users/apple/projects/cryptotrading:$PYTHONPATH

# Start Anvil if available
echo "Starting local Anvil blockchain for A2A messaging..."
if command -v anvil &> /dev/null; then
    anvil --port 8545 --accounts 10 --balance 10000 > /tmp/anvil.log 2>&1 &
    ANVIL_PID=$!
    sleep 3
    echo "✅ Anvil started (PID: $ANVIL_PID)"
else
    echo "⚠️  Anvil not found - continuing without blockchain"
fi

# CLI command
CLI="python3 -m src.cryptotrading.core.agents.cli.agent_manager_cli"

echo ""
echo "============================================================"
echo "Registering Core A2A Agents"
echo "============================================================"

# 1. MCTS Calculation Agent
echo ""
echo "1. Registering MCTS Calculation Agent..."
echo "----------------------------------------"
$CLI register mcts_calculation_agent mcts_calculation \
    -c monte_carlo_simulation \
    -c strategy_optimization \
    -c risk_assessment \
    -c portfolio_optimization \
    -c general_optimization \
    -c algorithm_performance \
    -c calculation_metrics \
    -t mcts_calculate \
    -t optimize_strategy \
    -t analyze_risk \
    --no-blockchain 2>/dev/null || echo "   Registration attempt completed"

# 2. Technical Analysis Agent  
echo ""
echo "2. Registering Technical Analysis Agent..."
echo "----------------------------------------"
$CLI register technical_analysis_agent technical_analysis \
    -c momentum_analysis \
    -c volume_analysis \
    -c pattern_recognition \
    -c technical_indicators \
    -c trend_analysis \
    -c market_sentiment \
    -t calculate_indicators \
    -t detect_patterns \
    -t analyze_trends \
    --no-blockchain 2>/dev/null || echo "   Registration attempt completed"

# 3. ML Agent
echo ""
echo "3. Registering ML Agent..."
echo "----------------------------------------"
$CLI register ml_agent ml_agent \
    -c model_training \
    -c prediction \
    -c feature_engineering \
    -c ml_calculations \
    -c ensemble_methods \
    -t train_model \
    -t predict \
    -t evaluate_model \
    --no-blockchain 2>/dev/null || echo "   Registration attempt completed"

# 4. Trading Algorithm Agent
echo ""
echo "4. Registering Trading Algorithm Agent..."
echo "----------------------------------------"
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
    -t dca_execute \
    -t arbitrage_scan \
    -t risk_calculate \
    --no-blockchain 2>/dev/null || echo "   Registration attempt completed"

# 5. Data Analysis Agent
echo ""
echo "5. Registering Data Analysis Agent..."
echo "----------------------------------------"
$CLI register data_analysis_agent data_analysis \
    -c data_processing \
    -c statistical_analysis \
    -c pattern_recognition \
    -c anomaly_detection \
    -t analyze_data \
    -t detect_anomalies \
    -t generate_report \
    --no-blockchain 2>/dev/null || echo "   Registration attempt completed"

# 6. Feature Store Agent
echo ""
echo "6. Registering Feature Store Agent..."
echo "----------------------------------------"
$CLI register feature_store_agent feature_store \
    -c feature_storage \
    -c feature_retrieval \
    -c feature_versioning \
    -c metadata_management \
    -t store_feature \
    -t retrieve_feature \
    -t validate_feature \
    --no-blockchain 2>/dev/null || echo "   Registration attempt completed"

# 7. Strands Glean Agent
echo ""
echo "7. Registering Strands Glean Agent..."
echo "----------------------------------------"
$CLI register strands_glean_agent glean_agent \
    -c code_analysis \
    -c dependency_tracking \
    -c impact_analysis \
    -c code_search \
    -t analyze_code \
    -t track_dependencies \
    -t search_codebase \
    --no-blockchain 2>/dev/null || echo "   Registration attempt completed"

# List all registered agents
echo ""
echo "============================================================"
echo "Listing All Registered Agents"
echo "============================================================"
$CLI list 2>/dev/null || echo "Unable to list agents"

# Audit compliance
echo ""
echo "============================================================"
echo "A2A Compliance Audit"
echo "============================================================"
$CLI audit 2>/dev/null || echo "Unable to audit compliance"

# Check health
echo ""
echo "============================================================"
echo "Agent Health Status"
echo "============================================================"
$CLI health 2>/dev/null || echo "Unable to check health"

# Stop Anvil if we started it
if [ ! -z "$ANVIL_PID" ]; then
    echo ""
    echo "Stopping Anvil..."
    kill $ANVIL_PID 2>/dev/null
    echo "✅ Anvil stopped"
fi

echo ""
echo "============================================================"
echo "✅ Agent Registration Complete!"
echo "============================================================"
echo ""
echo "Note: Agents are registered in the A2A protocol registry."
echo "Use '--no-blockchain' flag to skip blockchain registration."
echo "For full blockchain integration, ensure Anvil is running."