#!/usr/bin/env python3
"""
Demonstration of A2A Data Communication Between Agents
Shows how agents exchange actual data through the CDS A2A service
"""
import json
import requests
from datetime import datetime

# CDS A2A Service endpoint
A2A_SERVICE_URL = "http://localhost:4004/api/odata/v4/A2AService"

# Agent IDs from the registered agents
AGENTS = {
    "technical_analysis": "5",  # Technical Analysis Agent
    "ml_agent": "2",            # ML Predictor Agent
    "feature_store": "7",       # Feature Store Agent
    "data_analysis": "1"        # Data Analysis Agent
}

def send_a2a_message(from_agent, to_agent, message_type, data_payload):
    """Send a message with data payload through A2A"""
    
    # Prepare the message with actual data
    message = {
        "fromAgentId": AGENTS[from_agent],
        "toAgentId": AGENTS[to_agent],
        "messageType": message_type,
        "payload": json.dumps(data_payload),  # Data is sent as JSON string
        "priority": "high"
    }
    
    # Send via CDS A2A Service
    response = requests.post(
        f"{A2A_SERVICE_URL}/sendMessage",
        json=message,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()

# EXAMPLE 1: Technical Analysis sends indicators to ML Agent
print("=" * 60)
print("EXAMPLE 1: Technical Analysis → ML Agent")
print("=" * 60)

technical_indicators = {
    "symbol": "BTCUSDT",
    "timestamp": datetime.now().isoformat(),
    "indicators": {
        "rsi": 68.5,
        "macd": {
            "value": 125.3,
            "signal": 110.2,
            "histogram": 15.1
        },
        "bollinger_bands": {
            "upper": 52000,
            "middle": 50000,
            "lower": 48000
        },
        "volume": 1234567,
        "support_levels": [48500, 47000, 45000],
        "resistance_levels": [51500, 53000, 55000]
    },
    "patterns_detected": ["ascending_triangle", "bullish_divergence"],
    "signal": "BUY",
    "confidence": 0.78
}

result = send_a2a_message(
    "technical_analysis", 
    "ml_agent",
    "technical_indicators_update",
    technical_indicators
)
print(f"Message sent: {result}")
print(f"Data payload size: {len(json.dumps(technical_indicators))} bytes")
print()

# EXAMPLE 2: ML Agent sends predictions to Feature Store
print("=" * 60)
print("EXAMPLE 2: ML Agent → Feature Store")
print("=" * 60)

ml_predictions = {
    "model_id": "ensemble_btc_20250830",
    "timestamp": datetime.now().isoformat(),
    "predictions": [
        {"horizon": "1h", "price": 50500, "confidence": 0.82, "direction": "up"},
        {"horizon": "4h", "price": 51200, "confidence": 0.75, "direction": "up"},
        {"horizon": "24h", "price": 52800, "confidence": 0.65, "direction": "up"}
    ],
    "features_used": [
        "price_ma_20", "volume_ratio", "rsi", "macd", 
        "sentiment_score", "market_cap_ratio"
    ],
    "model_metrics": {
        "training_accuracy": 0.87,
        "validation_accuracy": 0.83,
        "last_retrain": "2025-08-30T10:00:00Z"
    }
}

result = send_a2a_message(
    "ml_agent",
    "feature_store",
    "store_predictions",
    ml_predictions
)
print(f"Message sent: {result}")
print(f"Data payload size: {len(json.dumps(ml_predictions))} bytes")
print()

# EXAMPLE 3: Feature Store sends computed features to Data Analysis
print("=" * 60)
print("EXAMPLE 3: Feature Store → Data Analysis")
print("=" * 60)

computed_features = {
    "feature_set_id": "crypto_features_v2",
    "timestamp": datetime.now().isoformat(),
    "features": {
        "price_features": {
            "returns_1h": 0.015,
            "returns_24h": 0.032,
            "volatility_20d": 0.28,
            "sharpe_ratio": 1.45
        },
        "market_features": {
            "market_cap_dominance": 0.48,
            "volume_trend": "increasing",
            "correlation_with_btc": 0.92
        },
        "technical_features": {
            "trend_strength": 0.73,
            "momentum_score": 8.2,
            "mean_reversion_signal": -0.15
        }
    },
    "data_quality": {
        "completeness": 0.98,
        "freshness_seconds": 5,
        "outliers_detected": 2
    }
}

result = send_a2a_message(
    "feature_store",
    "data_analysis",
    "feature_update",
    computed_features
)
print(f"Message sent: {result}")
print(f"Data payload size: {len(json.dumps(computed_features))} bytes")
print()

# EXAMPLE 4: Multi-Agent Workflow Coordination
print("=" * 60)
print("EXAMPLE 4: Complete Multi-Agent Workflow")
print("=" * 60)

workflow_request = {
    "workflow_id": "price_prediction_workflow",
    "initiated_by": "user_request",
    "timestamp": datetime.now().isoformat(),
    "steps": [
        {
            "step": 1,
            "agent": "data_analysis",
            "action": "validate_data",
            "input": {"symbol": "BTCUSDT", "period": "24h"}
        },
        {
            "step": 2,
            "agent": "technical_analysis",
            "action": "compute_indicators",
            "depends_on": 1
        },
        {
            "step": 3,
            "agent": "feature_store",
            "action": "engineer_features",
            "depends_on": [1, 2]
        },
        {
            "step": 4,
            "agent": "ml_agent",
            "action": "predict_price",
            "depends_on": 3
        }
    ],
    "expected_output": "price_prediction_with_confidence"
}

# Data Analysis starts the workflow
result = send_a2a_message(
    "data_analysis",
    "technical_analysis",
    "workflow_initiate",
    workflow_request
)
print(f"Workflow initiated: {result}")
print(f"Workflow payload size: {len(json.dumps(workflow_request))} bytes")

print("\n" + "=" * 60)
print("A2A DATA COMMUNICATION SUMMARY")
print("=" * 60)
print("✅ Agents exchange structured JSON data through payloads")
print("✅ Data includes: indicators, predictions, features, metrics")
print("✅ Messages are queued and delivered asynchronously")
print("✅ Each agent can process received data independently")
print("✅ Supports complex multi-agent workflows")
