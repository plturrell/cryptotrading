"""
Vercel API route for ML model performance tracking
"""

import os
import sys
import json
from datetime import datetime
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cryptotrading.core.ml.model_server import ModelPerformanceTracker, ModelRegistry
from cryptotrading.core.storage.vercel_kv import get_cache_client


def handler(request):
    """
    Vercel serverless function handler for model performance
    GET /api/ml/performance/{symbol}?horizon=24h
    POST /api/ml/performance/track - Track actual vs predicted
    """
    try:
        if request.method == "GET":
            # Get performance metrics
            path_parts = request.path.strip("/").split("/")
            symbol = path_parts[-1] if len(path_parts) > 3 else request.args.get("symbol", "BTC")
            symbol = symbol.upper()

            # Initialize tracker
            tracker = ModelPerformanceTracker()

            # Get performance summary
            summary = asyncio.run(tracker.get_performance_summary(symbol))

            # Get model metadata from registry
            registry = ModelRegistry()

            try:
                # Get current model version info
                model_versions = asyncio.run(registry._get_best_version(symbol))

                if model_versions:
                    metadata_key = f"model_metadata:{symbol}:{model_versions}"
                    cache = get_cache_client()
                    metadata_str = asyncio.run(cache.kv.get(metadata_key))

                    if metadata_str:
                        metadata = json.loads(metadata_str)
                        summary["model_info"] = {
                            "version": model_versions,
                            "trained_at": metadata.get("trained_at"),
                            "metrics": metadata.get("metrics", {}),
                            "features": len(metadata.get("features", [])),
                            "status": metadata.get("status", "unknown"),
                        }
            except:
                pass

            # Add current timestamp
            summary["timestamp"] = datetime.now().isoformat()

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Cache-Control": "public, s-maxage=60",  # Cache for 1 minute
                },
                "body": json.dumps(summary),
            }

        elif request.method == "POST":
            # Track actual outcome
            body = json.loads(request.body)

            model_id = body.get("model_id", body.get("symbol", "BTC"))
            prediction_id = body.get("prediction_id")
            predicted_value = body.get("predicted_value")
            actual_value = body.get("actual_value")

            if not all([prediction_id, predicted_value is not None, actual_value is not None]):
                return {"statusCode": 400, "body": json.dumps({"error": "Missing required fields"})}

            # Track the outcome
            tracker = ModelPerformanceTracker()

            asyncio.run(
                tracker.track_actual_outcome(
                    model_id, prediction_id, float(predicted_value), float(actual_value)
                )
            )

            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {
                        "message": "Outcome tracked successfully",
                        "model_id": model_id,
                        "prediction_id": prediction_id,
                        "error": abs(float(actual_value) - float(predicted_value)),
                        "error_pct": abs(float(actual_value) - float(predicted_value))
                        / float(actual_value)
                        * 100
                        if actual_value != 0
                        else 0,
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
            }

        else:
            return {"statusCode": 405, "body": json.dumps({"error": "Method not allowed"})}

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "timestamp": datetime.now().isoformat()}),
        }


# For testing locally
if __name__ == "__main__":

    class MockRequest:
        def __init__(self, method="GET", path="/api/ml/performance/BTC"):
            self.method = method
            self.path = path
            self.args = {"symbol": "BTC"}
            self.body = None

    # Test GET
    result = handler(MockRequest())
    print("GET Performance:")
    print(json.dumps(result, indent=2))

    # Test POST
    post_request = MockRequest(method="POST")
    post_request.body = json.dumps(
        {
            "model_id": "BTC",
            "prediction_id": "test123",
            "predicted_value": 51000,
            "actual_value": 50500,
        }
    )

    result = handler(post_request)
    print("\nPOST Outcome:")
    print(json.dumps(result, indent=2))
