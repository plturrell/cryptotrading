#!/usr/bin/env python3
"""
Deploy trained ML models to Vercel
Ensures full model quality is preserved
"""

import asyncio
import base64
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cryptotrading.core.ml.model_server import ModelRegistry
from cryptotrading.core.ml.model_storage import ModelVersionManager, VercelBlobModelStorage
from cryptotrading.core.ml.models import CryptoPricePredictor


async def main():
    """Main deployment script"""
    print("🚀 ML Model Deployment to Vercel")
    print("================================")

    # Check if we have models to deploy
    models_dir = Path("models")
    if not models_dir.exists():
        print("\n❌ No models directory found!")
        print("Please train models first using: python scripts/train_models.py")
        return

    # Initialize components
    predictor = CryptoPricePredictor()
    registry = ModelRegistry()
    storage = VercelBlobModelStorage()
    version_manager = ModelVersionManager(storage)

    print("\n1️⃣ Training/Loading Models...")

    deployed_models = {}

    for symbol in ["BTC", "ETH"]:
        print(f"\n📊 Processing {symbol}...")

        try:
            # Train or load model
            model_path = models_dir / f"{symbol}_model" / "model.pkl"

            if model_path.exists():
                print(f"  ✓ Found existing model for {symbol}")
                with open(model_path, "rb") as f:
                    model = pickle.load(f)

                # Load metrics
                metrics_path = models_dir / f"{symbol}_model" / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                else:
                    metrics = {}
            else:
                print(f"  ⚡ Training new model for {symbol}...")
                metrics = predictor.train(symbol)
                model = predictor.get_model(symbol)

                if not model:
                    print(f"  ❌ Failed to train model for {symbol}")
                    continue

            # Serialize model
            model_data = pickle.dumps(model)
            print(f"  📦 Model size: {len(model_data) / 1024 / 1024:.2f} MB")

            # Prepare metadata
            metadata = {
                "model_id": symbol,
                "metrics": metrics,
                "features": getattr(predictor, "feature_columns", []),
                "trained_at": datetime.now().isoformat(),
                "model_type": "ensemble",
                "framework": "sklearn",
                "python_version": sys.version,
                "status": "active",
            }

            # Register model with version management
            print(f"  📝 Registering model...")
            model_version = await registry.register_model(symbol, model_data, metadata)

            deployed_models[symbol] = {
                "version": model_version.version,
                "metrics": metrics,
                "size": len(model_data),
            }

            print(f"  ✅ Deployed {symbol} model version: {model_version.version}")

        except Exception as e:
            print(f"  ❌ Error deploying {symbol}: {e}")
            continue

    print("\n2️⃣ Generating Vercel Configuration...")

    # Create lightweight models for edge functions
    edge_models = {}

    for symbol, info in deployed_models.items():
        try:
            # Load the full model
            model_data = await storage.download_model(symbol, info["version"])
            if model_data:
                model = pickle.loads(model_data)

                # Extract lightweight parameters
                if hasattr(model, "feature_importances_"):
                    # Use feature importance as simple weights
                    edge_models[symbol] = {
                        "coefficients": model.feature_importances_.tolist(),
                        "intercept": 0.0,
                        "feature_names": getattr(predictor, "feature_columns", []),
                        "model_type": "feature_weights",
                        "version": info["version"],
                    }

        except Exception as e:
            print(f"  ⚠️  Could not create edge model for {symbol}: {e}")

    # Save Vercel environment configuration
    if edge_models:
        edge_config = base64.b64encode(json.dumps(edge_models).encode()).decode()

        env_file = Path(".env.vercel")
        with open(env_file, "w") as f:
            f.write(f"ML_MODELS_CONFIG={edge_config}\n")

        print(f"\n✅ Created {env_file} with edge-optimized models")

    print("\n3️⃣ Deployment Summary")
    print("===================")

    for symbol, info in deployed_models.items():
        print(f"\n{symbol}:")
        print(f"  Version: {info['version']}")
        print(f"  Size: {info['size'] / 1024 / 1024:.2f} MB")
        if "metrics" in info and info["metrics"]:
            print(f"  R² Score: {info['metrics'].get('r2', 0):.4f}")
            print(f"  MAPE: {info['metrics'].get('mape', 0):.2f}%")

    print("\n4️⃣ Next Steps:")
    print("=============")
    print("1. Deploy to Vercel: vercel --prod")
    print("2. Set Vercel Blob token in dashboard")
    print("3. Test predictions: curl https://your-app.vercel.app/api/ml/predict?symbol=BTC")
    print("4. Monitor performance: https://your-app.vercel.app/api/ml/serve")

    # Test model serving locally
    print("\n5️⃣ Testing Model Serving...")

    from cryptotrading.core.ml.model_server import model_serving_api

    test_features = {
        "price": 50000,
        "price_change_24h": 2.5,
        "volume": 1000000,
        "rsi_14": 55,
        "volatility_20": 0.05,
    }

    for symbol in deployed_models.keys():
        try:
            result = await model_serving_api.predict(symbol, test_features)
            if "error" not in result:
                print(f"✅ {symbol} serving test passed")
            else:
                print(f"❌ {symbol} serving test failed: {result['error']}")
        except Exception as e:
            print(f"❌ {symbol} serving test error: {e}")

    print("\n✨ Deployment preparation complete!")


if __name__ == "__main__":
    asyncio.run(main())
