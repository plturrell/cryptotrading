#!/usr/bin/env python3
"""
Export trained ML models for Vercel deployment
Converts heavy sklearn models to lightweight coefficient-based models
"""

import base64
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from cryptotrading.core.ml.models import CryptoPricePredictor
    from cryptotrading.core.ml.training import training_pipeline
    from cryptotrading.core.ml.vercel_deployment import export_models_for_vercel
    from cryptotrading.data.historical.yahoo_finance import YahooFinanceClient
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to install dependencies: pip install -r requirements.txt")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_sample_models():
    """Train sample models for demonstration"""
    logger.info("Training sample models for Vercel export...")

    yahoo_client = YahooFinanceClient()
    symbols = ["BTC", "ETH"]  # Just train a few for demo

    models_trained = []

    for symbol in symbols:
        try:
            logger.info(f"Training model for {symbol}...")

            # Get historical data
            from datetime import datetime, timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # 3 months of data

            data = yahoo_client.get_historical_data(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                interval="1h",
            )

            if data is None or len(data) < 100:
                logger.warning(f"Insufficient data for {symbol}, skipping...")
                continue

            # Train ensemble model
            model = CryptoPricePredictor(model_type="ensemble")
            metrics = model.train(data, target_hours=24)

            logger.info(
                f"Model trained for {symbol} - R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%"
            )

            models_trained.append({"symbol": symbol, "model": model, "metrics": metrics})

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")

    return models_trained


def create_vercel_model_config(models_trained):
    """Create lightweight model configuration for Vercel"""
    vercel_config = {}

    for model_info in models_trained:
        symbol = model_info["symbol"]
        model = model_info["model"]
        metrics = model_info["metrics"]

        try:
            # Get the random forest model (most interpretable)
            rf_model = model.models.get("rf")
            if rf_model is None:
                logger.warning(f"No RF model found for {symbol}, skipping...")
                continue

            # Extract feature importance as coefficients
            feature_names = model.metadata.get("features", [])
            feature_importance = model.feature_importance

            if not feature_importance:
                logger.warning(f"No feature importance for {symbol}, using dummy values...")
                feature_importance = {f"feature_{i}": 0.1 for i in range(10)}

            # Create lightweight model config
            # Use top 10 most important features only
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]

            coefficients = [importance for _, importance in sorted_features]
            feature_names = [name for name, _ in sorted_features]

            vercel_config[symbol] = {
                "coefficients": coefficients,
                "intercept": 0.0,  # Simple linear model
                "feature_names": feature_names,
                "scaler_params": {
                    "mean": [0.0] * len(coefficients),  # Normalized features
                    "scale": [1.0] * len(coefficients),
                },
                "metadata": {
                    "model_type": "linear_approximation",
                    "source_model": "random_forest",
                    "training_r2": metrics["r2"],
                    "training_mape": metrics["mape"],
                    "created_at": datetime.now().isoformat(),
                    "features_count": len(feature_names),
                },
            }

            logger.info(f"Created Vercel config for {symbol} with {len(feature_names)} features")

        except Exception as e:
            logger.error(f"Error creating Vercel config for {symbol}: {e}")

    return vercel_config


def export_for_vercel(vercel_config):
    """Export configuration for Vercel deployment"""

    # Save to local file
    output_dir = Path(__file__).parent.parent / "models" / "vercel"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_file = output_dir / "models_config.json"
    with open(config_file, "w") as f:
        json.dump(vercel_config, f, indent=2)

    logger.info(f"Saved Vercel config to {config_file}")

    # Create environment variable format
    config_str = json.dumps(vercel_config)
    config_base64 = base64.b64encode(config_str.encode()).decode()

    env_file = output_dir / "vercel.env"
    with open(env_file, "w") as f:
        f.write(f"ML_MODELS_CONFIG={config_base64}\n")

    logger.info(f"Created Vercel environment file: {env_file}")

    # Create deployment script
    deploy_script = output_dir / "deploy.sh"
    with open(deploy_script, "w") as f:
        f.write(
            f"""#!/bin/bash
# Deploy ML models to Vercel

echo "Setting up ML models for Vercel deployment..."

# Set environment variable
vercel env add ML_MODELS_CONFIG '{config_base64}' production

echo "ML models configuration uploaded to Vercel"
echo "Models available: {', '.join(vercel_config.keys())}"
echo "Deploy with: vercel --prod"
"""
        )

    deploy_script.chmod(0o755)
    logger.info(f"Created deployment script: {deploy_script}")

    return config_file, env_file, deploy_script


def main():
    """Main export process"""
    logger.info("Starting ML model export for Vercel deployment...")

    # Check if we have existing models or need to train
    models_dir = Path(__file__).parent.parent / "models"

    print("\n" + "=" * 60)
    print("ML MODEL EXPORT FOR VERCEL DEPLOYMENT")
    print("=" * 60)

    choice = input(
        "Do you want to:\n1. Train new models\n2. Use existing models (if available)\n\nEnter choice (1 or 2): "
    ).strip()

    if choice == "1":
        print("\nTraining new models...")
        models_trained = train_sample_models()

        if not models_trained:
            print("No models were successfully trained. Exiting.")
            return

    elif choice == "2":
        print("Using existing models is not implemented yet.")
        print("Please choose option 1 to train new models.")
        return
    else:
        print("Invalid choice. Exiting.")
        return

    # Create Vercel configuration
    print("\nCreating Vercel-compatible model configuration...")
    vercel_config = create_vercel_model_config(models_trained)

    if not vercel_config:
        print("No Vercel configuration created. Exiting.")
        return

    # Export for Vercel
    print("\nExporting for Vercel deployment...")
    config_file, env_file, deploy_script = export_for_vercel(vercel_config)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"Models exported: {', '.join(vercel_config.keys())}")
    print(f"Configuration file: {config_file}")
    print(f"Environment file: {env_file}")
    print(f"Deployment script: {deploy_script}")
    print("\nNext steps:")
    print("1. Run the deployment script to upload to Vercel")
    print("2. Deploy with: vercel --prod")
    print("3. Test the ML API endpoints")


if __name__ == "__main__":
    main()
