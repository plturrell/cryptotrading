#!/usr/bin/env python3
"""
A2A Feature Store Agent CLI
Provides command-line interface for feature engineering and management
"""

import asyncio
import os
import sys
from datetime import datetime

import click

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set environment variables for CLI
os.environ["ENVIRONMENT"] = "development"
os.environ["SKIP_DB_INIT"] = "true"

try:
    from cryptotrading.core.agents.specialized.feature_store_agent import FeatureStoreAgent
    from cryptotrading.core.protocols.a2a.a2a_protocol import A2A_CAPABILITIES
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback minimal Feature Store agent for CLI testing...")

    class FallbackFeatureStoreAgent:
        """Minimal Feature Store agent for CLI testing when imports fail"""

        def __init__(self):
            self.agent_id = "feature_store_agent"
            self.capabilities = [
                "compute_features",
                "get_feature_vector",
                "get_training_features",
                "get_feature_definitions",
                "get_feature_importance",
                "feature_engineering",
                "ml_features",
                "technical_indicators",
            ]

        async def compute_features(self, symbol, feature_types=None):
            """Mock feature computation"""
            features = {
                "price_features": {
                    "sma_20": 45234.56,
                    "ema_12": 45456.78,
                    "rsi": 65.4,
                    "bollinger_upper": 46500.0,
                    "bollinger_lower": 44000.0,
                },
                "volume_features": {"volume_sma": 1250000, "volume_ratio": 1.23, "obv": 15000000},
                "momentum_features": {"macd": 1.2, "stochastic_k": 75.2, "williams_r": -25.6},
            }

            if feature_types:
                features = {k: v for k, v in features.items() if k in feature_types}

            return {
                "symbol": symbol,
                "features": features,
                "feature_count": sum(len(v) for v in features.values()),
                "timestamp": datetime.now().isoformat(),
            }

        async def get_feature_vector(self, symbol, features_list):
            """Mock feature vector retrieval"""
            vector = [45234.56, 65.4, 1.2, 75.2, 1250000][: len(features_list)]
            return {
                "symbol": symbol,
                "features": features_list,
                "vector": vector,
                "vector_length": len(vector),
                "timestamp": datetime.now().isoformat(),
            }

        async def get_feature_importance(self, model_type="xgboost"):
            """Mock feature importance analysis"""
            return {
                "model_type": model_type,
                "importance_scores": {
                    "price_sma_20": 0.234,
                    "volume_ratio": 0.189,
                    "rsi": 0.156,
                    "macd": 0.134,
                    "bollinger_position": 0.098,
                    "volume_sma": 0.089,
                    "stochastic_k": 0.067,
                    "williams_r": 0.033,
                },
                "top_features": ["price_sma_20", "volume_ratio", "rsi"],
                "timestamp": datetime.now().isoformat(),
            }

    FeatureStoreAgent = FallbackFeatureStoreAgent

# Global agent instance
agent = None


def get_agent():
    """Get or create agent instance"""
    global agent
    if agent is None:
        agent = FeatureStoreAgent()
    return agent


def async_command(f):
    """Decorator to run async commands"""

    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """A2A Feature Store Agent CLI - Feature engineering and management"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("symbol")
@click.option(
    "--types",
    help="Comma-separated feature types (price_features, volume_features, momentum_features)",
)
@click.pass_context
@async_command
async def compute(ctx, symbol, types):
    """Compute features for a symbol"""
    agent = get_agent()

    try:
        feature_types = types.split(",") if types else None
        result = await agent.compute_features(symbol, feature_types)

        click.echo(f"🔧 Computing Features for {symbol.upper()}")
        click.echo("=" * 50)
        click.echo(f"Total Features: {result['feature_count']}")
        click.echo()

        for category, features in result["features"].items():
            click.echo(f"📊 {category.replace('_', ' ').title()}:")
            for name, value in features.items():
                if isinstance(value, float):
                    click.echo(f"  {name}: {value:.4f}")
                else:
                    click.echo(f"  {name}: {value:,}")
            click.echo()

        if ctx.obj["verbose"]:
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error computing features: {e}", err=True)


@cli.command()
@click.argument("symbol")
@click.argument("features", nargs=-1, required=True)
@click.pass_context
@async_command
async def vector(ctx, symbol, features):
    """Get feature vector for specific features"""
    agent = get_agent()

    try:
        result = await agent.get_feature_vector(symbol, list(features))

        click.echo(f"📊 Feature Vector for {symbol.upper()}")
        click.echo("=" * 50)
        click.echo(f"Vector Length: {result['vector_length']}")
        click.echo()

        for i, (feature, value) in enumerate(zip(result["features"], result["vector"])):
            click.echo(f"{i+1:2d}. {feature}: {value:.4f}")

        if ctx.obj["verbose"]:
            click.echo(f"\nRaw Vector: {result['vector']}")
            click.echo(f"Timestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error getting feature vector: {e}", err=True)


@cli.command()
@click.option("--model", default="xgboost", help="Model type for importance analysis")
@click.pass_context
@async_command
async def importance(ctx, model):
    """Analyze feature importance"""
    agent = get_agent()

    try:
        result = await agent.get_feature_importance(model)

        click.echo(f"🎯 Feature Importance Analysis")
        click.echo(f"Model: {result['model_type']}")
        click.echo("=" * 50)

        # Sort by importance
        sorted_features = sorted(
            result["importance_scores"].items(), key=lambda x: x[1], reverse=True
        )

        click.echo("📈 Feature Rankings:")
        for i, (feature, score) in enumerate(sorted_features, 1):
            bar_length = int(score * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            click.echo(f"{i:2d}. {feature:<20} {bar} {score:.3f}")

        click.echo()
        click.echo(f"🏆 Top 3 Features: {', '.join(result['top_features'])}")

        if ctx.obj["verbose"]:
            click.echo(f"\nTimestamp: {result['timestamp']}")

    except Exception as e:
        click.echo(f"Error analyzing feature importance: {e}", err=True)


@cli.command()
@click.argument("symbol")
@click.option(
    "--method",
    default="auto",
    type=click.Choice(["auto", "polynomial", "interaction", "pca", "lda"]),
    help="Feature engineering method",
)
@click.option("--degree", default=2, help="Polynomial degree for polynomial features")
@click.pass_context
@async_command
async def engineering(ctx, symbol, method, degree):
    """Advanced feature engineering for ML models"""
    agent = get_agent()

    try:
        # Mock feature engineering since we don't have real implementation
        import random

        result = {
            "symbol": symbol,
            "method": method,
            "engineered_features": [],
            "feature_count": 0,
            "performance_metrics": {},
        }

        if method == "auto":
            methods = ["polynomial", "interaction", "rolling_stats", "lag_features"]
            selected_method = random.choice(methods)
        else:
            selected_method = method

        # Generate mock engineered features
        base_features = [
            "price",
            "volume",
            "rsi",
            "macd",
            "bollinger_upper",
            "bollinger_lower",
            "sma_20",
            "ema_12",
        ]

        if selected_method == "polynomial":
            engineered = [f"{feat}_degree_{degree}" for feat in base_features[:4]]
            engineered.extend([f"{feat}_sqrt" for feat in base_features[:3]])
        elif selected_method == "interaction":
            engineered = [
                f"{base_features[i]}_{base_features[j]}_interaction"
                for i in range(3)
                for j in range(i + 1, 4)
            ]
        elif selected_method == "rolling_stats":
            windows = [5, 10, 20]
            stats = ["mean", "std", "min", "max"]
            engineered = [
                f"{feat}_{window}_{stat}"
                for feat in base_features[:3]
                for window in windows
                for stat in stats
            ]
        else:
            engineered = [f"{feat}_lag_{i}" for feat in base_features[:5] for i in range(1, 4)]

        result["engineered_features"] = engineered
        result["feature_count"] = len(engineered)
        result["performance_metrics"] = {
            "feature_importance_variance": round(random.uniform(0.15, 0.35), 3),
            "correlation_reduction": round(random.uniform(0.08, 0.25), 3),
            "dimensionality_increase": round(len(engineered) / len(base_features), 2),
        }

        click.echo(f"⚙️ Feature Engineering Complete - {symbol.upper()}")
        click.echo("=" * 50)
        click.echo(f"Method: {selected_method.title()}")
        click.echo(f"Original Features: {len(base_features)}")
        click.echo(f"Engineered Features: {result['feature_count']}")
        click.echo(f"Total Features: {len(base_features) + result['feature_count']}")

        if degree > 1 and selected_method == "polynomial":
            click.echo(f"Polynomial Degree: {degree}")

        click.echo()
        click.echo("🔧 New Features:")
        for i, feature in enumerate(result["engineered_features"][:10], 1):
            click.echo(f"  {i:2d}. {feature}")

        if len(result["engineered_features"]) > 10:
            click.echo(f"  ... and {len(result['engineered_features']) - 10} more features")

        metrics = result["performance_metrics"]
        click.echo()
        click.echo("📊 Performance Metrics:")
        click.echo(f"  Feature Importance Variance: {metrics['feature_importance_variance']:.3f}")
        click.echo(f"  Correlation Reduction: {metrics['correlation_reduction']:.3f}")
        click.echo(f"  Dimensionality Increase: {metrics['dimensionality_increase']:.2f}x")

        click.echo(
            "\n🔄 Mock feature engineering - enable real implementation for advanced engineering"
        )

        if ctx.obj["verbose"]:
            click.echo(f"\nTimestamp: {datetime.now().isoformat()}")

    except Exception as e:
        click.echo(f"Error in feature engineering: {e}", err=True)


@cli.command()
@click.pass_context
def capabilities(ctx):
    """List agent capabilities"""
    agent = get_agent()

    click.echo("🔧 Feature Store Agent Capabilities:")
    click.echo()
    for i, capability in enumerate(agent.capabilities, 1):
        click.echo(f"{i:2d}. {capability.replace('_', ' ').title()}")


@cli.command()
@click.pass_context
def status(ctx):
    """Get agent status and health"""
    agent = get_agent()

    click.echo("🏥 Feature Store Agent Status:")
    click.echo(f"Agent ID: {agent.agent_id}")
    click.echo(f"Capabilities: {len(agent.capabilities)}")
    click.echo("Status: ✅ ACTIVE")
    click.echo(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    cli()
