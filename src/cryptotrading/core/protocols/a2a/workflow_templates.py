"""
Pre-defined Workflow Templates for Cross-Agent Operations
Templates for common A2A workflows with on-chain data exchange
"""

import logging
from typing import Dict, List, Any

from .workflow_orchestration import (
    WorkflowTemplate,
    WorkflowStep,
    WorkflowStepType
)

logger = logging.getLogger(__name__)


class WorkflowTemplateLibrary:
    """Library of pre-defined workflow templates"""
    
    @staticmethod
    def create_ml_training_workflow() -> WorkflowTemplate:
        """Create ML model training workflow"""
        return WorkflowTemplate(
            template_id="ml_training_v1",
            name="ML Model Training Pipeline",
            description="End-to-end ML model training with data validation",
            steps=[
                WorkflowStep(
                    step_id="fetch_training_data",
                    step_type=WorkflowStepType.DATA_COLLECTION,
                    agent_id="aws-data-exchange-agent",
                    parameters={
                        "dataset": "crypto_historical",
                        "timeframe": "30d",
                        "features": ["price", "volume", "volatility"]
                    }
                ),
                WorkflowStep(
                    step_id="validate_data",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="data-analysis-agent",
                    parameters={
                        "validation_checks": ["completeness", "outliers", "stationarity"],
                        "threshold": 0.95
                    }
                ),
                WorkflowStep(
                    step_id="feature_engineering",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="strands-glean-agent",
                    parameters={
                        "feature_types": ["technical", "market_microstructure", "sentiment"],
                        "glean_patterns": True
                    }
                ),
                WorkflowStep(
                    step_id="train_model",
                    step_type=WorkflowStepType.ML_PREDICTION,
                    agent_id="ml-agent",
                    parameters={
                        "model_type": "ensemble",
                        "algorithms": ["xgboost", "lstm", "random_forest"],
                        "validation_split": 0.2
                    }
                ),
                WorkflowStep(
                    step_id="evaluate_performance",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="mcts-calculation-agent",
                    parameters={
                        "metrics": ["sharpe", "max_drawdown", "win_rate"],
                        "monte_carlo_simulations": 1000
                    }
                ),
                WorkflowStep(
                    step_id="generate_report",
                    step_type=WorkflowStepType.REPORT_GENERATION,
                    agent_id="data-analysis-agent",
                    parameters={
                        "report_type": "model_evaluation",
                        "include_visualizations": True
                    }
                )
            ],
            required_agents={
                "aws-data-exchange-agent",
                "data-analysis-agent",
                "strands-glean-agent",
                "ml-agent",
                "mcts-calculation-agent"
            },
            expected_duration_seconds=600
        )
    
    @staticmethod
    def create_real_time_trading_workflow() -> WorkflowTemplate:
        """Create real-time trading signal workflow"""
        return WorkflowTemplate(
            template_id="realtime_trading_v1",
            name="Real-Time Trading Signal Generation",
            description="Generate and validate trading signals in real-time",
            steps=[
                WorkflowStep(
                    step_id="stream_market_data",
                    step_type=WorkflowStepType.DATA_COLLECTION,
                    agent_id="aws-data-exchange-agent",
                    parameters={
                        "stream_type": "realtime",
                        "symbols": ["BTC/USDT", "ETH/USDT"],
                        "data_points": ["price", "volume", "order_book"]
                    }
                ),
                WorkflowStep(
                    step_id="detect_patterns",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="strands-glean-agent",
                    parameters={
                        "pattern_types": ["breakout", "reversal", "momentum"],
                        "sensitivity": "high",
                        "glean_depth": 3
                    }
                ),
                WorkflowStep(
                    step_id="analyze_technicals",
                    step_type=WorkflowStepType.TECHNICAL_ANALYSIS,
                    agent_id="technical-analysis-agent",
                    parameters={
                        "indicators": ["RSI", "MACD", "BB", "VWAP"],
                        "timeframes": ["1m", "5m", "15m"]
                    }
                ),
                WorkflowStep(
                    step_id="predict_movement",
                    step_type=WorkflowStepType.ML_PREDICTION,
                    agent_id="ml-agent",
                    parameters={
                        "prediction_horizon": "5m",
                        "confidence_threshold": 0.75,
                        "use_ensemble": True
                    }
                ),
                WorkflowStep(
                    step_id="assess_risk",
                    step_type=WorkflowStepType.RISK_ASSESSMENT,
                    agent_id="mcts-calculation-agent",
                    parameters={
                        "risk_metrics": ["VaR", "expected_shortfall"],
                        "position_sizing": "kelly_criterion",
                        "max_risk_percent": 2
                    }
                ),
                WorkflowStep(
                    step_id="generate_signal",
                    step_type=WorkflowStepType.TRADE_EXECUTION,
                    agent_id="trading-agent",
                    parameters={
                        "signal_type": "limit_order",
                        "execution_mode": "paper_trading"
                    }
                )
            ],
            required_agents={
                "aws-data-exchange-agent",
                "strands-glean-agent",
                "technical-analysis-agent",
                "ml-agent",
                "mcts-calculation-agent",
                "trading-agent"
            },
            expected_duration_seconds=30
        )
    
    @staticmethod
    def create_defi_yield_optimization_workflow() -> WorkflowTemplate:
        """Create DeFi yield farming optimization workflow"""
        return WorkflowTemplate(
            template_id="defi_yield_v1",
            name="DeFi Yield Optimization",
            description="Optimize yield farming strategies across DeFi protocols",
            steps=[
                WorkflowStep(
                    step_id="scan_protocols",
                    step_type=WorkflowStepType.DATA_COLLECTION,
                    agent_id="aws-data-exchange-agent",
                    parameters={
                        "data_source": "defi_pulse",
                        "protocols": ["aave", "compound", "uniswap", "curve"],
                        "metrics": ["tvl", "apy", "risk_score"]
                    }
                ),
                WorkflowStep(
                    step_id="analyze_yields",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="data-analysis-agent",
                    parameters={
                        "analysis_type": "comparative",
                        "include_impermanent_loss": True,
                        "gas_cost_estimation": True
                    }
                ),
                WorkflowStep(
                    step_id="detect_opportunities",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="strands-glean-agent",
                    parameters={
                        "opportunity_types": ["arbitrage", "yield_farming", "liquidity_provision"],
                        "min_apy_threshold": 10,
                        "glean_cross_protocol": True
                    }
                ),
                WorkflowStep(
                    step_id="simulate_strategies",
                    step_type=WorkflowStepType.ML_PREDICTION,
                    agent_id="mcts-calculation-agent",
                    parameters={
                        "simulation_count": 10000,
                        "time_horizon": "30d",
                        "include_protocol_risks": True
                    }
                ),
                WorkflowStep(
                    step_id="optimize_allocation",
                    step_type=WorkflowStepType.ML_PREDICTION,
                    agent_id="ml-agent",
                    parameters={
                        "optimization_method": "portfolio_theory",
                        "constraints": ["max_protocol_exposure", "min_liquidity"],
                        "rebalance_frequency": "weekly"
                    }
                ),
                WorkflowStep(
                    step_id="generate_strategy",
                    step_type=WorkflowStepType.REPORT_GENERATION,
                    agent_id="strategy-agent",
                    parameters={
                        "strategy_format": "actionable",
                        "include_gas_optimization": True
                    }
                )
            ],
            required_agents={
                "aws-data-exchange-agent",
                "data-analysis-agent",
                "strands-glean-agent",
                "mcts-calculation-agent",
                "ml-agent",
                "strategy-agent"
            },
            expected_duration_seconds=300
        )
    
    @staticmethod
    def create_risk_monitoring_workflow() -> WorkflowTemplate:
        """Create continuous risk monitoring workflow"""
        return WorkflowTemplate(
            template_id="risk_monitoring_v1",
            name="Continuous Risk Monitoring",
            description="Monitor and alert on portfolio risk metrics",
            steps=[
                WorkflowStep(
                    step_id="fetch_positions",
                    step_type=WorkflowStepType.DATA_COLLECTION,
                    agent_id="portfolio-agent",
                    parameters={
                        "include_derivatives": True,
                        "include_defi_positions": True
                    }
                ),
                WorkflowStep(
                    step_id="calculate_exposures",
                    step_type=WorkflowStepType.RISK_ASSESSMENT,
                    agent_id="risk-agent",
                    parameters={
                        "exposure_types": ["directional", "volatility", "correlation"],
                        "granularity": "position_level"
                    }
                ),
                WorkflowStep(
                    step_id="stress_testing",
                    step_type=WorkflowStepType.RISK_ASSESSMENT,
                    agent_id="mcts-calculation-agent",
                    parameters={
                        "scenarios": ["market_crash", "flash_crash", "black_swan"],
                        "confidence_intervals": [0.95, 0.99],
                        "monte_carlo_paths": 10000
                    }
                ),
                WorkflowStep(
                    step_id="detect_anomalies",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="strands-glean-agent",
                    parameters={
                        "anomaly_detection": "statistical",
                        "sensitivity": "medium",
                        "glean_historical_patterns": True
                    }
                ),
                WorkflowStep(
                    step_id="generate_alerts",
                    step_type=WorkflowStepType.REPORT_GENERATION,
                    agent_id="alert-agent",
                    parameters={
                        "alert_levels": ["info", "warning", "critical"],
                        "notification_channels": ["dashboard", "email"]
                    }
                )
            ],
            required_agents={
                "portfolio-agent",
                "risk-agent",
                "mcts-calculation-agent",
                "strands-glean-agent",
                "alert-agent"
            },
            expected_duration_seconds=60
        )
    
    @staticmethod
    def create_market_research_workflow() -> WorkflowTemplate:
        """Create comprehensive market research workflow"""
        return WorkflowTemplate(
            template_id="market_research_v1",
            name="Market Research Pipeline",
            description="Comprehensive market analysis and research",
            steps=[
                WorkflowStep(
                    step_id="collect_market_data",
                    step_type=WorkflowStepType.DATA_COLLECTION,
                    agent_id="aws-data-exchange-agent",
                    parameters={
                        "data_types": ["price", "volume", "social", "news", "on_chain"],
                        "lookback_period": "90d",
                        "granularity": "1h"
                    }
                ),
                WorkflowStep(
                    step_id="fundamental_analysis",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="data-analysis-agent",
                    parameters={
                        "metrics": ["market_cap", "volume_profile", "holder_distribution"],
                        "comparative_analysis": True
                    }
                ),
                WorkflowStep(
                    step_id="pattern_recognition",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="strands-glean-agent",
                    parameters={
                        "pattern_library": "comprehensive",
                        "multi_timeframe": True,
                        "glean_emerging_patterns": True
                    }
                ),
                WorkflowStep(
                    step_id="sentiment_analysis",
                    step_type=WorkflowStepType.DATA_ANALYSIS,
                    agent_id="sentiment-agent",
                    parameters={
                        "sources": ["twitter", "reddit", "news"],
                        "sentiment_model": "fine_tuned_crypto"
                    }
                ),
                WorkflowStep(
                    step_id="forecast_generation",
                    step_type=WorkflowStepType.ML_PREDICTION,
                    agent_id="ml-agent",
                    parameters={
                        "forecast_horizons": ["1d", "7d", "30d"],
                        "confidence_bands": True,
                        "scenario_analysis": True
                    }
                ),
                WorkflowStep(
                    step_id="compile_research",
                    step_type=WorkflowStepType.REPORT_GENERATION,
                    agent_id="research-agent",
                    parameters={
                        "report_format": "comprehensive",
                        "include_charts": True,
                        "executive_summary": True
                    }
                )
            ],
            required_agents={
                "aws-data-exchange-agent",
                "data-analysis-agent",
                "strands-glean-agent",
                "sentiment-agent",
                "ml-agent",
                "research-agent"
            },
            expected_duration_seconds=900
        )
    
    @staticmethod
    def get_all_templates() -> Dict[str, WorkflowTemplate]:
        """Get all available workflow templates"""
        return {
            "ml_training": WorkflowTemplateLibrary.create_ml_training_workflow(),
            "realtime_trading": WorkflowTemplateLibrary.create_real_time_trading_workflow(),
            "defi_yield": WorkflowTemplateLibrary.create_defi_yield_optimization_workflow(),
            "risk_monitoring": WorkflowTemplateLibrary.create_risk_monitoring_workflow(),
            "market_research": WorkflowTemplateLibrary.create_market_research_workflow()
        }
    
    @staticmethod
    def create_custom_workflow(
        template_id: str,
        name: str,
        description: str,
        steps: List[Dict[str, Any]]
    ) -> WorkflowTemplate:
        """
        Create a custom workflow template
        
        Args:
            template_id: Unique template identifier
            name: Template name
            description: Template description
            steps: List of step definitions
        
        Returns:
            Custom workflow template
        """
        workflow_steps = []
        required_agents = set()
        
        for step_def in steps:
            step = WorkflowStep(
                step_id=step_def["step_id"],
                step_type=WorkflowStepType[step_def["step_type"]],
                agent_id=step_def["agent_id"],
                parameters=step_def.get("parameters", {})
            )
            workflow_steps.append(step)
            required_agents.add(step_def["agent_id"])
        
        return WorkflowTemplate(
            template_id=template_id,
            name=name,
            description=description,
            steps=workflow_steps,
            required_agents=required_agents,
            expected_duration_seconds=len(steps) * 60  # Estimate 60s per step
        )


# Export templates for easy access
ML_TRAINING_WORKFLOW = WorkflowTemplateLibrary.create_ml_training_workflow()
REALTIME_TRADING_WORKFLOW = WorkflowTemplateLibrary.create_real_time_trading_workflow()
DEFI_YIELD_WORKFLOW = WorkflowTemplateLibrary.create_defi_yield_optimization_workflow()
RISK_MONITORING_WORKFLOW = WorkflowTemplateLibrary.create_risk_monitoring_workflow()
MARKET_RESEARCH_WORKFLOW = WorkflowTemplateLibrary.create_market_research_workflow()