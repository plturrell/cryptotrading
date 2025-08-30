"""
Intelligence Integration Hub
Coordinates AI insights, MCTS decisions, and ML predictions
All intelligence is stored and accumulated in database
"""
import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ...data.database.client import get_db
from ...data.database.intelligence_schema import DecisionStatus, IntelligenceType
from ..agents.specialized.mcts_real_implementation import RealProductionMCTSAgent
from ..ai.grok4_client import Grok4Client, MarketInsight
from ..memory.persistent_memory import PersistentMemorySystem, create_memory_system

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceContext:
    """Context for intelligence processing"""

    session_id: str
    symbol: str
    market_data: Dict[str, Any]
    portfolio: Dict[str, float]
    timestamp: datetime


@dataclass
class CombinedIntelligence:
    """Combined intelligence from all sources"""

    ai_insights: List[MarketInsight]
    mcts_decision: Dict[str, Any]
    ml_predictions: Optional[Dict[str, Any]]
    final_recommendation: str
    confidence: float
    reasoning: str
    risk_assessment: Dict[str, Any]


class IntelligenceHub:
    """
    Central hub for coordinating all AI intelligence
    Ensures insights are stored, shared, and accumulated
    """

    def __init__(self, agent_id: str = "intelligence_hub"):
        self.agent_id = agent_id
        # Use unified database instead of legacy get_db
        from ...infrastructure.database.unified_database import UnifiedDatabase

        self.db = UnifiedDatabase()
        self.memory = None
        self.session_id = str(uuid.uuid4())

        # Initialize components
        self.grok_client = None
        self.mcts_agent = None

    async def initialize(self):
        """Initialize the intelligence hub"""
        # Initialize database first
        await self.db.initialize()

        # Initialize persistent memory
        self.memory = await create_memory_system(self.agent_id)

        # Initialize AI components
        try:
            self.grok_client = Grok4Client()
            logger.info("Grok4 client initialized")
        except Exception as e:
            logger.warning(f"Grok4 client unavailable: {e}")

        # Initialize database schemas
        await self._ensure_schemas()

        logger.info("Intelligence Hub initialized")

    async def _ensure_schemas(self):
        """Ensure intelligence database schemas exist"""
        try:
            from ...data.database.intelligence_schema import get_intelligence_schemas

            schemas = get_intelligence_schemas()

            # Use unified database connection directly
            cursor = self.db.db_conn.cursor()

            try:
                for table_name, table_schema in schemas.items():
                    # Always use SQLite for local development
                    cursor.executescript(table_schema["sqlite"])

                self.db.db_conn.commit()
                logger.info("Intelligence schemas ensured")

            finally:
                cursor.close()

        except Exception as e:
            logger.error(f"Failed to create intelligence schemas: {e}")

    async def analyze_and_decide(self, context: IntelligenceContext) -> CombinedIntelligence:
        """
        Main intelligence processing pipeline
        1. Get AI insights (and store them)
        2. Get historical intelligence from memory
        3. Make MCTS decision using all intelligence
        4. Store decision and outcomes
        5. Return combined intelligence
        """
        logger.info(f"Starting intelligence analysis for {context.symbol}")

        # 1. Get AI insights from Grok4
        ai_insights = await self._get_ai_insights(context)

        # 2. Retrieve relevant historical intelligence
        historical_intelligence = await self._get_historical_intelligence(context)

        # 3. Make MCTS decision using all available intelligence
        mcts_decision = await self._make_mcts_decision(
            context, ai_insights, historical_intelligence
        )

        # 4. Get ML predictions if available
        ml_predictions = await self._get_ml_predictions(context)

        # 5. Combine all intelligence into final recommendation
        combined = await self._combine_intelligence(
            context, ai_insights, mcts_decision, ml_predictions
        )

        # 6. Store the combined intelligence
        await self._store_combined_intelligence(context, combined)

        # 7. Learn from this interaction
        await self._learn_from_decision(context, combined)

        return combined

    async def _get_ai_insights(self, context: IntelligenceContext) -> List[MarketInsight]:
        """Get AI insights and store them in database"""
        if not self.grok_client:
            return []

        try:
            # Get fresh insights from Grok4
            insights = await self.grok_client.analyze_market_sentiment([context.symbol])

            # Store each insight in database
            for insight in insights:
                await self._store_ai_insight(insight, context)

                # Store in memory for quick access
                await self.memory.store(
                    key=f"ai_insight_{context.symbol}_{context.timestamp.isoformat()}",
                    value=asdict(insight),
                    memory_type="ai_insight",
                    importance=insight.confidence,
                    context=f"AI analysis for {context.symbol}",
                    metadata={"source": "grok4", "session_id": context.session_id},
                )

            logger.info(f"Stored {len(insights)} AI insights for {context.symbol}")
            return insights

        except Exception as e:
            logger.error(f"Failed to get AI insights: {e}")
            return []

    async def _store_ai_insight(self, insight: MarketInsight, context: IntelligenceContext):
        """Store AI insight in database"""
        cursor = self.db.db_conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO ai_insights 
                (insight_type, symbol, recommendation, confidence, score, 
                 risk_level, reasoning, source, created_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    insight.analysis_type.value,
                    insight.symbol,
                    insight.recommendation,
                    insight.confidence,
                    insight.score,
                    insight.risk_level,
                    insight.reasoning,
                    "grok4",
                    context.timestamp,
                    context.session_id,
                ),
            )
            self.db.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to store AI insight: {e}")
            self.db.db_conn.rollback()
        finally:
            cursor.close()

    async def _get_historical_intelligence(self, context: IntelligenceContext) -> Dict[str, Any]:
        """Retrieve historical intelligence for this symbol"""
        try:
            # Get recent AI insights
            recent_insights = await self.memory.search(
                query=context.symbol,
                memory_types=["ai_insight", "mcts_decision", "ml_prediction"],
                min_importance=0.6,
                limit=10,
            )

            # Get similar market conditions from database
            cursor = self.db.db_conn.cursor()

            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT ai.recommendation, ai.confidence, ai.reasoning,
                           td.action, td.confidence, td.expected_value,
                           do.profit_loss, do.success
                    FROM ai_insights ai
                    LEFT JOIN trading_decisions td ON ai.id = td.parent_insight_id
                    LEFT JOIN decision_outcomes do ON td.id = do.decision_id
                    WHERE ai.symbol = ? 
                    AND ai.created_at > ?
                    ORDER BY ai.created_at DESC
                    LIMIT 5
                """,
                    (context.symbol, datetime.utcnow() - timedelta(days=30)),
                )
            else:
                cursor.execute(
                    """
                    SELECT ai.recommendation, ai.confidence, ai.reasoning,
                           td.action, td.confidence, td.expected_value,
                           do.profit_loss, do.success
                    FROM ai_insights ai
                    LEFT JOIN trading_decisions td ON ai.id = td.parent_insight_id
                    LEFT JOIN decision_outcomes do ON td.id = do.decision_id
                    WHERE ai.symbol = %s 
                    AND ai.created_at > %s
                    ORDER BY ai.created_at DESC
                    LIMIT 5
                """,
                    (context.symbol, datetime.utcnow() - timedelta(days=30)),
                )

            historical_data = cursor.fetchall()

            # Analyze historical performance
            successful_decisions = [h for h in historical_data if h[7] is True]
            avg_success_rate = (
                len(successful_decisions) / len(historical_data) if historical_data else 0
            )

            return {
                "recent_memories": [asdict(memory) for memory in recent_insights],
                "historical_decisions": [dict(row) for row in historical_data],
                "success_rate": avg_success_rate,
                "pattern_count": len(historical_data),
            }

        except Exception as e:
            logger.error(f"Failed to get historical intelligence: {e}")
            return {}
        finally:
            cursor.close()

    async def _make_mcts_decision(
        self,
        context: IntelligenceContext,
        ai_insights: List[MarketInsight],
        historical_intelligence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make MCTS decision using AI insights and historical data"""
        try:
            # Enhance market data with AI insights
            enhanced_market_data = context.market_data.copy()

            if ai_insights:
                # Incorporate AI sentiment into market data
                ai_sentiment_score = sum(insight.score for insight in ai_insights) / len(
                    ai_insights
                )
                ai_confidence = sum(insight.confidence for insight in ai_insights) / len(
                    ai_insights
                )

                enhanced_market_data.update(
                    {
                        "ai_sentiment_score": ai_sentiment_score,
                        "ai_confidence": ai_confidence,
                        "ai_recommendation": ai_insights[0].recommendation,
                        "historical_success_rate": historical_intelligence.get("success_rate", 0.5),
                    }
                )

            # Initialize MCTS agent if needed
            if not self.mcts_agent:
                from ..agents.specialized.mcts_real_implementation import (
                    RealMCTSConfig,
                    RealProductionMCTSAgent,
                )

                config = RealMCTSConfig(simulation_budget=200)
                self.mcts_agent = RealProductionMCTSAgent(config)

                # Initialize MCTS agent with persistent memory
                if self.memory:
                    # Load historical MCTS performance data
                    mcts_memories = await self.memory.search(
                        query="mcts_decision",
                        memory_types=["mcts_decision", "learning"],
                        min_importance=0.5,
                        limit=20,
                    )

                    # Pass historical patterns to MCTS agent
                    historical_patterns = []
                    for mem in mcts_memories:
                        if isinstance(mem.value, dict):
                            historical_patterns.append(mem.value)

                    if hasattr(self.mcts_agent, "set_historical_patterns"):
                        self.mcts_agent.set_historical_patterns(historical_patterns)

            # Make decision with enhanced data
            decision = await self.mcts_agent.calculate_optimal_action(
                context.portfolio, {context.symbol: enhanced_market_data}
            )

            # Store decision in database
            await self._store_trading_decision(decision, context, ai_insights)

            # Store in memory
            await self.memory.store(
                key=f"mcts_decision_{context.symbol}_{context.timestamp.isoformat()}",
                value=decision,
                memory_type="mcts_decision",
                importance=decision.get("mcts_value", 0.5),
                context=f"MCTS decision for {context.symbol}",
                metadata={"ai_enhanced": True, "session_id": context.session_id},
            )

            return decision

        except Exception as e:
            logger.error(f"Failed to make MCTS decision: {e}")
            return {"type": "hold", "symbol": context.symbol, "amount": 0, "confidence": 0}

    async def _store_trading_decision(
        self,
        decision: Dict[str, Any],
        context: IntelligenceContext,
        ai_insights: List[MarketInsight],
    ):
        """Store trading decision in database"""
        cursor = self.db.db_conn.cursor()
        try:
            # Get the most recent AI insight ID for linking
            parent_insight_id = None
            if ai_insights:
                if self.db.config.mode.value == "local":
                    cursor.execute(
                        """
                        SELECT id FROM ai_insights 
                        WHERE symbol = ? AND session_id = ?
                        ORDER BY created_at DESC LIMIT 1
                    """,
                        (context.symbol, context.session_id),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id FROM ai_insights 
                        WHERE symbol = %s AND session_id = %s
                        ORDER BY created_at DESC LIMIT 1
                    """,
                        (context.symbol, context.session_id),
                    )

                result = cursor.fetchone()
                if result:
                    parent_insight_id = result[0]

            # Store decision
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    INSERT INTO trading_decisions 
                    (decision_type, action, symbol, amount, confidence, expected_value,
                     reasoning, algorithm, parent_insight_id, session_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        "mcts_enhanced",
                        decision.get("type"),
                        context.symbol,
                        decision.get("amount", 0),
                        decision.get("confidence", 0),
                        decision.get("mcts_value", 0),
                        decision.get("reason", ""),
                        "mcts_with_ai",
                        parent_insight_id,
                        context.session_id,
                        context.timestamp,
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO trading_decisions 
                    (decision_type, action, symbol, amount, confidence, expected_value,
                     reasoning, algorithm, parent_insight_id, session_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        "mcts_enhanced",
                        decision.get("type"),
                        context.symbol,
                        decision.get("amount", 0),
                        decision.get("confidence", 0),
                        decision.get("mcts_value", 0),
                        decision.get("reason", ""),
                        "mcts_with_ai",
                        parent_insight_id,
                        context.session_id,
                        context.timestamp,
                    ),
                )

            self.db.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to store trading decision: {e}")
            self.db.db_conn.rollback()
        finally:
            cursor.close()

    async def _get_ml_predictions(self, context: IntelligenceContext) -> Optional[Dict[str, Any]]:
        """Get ML predictions from inference service and store them"""
        try:
            # Import ML inference service
            from ..ml.inference import PredictionRequest, inference_service

            # Create prediction request
            request = PredictionRequest(
                symbol=context.symbol, horizon="24h", include_confidence=True, include_features=True
            )

            # Get ML prediction
            prediction = await inference_service.get_prediction(request)

            if prediction:
                # Store ML prediction in database
                await self._store_ml_prediction(prediction, context)

                # Store in memory
                await self.memory.store(
                    key=f"ml_prediction_{context.symbol}_{context.timestamp.isoformat()}",
                    value={
                        "predicted_price": prediction.predicted_price,
                        "confidence": prediction.confidence,
                        "price_change_percent": prediction.price_change_percent,
                        "model_type": prediction.model_type,
                        "horizon": prediction.horizon,
                    },
                    memory_type="ml_prediction",
                    importance=prediction.confidence,
                    context=f"ML prediction for {context.symbol}",
                    metadata={"source": "ml_inference", "session_id": context.session_id},
                )

                logger.info(
                    f"Stored ML prediction for {context.symbol}: {prediction.price_change_percent:.2f}% change predicted"
                )

                return {
                    "predicted_price": prediction.predicted_price,
                    "current_price": prediction.current_price,
                    "price_change_percent": prediction.price_change_percent,
                    "confidence": prediction.confidence,
                    "model_type": prediction.model_type,
                    "horizon": prediction.horizon,
                    "features_used": prediction.features_used,
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get ML predictions: {e}")
            return None

    async def _store_ml_prediction(self, prediction, context: IntelligenceContext):
        """Store ML prediction in database"""
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    INSERT INTO ml_predictions 
                    (model_type, symbol, prediction_type, predicted_value, confidence,
                     time_horizon, features_used, model_version, created_at, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        prediction.model_type,
                        prediction.symbol,
                        "price_prediction",
                        prediction.predicted_price,
                        prediction.confidence,
                        prediction.horizon,
                        str(prediction.features_used),
                        prediction.model_version,
                        context.timestamp,
                        context.session_id,
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO ml_predictions 
                    (model_type, symbol, prediction_type, predicted_value, confidence,
                     time_horizon, features_used, model_version, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        prediction.model_type,
                        prediction.symbol,
                        "price_prediction",
                        prediction.predicted_price,
                        prediction.confidence,
                        prediction.horizon,
                        str(prediction.features_used),
                        prediction.model_version,
                        context.session_id,
                    ),
                )

            self.db.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to store ML prediction: {e}")
            self.db.db_conn.rollback()
        finally:
            cursor.close()

    async def _combine_intelligence(
        self,
        context: IntelligenceContext,
        ai_insights: List[MarketInsight],
        mcts_decision: Dict[str, Any],
        ml_predictions: Optional[Dict[str, Any]],
    ) -> CombinedIntelligence:
        """Combine all intelligence sources into final recommendation"""

        # Analyze AI consensus
        if ai_insights:
            ai_recommendations = [insight.recommendation for insight in ai_insights]
            ai_confidence = sum(insight.confidence for insight in ai_insights) / len(ai_insights)

            # Find consensus
            buy_votes = ai_recommendations.count("BUY")
            sell_votes = ai_recommendations.count("SELL")
            hold_votes = ai_recommendations.count("HOLD")

            if buy_votes > sell_votes and buy_votes > hold_votes:
                ai_consensus = "BUY"
            elif sell_votes > buy_votes and sell_votes > hold_votes:
                ai_consensus = "SELL"
            else:
                ai_consensus = "HOLD"
        else:
            ai_consensus = "HOLD"
            ai_confidence = 0.5

        # Get MCTS recommendation
        mcts_action = mcts_decision.get("type", "hold").upper()
        mcts_confidence = mcts_decision.get("confidence", 0.5)

        # Combine recommendations
        if ai_consensus == mcts_action:
            # Agreement between AI and MCTS
            final_recommendation = ai_consensus
            confidence = (ai_confidence + mcts_confidence) / 2
            reasoning = f"AI and MCTS agree: {ai_consensus}"
        else:
            # Disagreement - use higher confidence
            if ai_confidence > mcts_confidence:
                final_recommendation = ai_consensus
                confidence = ai_confidence * 0.8  # Reduce confidence due to disagreement
                reasoning = f"AI recommends {ai_consensus} (confidence: {ai_confidence:.2f}) vs MCTS {mcts_action}"
            else:
                final_recommendation = mcts_action
                confidence = mcts_confidence * 0.8
                reasoning = f"MCTS recommends {mcts_action} (confidence: {mcts_confidence:.2f}) vs AI {ai_consensus}"

        # Risk assessment
        risk_factors = []
        if ai_insights:
            for insight in ai_insights:
                if insight.risk_level == "HIGH":
                    risk_factors.append(f"AI identifies high risk: {insight.reasoning}")

        risk_assessment = {
            "overall_risk": "HIGH"
            if any("HIGH" in insight.risk_level for insight in ai_insights)
            else "MEDIUM",
            "confidence_agreement": ai_consensus == mcts_action,
            "risk_factors": risk_factors,
        }

        return CombinedIntelligence(
            ai_insights=ai_insights,
            mcts_decision=mcts_decision,
            ml_predictions=ml_predictions,
            final_recommendation=final_recommendation,
            confidence=confidence,
            reasoning=reasoning,
            risk_assessment=risk_assessment,
        )

    async def _store_combined_intelligence(
        self, context: IntelligenceContext, combined: CombinedIntelligence
    ):
        """Store the combined intelligence result"""
        # Store in memory for quick access
        await self.memory.store(
            key=f"combined_intelligence_{context.symbol}_{context.timestamp.isoformat()}",
            value=asdict(combined),
            memory_type="combined_intelligence",
            importance=combined.confidence,
            context=f"Combined analysis for {context.symbol}: {combined.final_recommendation}",
            metadata={
                "session_id": context.session_id,
                "ai_count": len(combined.ai_insights),
                "has_mcts": bool(combined.mcts_decision),
                "has_ml": bool(combined.ml_predictions),
            },
        )

        # Store conversation in database
        await self._store_conversation(context, combined)

    async def _store_conversation(
        self, context: IntelligenceContext, combined: CombinedIntelligence
    ):
        """Store conversation history"""
        cursor = self.db.db_conn.cursor()
        try:
            # Store user query
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    INSERT INTO conversation_history 
                    (session_id, agent_id, message_type, message_content, context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        context.session_id,
                        self.agent_id,
                        "query",
                        f"Analyze {context.symbol}",
                        json.dumps(asdict(context)),
                        context.timestamp,
                    ),
                )

                # Store response
                cursor.execute(
                    """
                    INSERT INTO conversation_history 
                    (session_id, agent_id, message_type, message_content, context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        context.session_id,
                        self.agent_id,
                        "response",
                        f"Recommendation: {combined.final_recommendation} (confidence: {combined.confidence:.2f})",
                        json.dumps(asdict(combined)),
                        context.timestamp,
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO conversation_history 
                    (session_id, agent_id, message_type, message_content, context, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """,
                    (
                        context.session_id,
                        self.agent_id,
                        "query",
                        f"Analyze {context.symbol}",
                        json.dumps(asdict(context)),
                        context.timestamp,
                    ),
                )

                cursor.execute(
                    """
                    INSERT INTO conversation_history 
                    (session_id, agent_id, message_type, message_content, context, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """,
                    (
                        context.session_id,
                        self.agent_id,
                        "response",
                        f"Recommendation: {combined.final_recommendation} (confidence: {combined.confidence:.2f})",
                        json.dumps(asdict(combined)),
                        context.timestamp,
                    ),
                )

            self.db.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            self.db.db_conn.rollback()
        finally:
            cursor.close()

    async def _learn_from_decision(
        self, context: IntelligenceContext, combined: CombinedIntelligence
    ):
        """Learn from this decision for future improvements"""
        # Extract key learnings
        learnings = {
            "symbol_analyzed": context.symbol,
            "ai_consensus": combined.ai_insights[0].recommendation
            if combined.ai_insights
            else None,
            "mcts_action": combined.mcts_decision.get("type"),
            "final_recommendation": combined.final_recommendation,
            "confidence": combined.confidence,
            "timestamp": context.timestamp.isoformat(),
        }

        # Store learning
        await self.memory.store(
            key=f"learning_{context.symbol}_{context.timestamp.strftime('%Y%m%d')}",
            value=learnings,
            memory_type="learning",
            importance=combined.confidence,
            context=f"Learning from {context.symbol} analysis",
            metadata={"type": "decision_learning", "session_id": context.session_id},
        )

        logger.info(f"Stored learning from {context.symbol} decision")

    async def get_intelligence_history(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Get intelligence history for a symbol"""
        cursor = self.db.db_conn.cursor()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT ai.recommendation, ai.confidence, ai.reasoning, ai.created_at,
                           td.action, td.expected_value, td.created_at as decision_time
                    FROM ai_insights ai
                    LEFT JOIN trading_decisions td ON ai.id = td.parent_insight_id
                    WHERE ai.symbol = ? AND ai.created_at > ?
                    ORDER BY ai.created_at DESC
                """,
                    (symbol, cutoff_date),
                )
            else:
                cursor.execute(
                    """
                    SELECT ai.recommendation, ai.confidence, ai.reasoning, ai.created_at,
                           td.action, td.expected_value, td.created_at as decision_time
                    FROM ai_insights ai
                    LEFT JOIN trading_decisions td ON ai.id = td.parent_insight_id
                    WHERE ai.symbol = %s AND ai.created_at > %s
                    ORDER BY ai.created_at DESC
                """,
                    (symbol, cutoff_date),
                )

            history = []
            for row in cursor.fetchall():
                history.append(
                    {
                        "ai_recommendation": row[0],
                        "ai_confidence": row[1],
                        "ai_reasoning": row[2],
                        "ai_timestamp": row[3],
                        "mcts_action": row[4],
                        "mcts_expected_value": row[5],
                        "decision_timestamp": row[6],
                    }
                )

            return {
                "symbol": symbol,
                "period_days": days,
                "total_analyses": len(history),
                "history": history,
            }

        except Exception as e:
            logger.error(f"Failed to get intelligence history: {e}")
            return {}
        finally:
            cursor.close()

    async def close(self):
        """Clean up resources"""
        if self.grok_client:
            await self.grok_client.close()
        logger.info("Intelligence Hub closed")


# Global intelligence hub instance
_intelligence_hub: Optional[IntelligenceHub] = None


async def get_intelligence_hub() -> IntelligenceHub:
    """Get global intelligence hub instance"""
    global _intelligence_hub
    if _intelligence_hub is None:
        _intelligence_hub = IntelligenceHub()
        await _intelligence_hub.initialize()
    return _intelligence_hub
