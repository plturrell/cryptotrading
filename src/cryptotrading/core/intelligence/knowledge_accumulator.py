"""
Knowledge Accumulation System
Builds and maintains accumulated intelligence from all interactions
Enables "Every interaction starts from accumulated knowledge and history"
"""
import asyncio
import json
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ...data.database.client import get_db
from ..memory.persistent_memory import PersistentMemorySystem
from .decision_audit import PerformanceMetrics, get_audit_trail

logger = logging.getLogger(__name__)


@dataclass
class KnowledgePattern:
    """A learned pattern from historical data"""

    pattern_id: str
    pattern_type: str  # market_condition, ai_accuracy, algorithm_performance
    description: str
    confidence: float
    success_rate: float
    sample_size: int
    conditions: Dict[str, Any]
    outcomes: List[Dict[str, Any]]
    last_updated: datetime


@dataclass
class AccumulatedKnowledge:
    """Complete accumulated knowledge state"""

    session_id: str
    total_interactions: int
    success_patterns: List[KnowledgePattern]
    failure_patterns: List[KnowledgePattern]
    market_insights: Dict[str, Any]
    agent_performance: Dict[str, PerformanceMetrics]
    learned_strategies: List[Dict[str, Any]]
    risk_assessments: Dict[str, Any]
    confidence_calibration: Dict[str, float]


class KnowledgeAccumulator:
    """
    Accumulates knowledge from all system interactions
    Ensures every new interaction builds on previous learning
    """

    def __init__(self, agent_id: str = "knowledge_accumulator"):
        self.agent_id = agent_id
        # Use unified database instead of legacy get_db
        from ...infrastructure.database.unified_database import UnifiedDatabase

        self.db = UnifiedDatabase()
        self.memory: Optional[PersistentMemorySystem] = None
        self.audit_trail = get_audit_trail()
        self._patterns_cache = {}

    async def initialize(self):
        """Initialize knowledge accumulator"""
        # Ensure database is initialized
        await self.db.initialize()

        from ..memory.persistent_memory import create_memory_system

        self.memory = await create_memory_system(self.agent_id)

        # Load existing knowledge patterns
        await self._load_existing_patterns()

        logger.info("Knowledge Accumulator initialized")

    async def get_accumulated_knowledge(self, session_id: str) -> AccumulatedKnowledge:
        """
        Get all accumulated knowledge for starting a new interaction
        This is the core function that provides historical context
        """
        logger.info("Building accumulated knowledge for new interaction")

        try:
            # Get interaction count
            total_interactions = await self._get_total_interactions()

            # Get success and failure patterns
            success_patterns = await self._get_success_patterns()
            failure_patterns = await self._get_failure_patterns()

            # Get market insights by symbol
            market_insights = await self._get_market_insights()

            # Get agent performance metrics
            agent_performance = await self._get_agent_performance()

            # Get learned strategies
            learned_strategies = await self._get_learned_strategies()

            # Get risk assessments
            risk_assessments = await self._get_risk_assessments()

            # Get confidence calibration
            confidence_calibration = await self._get_confidence_calibration()

            knowledge = AccumulatedKnowledge(
                session_id=session_id,
                total_interactions=total_interactions,
                success_patterns=success_patterns,
                failure_patterns=failure_patterns,
                market_insights=market_insights,
                agent_performance=agent_performance,
                learned_strategies=learned_strategies,
                risk_assessments=risk_assessments,
                confidence_calibration=confidence_calibration,
            )

            # Store this knowledge snapshot
            await self._store_knowledge_snapshot(knowledge)

            logger.info(
                f"Built accumulated knowledge: {total_interactions} interactions, "
                f"{len(success_patterns)} success patterns, {len(failure_patterns)} failure patterns"
            )

            return knowledge

        except Exception as e:
            logger.error(f"Failed to build accumulated knowledge: {e}")
            # Return minimal knowledge on error
            return AccumulatedKnowledge(
                session_id=session_id,
                total_interactions=0,
                success_patterns=[],
                failure_patterns=[],
                market_insights={},
                agent_performance={},
                learned_strategies=[],
                risk_assessments={},
                confidence_calibration={},
            )

    async def learn_from_interaction(self, interaction_data: Dict[str, Any]):
        """
        Learn from a completed interaction and update knowledge
        """
        try:
            logger.info("Learning from interaction")

            # Extract learning elements
            symbol = interaction_data.get("symbol")
            ai_insights = interaction_data.get("ai_insights", [])
            mcts_decision = interaction_data.get("mcts_decision", {})
            final_recommendation = interaction_data.get("final_recommendation")
            confidence = interaction_data.get("confidence", 0)
            outcome = interaction_data.get("outcome")

            # Update patterns
            if outcome:
                await self._update_success_failure_patterns(interaction_data)

            # Update market insights
            if symbol and ai_insights:
                await self._update_market_insights(symbol, ai_insights)

            # Update algorithm performance
            if mcts_decision:
                await self._update_algorithm_performance(mcts_decision, outcome)

            # Update confidence calibration
            if confidence and outcome:
                await self._update_confidence_calibration(confidence, outcome.get("success", False))

            # Update risk assessments
            await self._update_risk_assessments(interaction_data)

            # Store learning record
            await self._store_learning_record(interaction_data)

            logger.info(f"Learned from interaction for {symbol}")

        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")

    async def _get_total_interactions(self) -> int:
        """Get total number of interactions"""
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute(
                    "SELECT COUNT(*) FROM conversation_history WHERE message_type = 'query'"
                )
            else:
                cursor.execute(
                    "SELECT COUNT(*) FROM conversation_history WHERE message_type = 'query'"
                )

            result = cursor.fetchone()
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Failed to get total interactions: {e}")
            return 0
        finally:
            cursor.close()

    async def _get_success_patterns(self) -> List[KnowledgePattern]:
        """Extract patterns from successful decisions"""
        cursor = self.db.db_conn.cursor()
        try:
            # Get successful decisions with their contexts
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT td.algorithm, td.confidence, td.reasoning, td.symbol,
                           ai.recommendation, ai.confidence as ai_confidence,
                           do.profit_loss, do.market_conditions
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    LEFT JOIN ai_insights ai ON td.parent_insight_id = ai.id
                    WHERE do.success = 1
                    ORDER BY do.profit_loss DESC
                    LIMIT 50
                """
                )
            else:
                cursor.execute(
                    """
                    SELECT td.algorithm, td.confidence, td.reasoning, td.symbol,
                           ai.recommendation, ai.confidence as ai_confidence,
                           do.profit_loss, do.market_conditions
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    LEFT JOIN ai_insights ai ON td.parent_insight_id = ai.id
                    WHERE do.success = true
                    ORDER BY do.profit_loss DESC
                    LIMIT 50
                """
                )

            patterns = []
            algorithm_success = defaultdict(list)
            confidence_success = defaultdict(list)
            ai_agreement_success = []

            for row in cursor.fetchall():
                algorithm, confidence, reasoning, symbol, ai_rec, ai_conf, profit, market_str = row

                # Group by algorithm
                algorithm_success[algorithm].append(profit)

                # Group by confidence level
                conf_level = (
                    "high"
                    if confidence and confidence > 0.8
                    else "medium"
                    if confidence and confidence > 0.6
                    else "low"
                )
                confidence_success[conf_level].append(profit)

                # Track AI agreement
                if ai_rec and reasoning:
                    ai_agreement_success.append(
                        {
                            "ai_recommendation": ai_rec,
                            "ai_confidence": ai_conf,
                            "mcts_confidence": confidence,
                            "profit": profit,
                        }
                    )

            # Create algorithm patterns
            for algorithm, profits in algorithm_success.items():
                if len(profits) >= 3:  # Minimum sample size
                    patterns.append(
                        KnowledgePattern(
                            pattern_id=f"success_algorithm_{algorithm}",
                            pattern_type="algorithm_performance",
                            description=f"{algorithm} algorithm shows consistent success",
                            confidence=min(
                                0.95, len(profits) / 10
                            ),  # Confidence based on sample size
                            success_rate=1.0,  # These are all successful
                            sample_size=len(profits),
                            conditions={"algorithm": algorithm},
                            outcomes=[
                                {"avg_profit": statistics.mean(profits), "max_profit": max(profits)}
                            ],
                            last_updated=datetime.utcnow(),
                        )
                    )

            # Create confidence patterns
            for conf_level, profits in confidence_success.items():
                if len(profits) >= 3:
                    patterns.append(
                        KnowledgePattern(
                            pattern_id=f"success_confidence_{conf_level}",
                            pattern_type="confidence_level",
                            description=f"{conf_level} confidence decisions show success pattern",
                            confidence=min(0.9, len(profits) / 15),
                            success_rate=1.0,
                            sample_size=len(profits),
                            conditions={"confidence_level": conf_level},
                            outcomes=[{"avg_profit": statistics.mean(profits)}],
                            last_updated=datetime.utcnow(),
                        )
                    )

            return patterns

        except Exception as e:
            logger.error(f"Failed to get success patterns: {e}")
            return []
        finally:
            cursor.close()

    async def _get_failure_patterns(self) -> List[KnowledgePattern]:
        """Extract patterns from failed decisions"""
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT td.algorithm, td.confidence, td.reasoning, td.symbol,
                           ai.recommendation, ai.confidence as ai_confidence,
                           do.profit_loss, do.lessons_learned
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    LEFT JOIN ai_insights ai ON td.parent_insight_id = ai.id
                    WHERE do.success = 0
                    ORDER BY do.profit_loss ASC
                    LIMIT 30
                """
                )
            else:
                cursor.execute(
                    """
                    SELECT td.algorithm, td.confidence, td.reasoning, td.symbol,
                           ai.recommendation, ai.confidence as ai_confidence,
                           do.profit_loss, do.lessons_learned
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    LEFT JOIN ai_insights ai ON td.parent_insight_id = ai.id
                    WHERE do.success = false
                    ORDER BY do.profit_loss ASC
                    LIMIT 30
                """
                )

            patterns = []
            failure_reasons = defaultdict(list)

            for row in cursor.fetchall():
                algorithm, confidence, reasoning, symbol, ai_rec, ai_conf, loss, lessons = row

                # Extract failure reasons from lessons
                if lessons:
                    failure_reasons[algorithm].append(
                        {"loss": loss, "lessons": lessons, "confidence": confidence}
                    )

            # Create failure patterns
            for algorithm, failures in failure_reasons.items():
                if len(failures) >= 2:
                    avg_loss = statistics.mean([f["loss"] for f in failures])
                    common_lessons = [f["lessons"] for f in failures]

                    patterns.append(
                        KnowledgePattern(
                            pattern_id=f"failure_algorithm_{algorithm}",
                            pattern_type="algorithm_failure",
                            description=f"{algorithm} algorithm failure pattern identified",
                            confidence=min(0.8, len(failures) / 5),
                            success_rate=0.0,
                            sample_size=len(failures),
                            conditions={"algorithm": algorithm},
                            outcomes=[
                                {
                                    "avg_loss": avg_loss,
                                    "common_lessons": common_lessons[:3],  # Top 3 lessons
                                }
                            ],
                            last_updated=datetime.utcnow(),
                        )
                    )

            return patterns

        except Exception as e:
            logger.error(f"Failed to get failure patterns: {e}")
            return []
        finally:
            cursor.close()

    async def _get_market_insights(self) -> Dict[str, Any]:
        """Get accumulated market insights by symbol"""
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT symbol, recommendation, confidence, reasoning, score, risk_level
                    FROM ai_insights
                    WHERE created_at > date('now', '-30 days')
                    ORDER BY confidence DESC
                """
                )
            else:
                cursor.execute(
                    """
                    SELECT symbol, recommendation, confidence, reasoning, score, risk_level
                    FROM ai_insights
                    WHERE created_at > NOW() - INTERVAL '30 days'
                    ORDER BY confidence DESC
                """
                )

            symbol_insights = defaultdict(list)

            for row in cursor.fetchall():
                symbol, recommendation, confidence, reasoning, score, risk_level = row
                symbol_insights[symbol].append(
                    {
                        "recommendation": recommendation,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "score": score,
                        "risk_level": risk_level,
                    }
                )

            # Aggregate insights per symbol
            market_insights = {}
            for symbol, insights in symbol_insights.items():
                if len(insights) >= 2:
                    recommendations = [i["recommendation"] for i in insights]
                    rec_counter = Counter(recommendations)
                    most_common_rec = rec_counter.most_common(1)[0]

                    avg_confidence = statistics.mean(
                        [i["confidence"] for i in insights if i["confidence"]]
                    )
                    avg_score = statistics.mean([i["score"] for i in insights if i["score"]])

                    market_insights[symbol] = {
                        "most_common_recommendation": most_common_rec[0],
                        "recommendation_frequency": most_common_rec[1],
                        "avg_confidence": avg_confidence,
                        "avg_score": avg_score,
                        "total_analyses": len(insights),
                        "recent_reasoning": insights[0]["reasoning"],  # Most recent
                    }

            return market_insights

        except Exception as e:
            logger.error(f"Failed to get market insights: {e}")
            return {}
        finally:
            cursor.close()

    async def _get_agent_performance(self) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics for different agents/algorithms"""
        try:
            # Get overall performance
            overall_metrics = await self.audit_trail.get_performance_metrics()

            # Get performance by algorithm
            patterns = await self.audit_trail.get_decision_patterns()
            algorithm_performance = patterns.get("algorithm_performance", {})

            performance = {"overall": overall_metrics}

            # Convert algorithm performance to PerformanceMetrics format
            for algorithm, perf in algorithm_performance.items():
                if perf["total_decisions"] >= 3:
                    performance[algorithm] = PerformanceMetrics(
                        total_decisions=perf["total_decisions"],
                        successful_decisions=int(perf["success_rate"] * perf["total_decisions"]),
                        success_rate=perf["success_rate"],
                        total_profit_loss=perf["avg_profit_loss"] * perf["total_decisions"],
                        avg_profit_per_decision=perf["avg_profit_loss"],
                        best_decision_profit=0,  # Not available from patterns
                        worst_decision_loss=0,  # Not available from patterns
                        avg_confidence_successful=0,  # Not available from patterns
                        avg_confidence_failed=0,  # Not available from patterns
                    )

            return performance

        except Exception as e:
            logger.error(f"Failed to get agent performance: {e}")
            return {}

    async def _get_learned_strategies(self) -> List[Dict[str, Any]]:
        """Get learned trading strategies"""
        try:
            # Search memory for strategy learnings
            strategy_memories = await self.memory.search(
                query="strategy",
                memory_types=["learning", "strategy"],
                min_importance=0.7,
                limit=10,
            )

            strategies = []
            for memory in strategy_memories:
                if isinstance(memory.value, dict) and "strategy" in str(memory.value).lower():
                    strategies.append(
                        {
                            "strategy_name": memory.key,
                            "description": memory.context,
                            "confidence": memory.importance,
                            "learned_at": memory.created_at,
                            "details": memory.value,
                        }
                    )

            return strategies

        except Exception as e:
            logger.error(f"Failed to get learned strategies: {e}")
            return []

    async def _get_risk_assessments(self) -> Dict[str, Any]:
        """Get accumulated risk assessment knowledge"""
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT risk_level, COUNT(*) as count,
                           AVG(CASE WHEN td.id IS NOT NULL AND do.success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM ai_insights ai
                    LEFT JOIN trading_decisions td ON ai.id = td.parent_insight_id
                    LEFT JOIN decision_outcomes do ON td.id = do.decision_id
                    WHERE ai.created_at > date('now', '-30 days')
                    GROUP BY risk_level
                """
                )
            else:
                cursor.execute(
                    """
                    SELECT risk_level, COUNT(*) as count,
                           AVG(CASE WHEN td.id IS NOT NULL AND do.success = true THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM ai_insights ai
                    LEFT JOIN trading_decisions td ON ai.id = td.parent_insight_id
                    LEFT JOIN decision_outcomes do ON td.id = do.decision_id
                    WHERE ai.created_at > NOW() - INTERVAL '30 days'
                    GROUP BY risk_level
                """
                )

            risk_assessments = {}
            for row in cursor.fetchall():
                risk_level, count, success_rate = row
                if risk_level:
                    risk_assessments[risk_level] = {
                        "frequency": count,
                        "success_rate": success_rate or 0,
                        "reliability": min(1.0, count / 10),  # More reliable with more samples
                    }

            return risk_assessments

        except Exception as e:
            logger.error(f"Failed to get risk assessments: {e}")
            return {}
        finally:
            cursor.close()

    async def _get_confidence_calibration(self) -> Dict[str, float]:
        """Get confidence calibration data"""
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute(
                    """
                    SELECT 
                        CASE 
                            WHEN td.confidence >= 0.9 THEN '0.9+'
                            WHEN td.confidence >= 0.8 THEN '0.8-0.9'
                            WHEN td.confidence >= 0.7 THEN '0.7-0.8'
                            WHEN td.confidence >= 0.6 THEN '0.6-0.7'
                            ELSE '0.6-'
                        END as confidence_range,
                        AVG(CASE WHEN do.success = 1 THEN 1.0 ELSE 0.0 END) as actual_success_rate,
                        COUNT(*) as count
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    WHERE td.confidence IS NOT NULL
                    GROUP BY confidence_range
                """
                )
            else:
                cursor.execute(
                    """
                    SELECT 
                        CASE 
                            WHEN td.confidence >= 0.9 THEN '0.9+'
                            WHEN td.confidence >= 0.8 THEN '0.8-0.9'
                            WHEN td.confidence >= 0.7 THEN '0.7-0.8'
                            WHEN td.confidence >= 0.6 THEN '0.6-0.7'
                            ELSE '0.6-'
                        END as confidence_range,
                        AVG(CASE WHEN do.success = true THEN 1.0 ELSE 0.0 END) as actual_success_rate,
                        COUNT(*) as count
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    WHERE td.confidence IS NOT NULL
                    GROUP BY confidence_range
                """
                )

            calibration = {}
            for row in cursor.fetchall():
                confidence_range, success_rate, count = row
                if count >= 2:  # Minimum sample size
                    calibration[confidence_range] = success_rate

            return calibration

        except Exception as e:
            logger.error(f"Failed to get confidence calibration: {e}")
            return {}
        finally:
            cursor.close()

    async def _store_knowledge_snapshot(self, knowledge: AccumulatedKnowledge):
        """Store knowledge snapshot for reference"""
        await self.memory.store(
            key=f"knowledge_snapshot_{knowledge.session_id}",
            value=asdict(knowledge),
            memory_type="knowledge_snapshot",
            importance=1.0,
            context=f"Knowledge snapshot for session {knowledge.session_id}",
            metadata={
                "total_interactions": knowledge.total_interactions,
                "success_patterns": len(knowledge.success_patterns),
                "failure_patterns": len(knowledge.failure_patterns),
            },
        )

    # Additional helper methods for updating patterns...
    async def _update_success_failure_patterns(self, interaction_data: Dict[str, Any]):
        """Update success/failure patterns based on new interaction"""
        # Implementation would analyze the interaction and update existing patterns
        pass

    async def _update_market_insights(self, symbol: str, ai_insights: List[Any]):
        """Update market insights for a symbol"""
        # Implementation would aggregate new insights with existing ones
        pass

    async def _update_algorithm_performance(
        self, mcts_decision: Dict[str, Any], outcome: Optional[Dict[str, Any]]
    ):
        """Update algorithm performance metrics"""
        # Implementation would track algorithm performance over time
        pass

    async def _update_confidence_calibration(self, confidence: float, success: bool):
        """Update confidence calibration data"""
        # Implementation would update confidence vs actual success correlation
        pass

    async def _update_risk_assessments(self, interaction_data: Dict[str, Any]):
        """Update risk assessment knowledge"""
        # Implementation would update risk prediction accuracy
        pass

    async def _store_learning_record(self, interaction_data: Dict[str, Any]):
        """Store a record of what was learned from this interaction"""
        await self.memory.store(
            key=f"learning_{datetime.utcnow().isoformat()}",
            value=interaction_data,
            memory_type="learning",
            importance=0.8,
            context="Learning from interaction",
            metadata={"type": "interaction_learning"},
        )

    async def _load_existing_patterns(self):
        """Load existing knowledge patterns into cache"""
        try:
            pattern_memories = await self.memory.search(
                query="pattern", memory_types=["knowledge_pattern"], limit=50
            )

            for memory in pattern_memories:
                if isinstance(memory.value, dict):
                    self._patterns_cache[memory.key] = memory.value

            logger.info(f"Loaded {len(self._patterns_cache)} existing patterns")

        except Exception as e:
            logger.error(f"Failed to load existing patterns: {e}")


# Global knowledge accumulator
_knowledge_accumulator: Optional[KnowledgeAccumulator] = None


async def get_knowledge_accumulator() -> KnowledgeAccumulator:
    """Get global knowledge accumulator instance"""
    global _knowledge_accumulator
    if _knowledge_accumulator is None:
        _knowledge_accumulator = KnowledgeAccumulator()
        await _knowledge_accumulator.initialize()
    return _knowledge_accumulator
