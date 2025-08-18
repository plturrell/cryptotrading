"""
Decision Audit Trail System
Tracks trading decision outcomes and learns from results
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from ...data.database.client import get_db
from ...data.database.intelligence_schema import DecisionStatus

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Types of decision outcomes"""
    PROFIT = "profit"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    CANCELLED = "cancelled"
    PARTIAL_FILL = "partial_fill"


@dataclass
class DecisionOutcome:
    """Result of a trading decision"""
    decision_id: int
    outcome_type: OutcomeType
    expected_outcome: float
    actual_outcome: float
    profit_loss: float
    execution_price: float
    execution_time: datetime
    market_conditions: Dict[str, Any]
    success: bool
    lessons_learned: str
    
    
@dataclass
class PerformanceMetrics:
    """Performance metrics for decisions"""
    total_decisions: int
    successful_decisions: int
    success_rate: float
    total_profit_loss: float
    avg_profit_per_decision: float
    best_decision_profit: float
    worst_decision_loss: float
    avg_confidence_successful: float
    avg_confidence_failed: float


class DecisionAuditTrail:
    """
    Audit trail system for tracking decision outcomes and learning
    """
    
    def __init__(self):
        # Use unified database instead of legacy get_db
        from ...infrastructure.database.unified_database import UnifiedDatabase
        self.db = UnifiedDatabase()
        
    async def record_decision_outcome(self, decision_id: int, 
                                    execution_result: Dict[str, Any]) -> bool:
        """
        Record the outcome of a trading decision
        
        Args:
            decision_id: ID of the decision from trading_decisions table
            execution_result: Result from trade execution
        """
        try:
            # Calculate outcome metrics
            expected_price = execution_result.get('expected_price', 0)
            actual_price = execution_result.get('execution_price', 0)
            amount = execution_result.get('amount', 0)
            
            expected_outcome = expected_price * amount
            actual_outcome = actual_price * amount
            profit_loss = actual_outcome - expected_outcome
            
            # Determine outcome type
            if profit_loss > 0.01:  # Small threshold for floating point
                outcome_type = OutcomeType.PROFIT
                success = True
            elif profit_loss < -0.01:
                outcome_type = OutcomeType.LOSS
                success = False
            else:
                outcome_type = OutcomeType.BREAKEVEN
                success = True
            
            # Generate lessons learned
            lessons = await self._analyze_decision_lessons(
                decision_id, profit_loss, execution_result
            )
            
            # Store outcome
            outcome = DecisionOutcome(
                decision_id=decision_id,
                outcome_type=outcome_type,
                expected_outcome=expected_outcome,
                actual_outcome=actual_outcome,
                profit_loss=profit_loss,
                execution_price=actual_price,
                execution_time=execution_result.get('execution_time', datetime.utcnow()),
                market_conditions=execution_result.get('market_conditions', {}),
                success=success,
                lessons_learned=lessons
            )
            
            await self._store_outcome(outcome)
            
            # Update decision status
            await self._update_decision_status(
                decision_id, 
                DecisionStatus.SUCCESSFUL if success else DecisionStatus.FAILED
            )
            
            logger.info(f"Recorded outcome for decision {decision_id}: {outcome_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record decision outcome: {e}")
            return False
    
    async def _store_outcome(self, outcome: DecisionOutcome):
        """Store decision outcome in database"""
        cursor = self.db.db_conn.cursor()
        try:
            market_conditions_json = json.dumps(outcome.market_conditions)
            
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    INSERT INTO decision_outcomes 
                    (decision_id, outcome_type, expected_outcome, actual_outcome,
                     profit_loss, execution_price, execution_time, market_conditions,
                     success, lessons_learned, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    outcome.decision_id, outcome.outcome_type.value,
                    outcome.expected_outcome, outcome.actual_outcome,
                    outcome.profit_loss, outcome.execution_price,
                    outcome.execution_time, market_conditions_json,
                    outcome.success, outcome.lessons_learned, datetime.utcnow()
                ))
            else:
                cursor.execute("""
                    INSERT INTO decision_outcomes 
                    (decision_id, outcome_type, expected_outcome, actual_outcome,
                     profit_loss, execution_price, execution_time, market_conditions,
                     success, lessons_learned)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    outcome.decision_id, outcome.outcome_type.value,
                    outcome.expected_outcome, outcome.actual_outcome,
                    outcome.profit_loss, outcome.execution_price,
                    outcome.execution_time, market_conditions_json,
                    outcome.success, outcome.lessons_learned
                ))
            
            self.db.db_conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store outcome: {e}")
            self.db.db_conn.rollback()
            raise
        finally:
            cursor.close()
    
    async def _update_decision_status(self, decision_id: int, status: DecisionStatus):
        """Update decision status"""
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    UPDATE trading_decisions 
                    SET status = ?, executed_at = ?
                    WHERE id = ?
                """, (status.value, datetime.utcnow(), decision_id))
            else:
                cursor.execute("""
                    UPDATE trading_decisions 
                    SET status = %s, executed_at = %s
                    WHERE id = %s
                """, (status.value, datetime.utcnow(), decision_id))
            
            self.db.db_conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update decision status: {e}")
            self.db.db_conn.rollback()
        finally:
            cursor.close()
    
    async def _analyze_decision_lessons(self, decision_id: int, 
                                      profit_loss: float,
                                      execution_result: Dict[str, Any]) -> str:
        """Analyze decision and extract lessons learned"""
        try:
            # Get original decision details
            cursor = self.db.db_conn.cursor()
            
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    SELECT td.confidence, td.reasoning, td.algorithm, td.expected_value,
                           ai.recommendation, ai.confidence as ai_confidence, ai.reasoning as ai_reasoning
                    FROM trading_decisions td
                    LEFT JOIN ai_insights ai ON td.parent_insight_id = ai.id
                    WHERE td.id = ?
                """, (decision_id,))
            else:
                cursor.execute("""
                    SELECT td.confidence, td.reasoning, td.algorithm, td.expected_value,
                           ai.recommendation, ai.confidence as ai_confidence, ai.reasoning as ai_reasoning
                    FROM trading_decisions td
                    LEFT JOIN ai_insights ai ON td.parent_insight_id = ai.id
                    WHERE td.id = %s
                """, (decision_id,))
            
            decision_row = cursor.fetchone()
            
            if not decision_row:
                return "No decision details available for analysis"
            
            # Extract decision details
            confidence = decision_row[0] or 0
            reasoning = decision_row[1] or ""
            algorithm = decision_row[2] or ""
            expected_value = decision_row[3] or 0
            ai_recommendation = decision_row[4]
            ai_confidence = decision_row[5] or 0
            ai_reasoning = decision_row[6] or ""
            
            # Analyze lessons
            lessons = []
            
            # Confidence vs outcome analysis
            if profit_loss > 0:
                if confidence > 0.8:
                    lessons.append("High confidence decision was successful - good pattern")
                elif confidence < 0.5:
                    lessons.append("Low confidence decision succeeded - may have been overly cautious")
                
                if ai_confidence and ai_confidence > 0.8:
                    lessons.append("AI was highly confident and correct - trust AI insights")
            else:
                if confidence > 0.8:
                    lessons.append("High confidence decision failed - review decision criteria")
                elif confidence < 0.5:
                    lessons.append("Low confidence decision failed as expected")
                
                if ai_confidence and ai_confidence > 0.8:
                    lessons.append("AI was confident but wrong - check market conditions vs AI analysis")
            
            # Expected vs actual analysis
            if expected_value != 0:
                value_error = abs(profit_loss - expected_value) / abs(expected_value)
                if value_error > 0.5:
                    lessons.append(f"Value prediction was off by {value_error:.1%} - improve value estimation")
            
            # Algorithm performance
            if algorithm:
                if profit_loss > 0:
                    lessons.append(f"{algorithm} algorithm performed well in these conditions")
                else:
                    lessons.append(f"{algorithm} algorithm struggled - analyze market conditions")
            
            # Market condition lessons
            market_volatility = execution_result.get('market_conditions', {}).get('volatility', 0)
            if market_volatility > 0.05:  # High volatility
                if profit_loss > 0:
                    lessons.append("Decision succeeded despite high volatility - good risk management")
                else:
                    lessons.append("High volatility contributed to loss - adjust for volatility")
            
            return "; ".join(lessons) if lessons else "No specific lessons identified"
            
        except Exception as e:
            logger.error(f"Failed to analyze decision lessons: {e}")
            return f"Analysis failed: {str(e)}"
        finally:
            cursor.close()
    
    async def get_performance_metrics(self, symbol: Optional[str] = None,
                                    days: int = 30) -> PerformanceMetrics:
        """Get performance metrics for decisions"""
        # Ensure database is initialized
        if not hasattr(self.db, 'db_conn') or self.db.db_conn is None:
            await self.db.initialize()
        
        cursor = self.db.db_conn.cursor()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Build query
            if symbol:
                if self.db.config.mode.value == "local":
                    cursor.execute("""
                        SELECT td.confidence, do.profit_loss, do.success
                        FROM trading_decisions td
                        JOIN decision_outcomes do ON td.id = do.decision_id
                        WHERE td.symbol = ? AND td.created_at > ?
                    """, (symbol, cutoff_date))
                else:
                    cursor.execute("""
                        SELECT td.confidence, do.profit_loss, do.success
                        FROM trading_decisions td
                        JOIN decision_outcomes do ON td.id = do.decision_id
                        WHERE td.symbol = %s AND td.created_at > %s
                    """, (symbol, cutoff_date))
            else:
                if self.db.config.mode.value == "local":
                    cursor.execute("""
                        SELECT td.confidence, do.profit_loss, do.success
                        FROM trading_decisions td
                        JOIN decision_outcomes do ON td.id = do.decision_id
                        WHERE td.created_at > ?
                    """, (cutoff_date,))
                else:
                    cursor.execute("""
                        SELECT td.confidence, do.profit_loss, do.success
                        FROM trading_decisions td
                        JOIN decision_outcomes do ON td.id = do.decision_id
                        WHERE td.created_at > %s
                    """, (cutoff_date,))
            
            results = cursor.fetchall()
            
            if not results:
                return PerformanceMetrics(
                    total_decisions=0, successful_decisions=0, success_rate=0,
                    total_profit_loss=0, avg_profit_per_decision=0,
                    best_decision_profit=0, worst_decision_loss=0,
                    avg_confidence_successful=0, avg_confidence_failed=0
                )
            
            # Calculate metrics
            total_decisions = len(results)
            successful_decisions = sum(1 for _, _, success in results if success)
            success_rate = successful_decisions / total_decisions
            
            profit_losses = [pl for _, pl, _ in results]
            total_profit_loss = sum(profit_losses)
            avg_profit_per_decision = total_profit_loss / total_decisions
            best_decision_profit = max(profit_losses)
            worst_decision_loss = min(profit_losses)
            
            successful_confidences = [conf for conf, _, success in results if success and conf]
            failed_confidences = [conf for conf, _, success in results if not success and conf]
            
            avg_confidence_successful = (
                sum(successful_confidences) / len(successful_confidences) 
                if successful_confidences else 0
            )
            avg_confidence_failed = (
                sum(failed_confidences) / len(failed_confidences)
                if failed_confidences else 0
            )
            
            return PerformanceMetrics(
                total_decisions=total_decisions,
                successful_decisions=successful_decisions,
                success_rate=success_rate,
                total_profit_loss=total_profit_loss,
                avg_profit_per_decision=avg_profit_per_decision,
                best_decision_profit=best_decision_profit,
                worst_decision_loss=worst_decision_loss,
                avg_confidence_successful=avg_confidence_successful,
                avg_confidence_failed=avg_confidence_failed
            )
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(
                total_decisions=0, successful_decisions=0, success_rate=0,
                total_profit_loss=0, avg_profit_per_decision=0,
                best_decision_profit=0, worst_decision_loss=0,
                avg_confidence_successful=0, avg_confidence_failed=0
            )
        finally:
            cursor.close()
    
    async def get_lessons_learned(self, symbol: Optional[str] = None,
                                days: int = 30) -> List[Dict[str, Any]]:
        """Get lessons learned from recent decisions"""
        cursor = self.db.db_conn.cursor()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            if symbol:
                if self.db.config.mode.value == "local":
                    cursor.execute("""
                        SELECT do.lessons_learned, do.profit_loss, do.success,
                               td.symbol, td.confidence, td.reasoning, do.created_at
                        FROM decision_outcomes do
                        JOIN trading_decisions td ON do.decision_id = td.id
                        WHERE td.symbol = ? AND do.created_at > ?
                        ORDER BY do.created_at DESC
                    """, (symbol, cutoff_date))
                else:
                    cursor.execute("""
                        SELECT do.lessons_learned, do.profit_loss, do.success,
                               td.symbol, td.confidence, td.reasoning, do.created_at
                        FROM decision_outcomes do
                        JOIN trading_decisions td ON do.decision_id = td.id
                        WHERE td.symbol = %s AND do.created_at > %s
                        ORDER BY do.created_at DESC
                    """, (symbol, cutoff_date))
            else:
                if self.db.config.mode.value == "local":
                    cursor.execute("""
                        SELECT do.lessons_learned, do.profit_loss, do.success,
                               td.symbol, td.confidence, td.reasoning, do.created_at
                        FROM decision_outcomes do
                        JOIN trading_decisions td ON do.decision_id = td.id
                        WHERE do.created_at > ?
                        ORDER BY do.created_at DESC
                        LIMIT 20
                    """, (cutoff_date,))
                else:
                    cursor.execute("""
                        SELECT do.lessons_learned, do.profit_loss, do.success,
                               td.symbol, td.confidence, td.reasoning, do.created_at
                        FROM decision_outcomes do
                        JOIN trading_decisions td ON do.decision_id = td.id
                        WHERE do.created_at > %s
                        ORDER BY do.created_at DESC
                        LIMIT 20
                    """, (cutoff_date,))
            
            lessons = []
            for row in cursor.fetchall():
                lessons.append({
                    "lessons_learned": row[0],
                    "profit_loss": row[1],
                    "success": row[2],
                    "symbol": row[3],
                    "confidence": row[4],
                    "reasoning": row[5],
                    "date": row[6]
                })
            
            return lessons
            
        except Exception as e:
            logger.error(f"Failed to get lessons learned: {e}")
            return []
        finally:
            cursor.close()
    
    async def get_decision_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in successful vs failed decisions"""
        cursor = self.db.db_conn.cursor()
        try:
            # Get decision patterns
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    SELECT 
                        td.algorithm,
                        AVG(CASE WHEN do.success THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(do.profit_loss) as avg_profit_loss,
                        COUNT(*) as total_decisions
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    GROUP BY td.algorithm
                    HAVING COUNT(*) >= 3
                """)
            else:
                cursor.execute("""
                    SELECT 
                        td.algorithm,
                        AVG(CASE WHEN do.success THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(do.profit_loss) as avg_profit_loss,
                        COUNT(*) as total_decisions
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    GROUP BY td.algorithm
                    HAVING COUNT(*) >= 3
                """)
            
            algorithm_performance = {}
            for row in cursor.fetchall():
                algorithm_performance[row[0]] = {
                    "success_rate": row[1],
                    "avg_profit_loss": row[2],
                    "total_decisions": row[3]
                }
            
            # Get confidence level patterns
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN td.confidence >= 0.8 THEN 'high'
                            WHEN td.confidence >= 0.6 THEN 'medium'
                            ELSE 'low'
                        END as confidence_level,
                        AVG(CASE WHEN do.success THEN 1.0 ELSE 0.0 END) as success_rate,
                        COUNT(*) as count
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    WHERE td.confidence IS NOT NULL
                    GROUP BY confidence_level
                """)
            else:
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN td.confidence >= 0.8 THEN 'high'
                            WHEN td.confidence >= 0.6 THEN 'medium'
                            ELSE 'low'
                        END as confidence_level,
                        AVG(CASE WHEN do.success THEN 1.0 ELSE 0.0 END) as success_rate,
                        COUNT(*) as count
                    FROM trading_decisions td
                    JOIN decision_outcomes do ON td.id = do.decision_id
                    WHERE td.confidence IS NOT NULL
                    GROUP BY confidence_level
                """)
            
            confidence_patterns = {}
            for row in cursor.fetchall():
                confidence_patterns[row[0]] = {
                    "success_rate": row[1],
                    "count": row[2]
                }
            
            return {
                "algorithm_performance": algorithm_performance,
                "confidence_patterns": confidence_patterns
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze decision patterns: {e}")
            return {}
        finally:
            cursor.close()


# Global audit trail instance
_audit_trail: Optional[DecisionAuditTrail] = None

def get_audit_trail() -> DecisionAuditTrail:
    """Get global audit trail instance"""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = DecisionAuditTrail()
    return _audit_trail