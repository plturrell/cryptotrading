"""
Query Optimizer and Performance Monitoring
Provides query analysis, optimization suggestions, and performance tracking
"""

import hashlib
import json
import logging
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Valid identifier pattern for SQL objects (tables, columns)
SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_sql_identifier(identifier: str) -> str:
    """Validate and sanitize SQL identifier to prevent injection"""
    if not identifier or not SQL_IDENTIFIER_PATTERN.match(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    return identifier


class QueryOptimizer:
    """Query optimization and performance monitoring"""

    def __init__(self, db_client):
        self.db_client = db_client
        self.slow_query_threshold_ms = 1000  # 1 second
        self.query_cache = {}
        self.query_stats = {}

    @contextmanager
    def monitor_query(self, query: str, params: Optional[Dict] = None):
        """Context manager to monitor query performance"""
        query_hash = self._hash_query(query)
        start_time = time.time()

        try:
            yield
        finally:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._log_query_performance(query, query_hash, execution_time_ms, params)

            # Alert on slow queries
            if execution_time_ms > self.slow_query_threshold_ms:
                logger.warning(f"Slow query detected ({execution_time_ms}ms): {query[:100]}...")
                self._analyze_slow_query(query, execution_time_ms)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query and provide optimization suggestions"""
        analysis = {
            "query": query,
            "issues": [],
            "suggestions": [],
            "estimated_cost": None,
            "index_usage": [],
        }

        # Check for missing indexes
        missing_indexes = self._check_missing_indexes(query)
        if missing_indexes:
            analysis["issues"].append("Missing indexes detected")
            analysis["suggestions"].extend(missing_indexes)

        # Check for SELECT *
        if re.search(r"SELECT\s+\*", query, re.IGNORECASE):
            analysis["issues"].append("SELECT * detected")
            analysis["suggestions"].append("Specify only required columns instead of SELECT *")

        # Check for missing WHERE clause in UPDATE/DELETE
        if re.search(r"(UPDATE|DELETE)\s+FROM\s+\w+\s*;", query, re.IGNORECASE):
            analysis["issues"].append("UPDATE/DELETE without WHERE clause")
            analysis["suggestions"].append("Add WHERE clause to prevent full table operations")

        # Check for LIKE with leading wildcard
        if re.search(r"LIKE\s+'%[^']+", query, re.IGNORECASE):
            analysis["issues"].append("LIKE with leading wildcard")
            analysis["suggestions"].append(
                "Leading wildcards prevent index usage, consider full-text search"
            )

        # Check for OR conditions
        or_count = len(re.findall(r"\sOR\s", query, re.IGNORECASE))
        if or_count > 3:
            analysis["issues"].append(f"Multiple OR conditions ({or_count})")
            analysis["suggestions"].append(
                "Consider using IN clause or UNION for better performance"
            )

        # Get query plan
        if self.db_client.is_postgres:
            analysis["query_plan"] = self._get_postgres_query_plan(query)
        else:
            analysis["query_plan"] = self._get_sqlite_query_plan(query)

        return analysis

    def _check_missing_indexes(self, query: str) -> List[str]:
        """Check for potentially missing indexes"""
        suggestions = []

        # Extract table and column references from WHERE, JOIN, and ORDER BY
        where_pattern = r"WHERE\s+(\w+)\.?(\w+)\s*[=<>]"
        join_pattern = r"JOIN\s+\w+\s+ON\s+(\w+)\.?(\w+)\s*=\s*(\w+)\.?(\w+)"
        order_pattern = r"ORDER\s+BY\s+(\w+)\.?(\w+)"

        # Check WHERE conditions
        for match in re.finditer(where_pattern, query, re.IGNORECASE):
            try:
                table = match.group(1) if "." in match.group(0) else self._infer_table(query)
                column = match.group(2) if "." in match.group(0) else match.group(1)
                # Validate identifiers
                table = validate_sql_identifier(table)
                column = validate_sql_identifier(column)
                if not self._has_index(table, column):
                    suggestions.append(f"CREATE INDEX idx_{table}_{column} ON {table}({column})")
            except ValueError:
                continue  # Skip invalid identifiers

        # Check JOIN conditions
        for match in re.finditer(join_pattern, query, re.IGNORECASE):
            for i in [0, 2]:
                try:
                    table = validate_sql_identifier(match.group(i + 1))
                    column = validate_sql_identifier(match.group(i + 2))
                    if not self._has_index(table, column):
                        suggestions.append(
                            f"CREATE INDEX idx_{table}_{column} ON {table}({column})"
                        )
                except ValueError:
                    continue  # Skip invalid identifiers

        # Check ORDER BY
        for match in re.finditer(order_pattern, query, re.IGNORECASE):
            try:
                table = match.group(1) if "." in match.group(0) else self._infer_table(query)
                column = match.group(2) if "." in match.group(0) else match.group(1)
                # Validate identifiers
                table = validate_sql_identifier(table)
                column = validate_sql_identifier(column)
                if not self._has_index(table, column):
                    suggestions.append(f"Consider index on {table}({column}) for ORDER BY")
            except ValueError:
                continue  # Skip invalid identifiers

        return list(set(suggestions))  # Remove duplicates

    def _has_index(self, table: str, column: str) -> bool:
        """Check if an index exists for table.column"""
        if self.db_client.is_sqlite:
            query = (
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql LIKE ?"
            )
            params = (table, f"%{column}%")
        else:
            query = """
            SELECT indexname FROM pg_indexes 
            WHERE tablename = %s AND indexdef LIKE %s
            """
            params = (table, f"%{column}%")

        try:
            with self.db_client.engine.connect() as conn:
                result = conn.execute(query, params)
                return len(list(result)) > 0
        except Exception as e:
            logger.warning(f"Query execution failed during optimization check: {e}")
            return False

    def _infer_table(self, query: str) -> str:
        """Infer table name from query"""
        from_match = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE)
        return from_match.group(1) if from_match else "unknown"

    def _get_postgres_query_plan(self, query: str) -> Dict[str, Any]:
        """Get PostgreSQL query execution plan"""
        try:
            # Only allow SELECT queries for EXPLAIN
            if not query.strip().upper().startswith("SELECT"):
                return {"error": "Only SELECT queries can be explained"}

            # Use parameterized query for EXPLAIN
            with self.db_client.engine.connect() as conn:
                result = conn.execute("EXPLAIN (ANALYZE false, FORMAT JSON) " + query)
                plan = result.fetchone()[0]
                return self._parse_postgres_plan(plan)
        except Exception as e:
            logger.error(f"Failed to get query plan: {e}")
            return {}

    def _get_sqlite_query_plan(self, query: str) -> Dict[str, Any]:
        """Get SQLite query execution plan"""
        try:
            # Only allow SELECT queries for EXPLAIN
            if not query.strip().upper().startswith("SELECT"):
                return {"error": "Only SELECT queries can be explained"}

            with self.db_client.engine.connect() as conn:
                result = conn.execute("EXPLAIN QUERY PLAN " + query)
                rows = result.fetchall()
                return self._parse_sqlite_plan(rows)
        except Exception as e:
            logger.error(f"Failed to get query plan: {e}")
            return {}

    def _parse_postgres_plan(self, plan: List[Dict]) -> Dict[str, Any]:
        """Parse PostgreSQL query plan"""
        if not plan or len(plan) == 0:
            return {}

        root = plan[0]["Plan"]
        return {
            "type": root.get("Node Type"),
            "cost": root.get("Total Cost"),
            "rows": root.get("Plan Rows"),
            "width": root.get("Plan Width"),
            "uses_index": "Index" in str(plan),
        }

    def _parse_sqlite_plan(self, rows: List) -> Dict[str, Any]:
        """Parse SQLite query plan"""
        plan_text = "\n".join(str(row) for row in rows)
        return {
            "plan": plan_text,
            "uses_index": "USING INDEX" in plan_text,
            "full_scan": "SCAN TABLE" in plan_text,
        }

    def _hash_query(self, query: str) -> str:
        """Generate hash for query"""
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _log_query_performance(
        self, query: str, query_hash: str, execution_time_ms: int, params: Optional[Dict] = None
    ):
        """Log query performance metrics"""
        try:
            # Update in-memory stats
            if query_hash not in self.query_stats:
                self.query_stats[query_hash] = {
                    "count": 0,
                    "total_time_ms": 0,
                    "min_time_ms": float("inf"),
                    "max_time_ms": 0,
                    "avg_time_ms": 0,
                }

            stats = self.query_stats[query_hash]
            stats["count"] += 1
            stats["total_time_ms"] += execution_time_ms
            stats["min_time_ms"] = min(stats["min_time_ms"], execution_time_ms)
            stats["max_time_ms"] = max(stats["max_time_ms"], execution_time_ms)
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["count"]

            # Log to database if query is slow or frequently used
            if execution_time_ms > self.slow_query_threshold_ms or stats["count"] % 100 == 0:
                insert_sql = """
                INSERT INTO query_performance_log 
                (query_hash, query_text, execution_time_ms) 
                VALUES (?, ?, ?)
                """

                with self.db_client.engine.connect() as conn:
                    if self.db_client.is_postgres:
                        insert_sql = insert_sql.replace("?", "%s")
                    conn.execute(insert_sql, (query_hash, query[:1000], execution_time_ms))
                    conn.commit()

        except Exception as e:
            logger.error(f"Failed to log query performance: {e}")

    def _analyze_slow_query(self, query: str, execution_time_ms: int):
        """Analyze and log slow query with suggestions"""
        analysis = self.analyze_query(query)

        logger.warning(
            f"""
        Slow Query Analysis:
        Query: {query[:200]}...
        Execution Time: {execution_time_ms}ms
        Issues: {', '.join(analysis['issues'])}
        Suggestions: {', '.join(analysis['suggestions'][:3])}
        """
        )

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get query performance report for the last N hours"""
        since = datetime.now() - timedelta(hours=hours)

        report = {
            "period_hours": hours,
            "total_queries": sum(s["count"] for s in self.query_stats.values()),
            "unique_queries": len(self.query_stats),
            "slow_queries": [],
            "frequent_queries": [],
            "optimization_opportunities": [],
        }

        # Get slow queries from database
        slow_query_sql = """
        SELECT query_hash, query_text, 
               COUNT(*) as count,
               AVG(execution_time_ms) as avg_time_ms,
               MAX(execution_time_ms) as max_time_ms
        FROM query_performance_log
        WHERE created_at >= ? AND execution_time_ms > ?
        GROUP BY query_hash, query_text
        ORDER BY avg_time_ms DESC
        LIMIT 10
        """

        try:
            with self.db_client.engine.connect() as conn:
                if self.db_client.is_postgres:
                    slow_query_sql = slow_query_sql.replace("?", "%s")
                result = conn.execute(slow_query_sql, (since, self.slow_query_threshold_ms))

                for row in result:
                    report["slow_queries"].append(
                        {
                            "query": row["query_text"],
                            "count": row["count"],
                            "avg_time_ms": row["avg_time_ms"],
                            "max_time_ms": row["max_time_ms"],
                        }
                    )
        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")

        # Get frequently executed queries
        for query_hash, stats in sorted(
            self.query_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )[:10]:
            if stats["count"] > 100:
                report["frequent_queries"].append(
                    {
                        "query_hash": query_hash,
                        "count": stats["count"],
                        "avg_time_ms": stats["avg_time_ms"],
                    }
                )

        # Identify optimization opportunities
        for slow_query in report["slow_queries"]:
            analysis = self.analyze_query(slow_query["query"])
            if analysis["issues"]:
                report["optimization_opportunities"].append(
                    {
                        "query": slow_query["query"][:100] + "...",
                        "issues": analysis["issues"],
                        "suggestions": analysis["suggestions"][:2],
                        "potential_improvement": "High"
                        if len(analysis["issues"]) > 2
                        else "Medium",
                    }
                )

        return report


def optimize_query(func):
    """Decorator to automatically optimize and monitor queries"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get query from function
        query = args[0] if args else kwargs.get("query", "")

        # Skip if not a SELECT query
        if not query or not query.strip().upper().startswith("SELECT"):
            return func(self, *args, **kwargs)

        # Check if we have optimizer
        if hasattr(self, "query_optimizer"):
            with self.query_optimizer.monitor_query(query):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return wrapper
