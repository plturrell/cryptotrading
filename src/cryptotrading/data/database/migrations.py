"""
Database Migration System with Version Control
Supports both SQLite and PostgreSQL with automatic migration tracking
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Migration:
    """Represents a database migration"""

    def __init__(
        self, version: str, name: str, up_sql: str, down_sql: str, checksum: Optional[str] = None
    ):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.checksum = checksum or self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum for migration integrity"""
        content = f"{self.version}:{self.name}:{self.up_sql}:{self.down_sql}"
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "checksum": self.checksum,
            "up_sql": self.up_sql,
            "down_sql": self.down_sql,
        }


class DatabaseMigrator:
    """Handles database migrations with version control"""

    def __init__(self, db_client):
        self.db_client = db_client
        self.migrations_table = "schema_migrations"
        self.migrations: List[Migration] = []
        self._init_migrations_table()

    def _init_migrations_table(self):
        """Create migrations tracking table if not exists"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            checksum VARCHAR(64) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time_ms INTEGER,
            rollback_sql TEXT,
            metadata TEXT
        )
        """

        if self.db_client.is_sqlite:
            self.db_client.engine.execute(create_table_sql)
        else:
            with self.db_client.engine.connect() as conn:
                conn.execute(create_table_sql)
                conn.commit()

        logger.info("Migrations table initialized")

    def register_migration(self, migration: Migration):
        """Register a migration"""
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)

    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Get list of applied migrations"""
        query = "SELECT * FROM schema_migrations ORDER BY version"

        with self.db_client.engine.connect() as conn:
            result = conn.execute(query)
            return [dict(row) for row in result]

    def get_pending_migrations(self) -> List[Migration]:
        """Get migrations that haven't been applied yet"""
        applied_versions = {m["version"] for m in self.get_applied_migrations()}
        return [m for m in self.migrations if m.version not in applied_versions]

    def migrate(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """Run migrations up to target version"""
        pending = self.get_pending_migrations()

        if target_version:
            pending = [m for m in pending if m.version <= target_version]

        results = {"applied": [], "failed": [], "total_time_ms": 0}

        for migration in pending:
            start_time = datetime.now()

            try:
                # Execute migration
                self._execute_migration(migration)

                # Record migration
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                self._record_migration(migration, execution_time)

                results["applied"].append(
                    {
                        "version": migration.version,
                        "name": migration.name,
                        "execution_time_ms": execution_time,
                    }
                )
                results["total_time_ms"] += execution_time

                logger.info(f"Applied migration {migration.version}: {migration.name}")

            except Exception as e:
                logger.error(f"Failed to apply migration {migration.version}: {e}")
                results["failed"].append(
                    {"version": migration.version, "name": migration.name, "error": str(e)}
                )
                break  # Stop on first failure

        return results

    def rollback(self, target_version: str) -> Dict[str, Any]:
        """Rollback migrations to target version"""
        applied = self.get_applied_migrations()
        to_rollback = [m for m in applied if m["version"] > target_version]
        to_rollback.reverse()  # Rollback in reverse order

        results = {"rolled_back": [], "failed": [], "total_time_ms": 0}

        for migration_record in to_rollback:
            start_time = datetime.now()

            try:
                # Find migration object
                migration = next(
                    (m for m in self.migrations if m.version == migration_record["version"]), None
                )

                if not migration:
                    raise ValueError(f"Migration {migration_record['version']} not found")

                # Execute rollback
                self._execute_rollback(migration)

                # Remove migration record
                self._remove_migration_record(migration.version)

                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                results["rolled_back"].append(
                    {
                        "version": migration.version,
                        "name": migration.name,
                        "execution_time_ms": execution_time,
                    }
                )
                results["total_time_ms"] += execution_time

                logger.info(f"Rolled back migration {migration.version}: {migration.name}")

            except Exception as e:
                logger.error(f"Failed to rollback migration {migration_record['version']}: {e}")
                results["failed"].append({"version": migration_record["version"], "error": str(e)})
                break

        return results

    def _execute_migration(self, migration: Migration):
        """Execute a migration"""
        with self.db_client.engine.connect() as conn:
            trans = conn.begin()
            try:
                # Split and execute SQL statements
                for statement in migration.up_sql.split(";"):
                    statement = statement.strip()
                    if statement:
                        conn.execute(statement)
                trans.commit()
            except Exception:
                trans.rollback()
                raise

    def _execute_rollback(self, migration: Migration):
        """Execute a rollback"""
        with self.db_client.engine.connect() as conn:
            trans = conn.begin()
            try:
                # Split and execute SQL statements
                for statement in migration.down_sql.split(";"):
                    statement = statement.strip()
                    if statement:
                        conn.execute(statement)
                trans.commit()
            except Exception:
                trans.rollback()
                raise

    def _record_migration(self, migration: Migration, execution_time_ms: int):
        """Record a migration as applied"""
        insert_sql = """
        INSERT INTO schema_migrations 
        (version, name, checksum, execution_time_ms, rollback_sql, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """

        metadata = json.dumps(
            {
                "applied_by": os.getenv("USER", "system"),
                "environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        params = (
            migration.version,
            migration.name,
            migration.checksum,
            execution_time_ms,
            migration.down_sql,
            metadata,
        )

        with self.db_client.engine.connect() as conn:
            if self.db_client.is_postgres:
                # PostgreSQL uses %s for parameters
                insert_sql = insert_sql.replace("?", "%s")
            conn.execute(insert_sql, params)
            conn.commit()

    def _remove_migration_record(self, version: str):
        """Remove a migration record"""
        delete_sql = "DELETE FROM schema_migrations WHERE version = ?"

        with self.db_client.engine.connect() as conn:
            if self.db_client.is_postgres:
                delete_sql = delete_sql.replace("?", "%s")
            conn.execute(delete_sql, (version,))
            conn.commit()

    def validate_migrations(self) -> Dict[str, Any]:
        """Validate migration integrity"""
        applied = self.get_applied_migrations()
        issues = []

        for record in applied:
            migration = next((m for m in self.migrations if m.version == record["version"]), None)

            if not migration:
                issues.append({"version": record["version"], "issue": "Migration file not found"})
            elif migration.checksum != record["checksum"]:
                issues.append(
                    {
                        "version": record["version"],
                        "issue": "Checksum mismatch - migration has been modified",
                    }
                )

        return {"valid": len(issues) == 0, "issues": issues}

    def get_status(self) -> Dict[str, Any]:
        """Get migration status"""
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()

        return {
            "current_version": applied[-1]["version"] if applied else None,
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_migrations": [{"version": m["version"], "name": m["name"]} for m in applied],
            "pending_migrations": [{"version": m.version, "name": m.name} for m in pending],
        }


# Define migrations
def get_migrations() -> List[Migration]:
    """Get all database migrations"""
    migrations = []

    # Migration 001: Add indexes for performance
    migrations.append(
        Migration(
            version="001",
            name="add_performance_indexes",
            up_sql="""
        CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id);
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at);
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_portfolios_user_symbol ON portfolios(user_id, symbol);
        CREATE INDEX IF NOT EXISTS idx_ai_analyses_symbol_created ON ai_analyses(symbol, created_at);
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
        CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_created ON trading_signals(symbol, created_at);
        CREATE INDEX IF NOT EXISTS idx_dex_trades_token_timestamp ON dex_trades(token, timestamp);
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_created ON conversation_messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_memory_fragments_user_type ON memory_fragments(user_id, fragment_type);
        CREATE INDEX IF NOT EXISTS idx_a2a_agents_status ON a2a_agents(status);
        CREATE INDEX IF NOT EXISTS idx_a2a_messages_status_created ON a2a_messages(status, sent_at);
        """,
            down_sql="""
        DROP INDEX IF EXISTS idx_trades_user_id;
        DROP INDEX IF EXISTS idx_trades_symbol;
        DROP INDEX IF EXISTS idx_trades_executed_at;
        DROP INDEX IF EXISTS idx_trades_status;
        DROP INDEX IF EXISTS idx_portfolios_user_symbol;
        DROP INDEX IF EXISTS idx_ai_analyses_symbol_created;
        DROP INDEX IF EXISTS idx_market_data_symbol_timestamp;
        DROP INDEX IF EXISTS idx_trading_signals_symbol_created;
        DROP INDEX IF EXISTS idx_dex_trades_token_timestamp;
        DROP INDEX IF EXISTS idx_conversation_messages_session_created;
        DROP INDEX IF EXISTS idx_memory_fragments_user_type;
        DROP INDEX IF EXISTS idx_a2a_agents_status;
        DROP INDEX IF EXISTS idx_a2a_messages_status_created;
        """,
        )
    )

    # Migration 002: Add constraints and validations
    migrations.append(
        Migration(
            version="002",
            name="add_data_constraints",
            up_sql="""
        CREATE TABLE IF NOT EXISTS trade_validations (
            id INTEGER PRIMARY KEY,
            trade_id INTEGER NOT NULL,
            validation_type VARCHAR(50) NOT NULL,
            validation_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        );
        
        CREATE TABLE IF NOT EXISTS data_quality_metrics (
            id INTEGER PRIMARY KEY,
            table_name VARCHAR(100) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT,
            threshold_value FLOAT,
            is_passing BOOLEAN DEFAULT TRUE,
            measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
            down_sql="""
        DROP TABLE IF EXISTS trade_validations;
        DROP TABLE IF EXISTS data_quality_metrics;
        """,
        )
    )

    # Migration 003: Add performance monitoring tables
    migrations.append(
        Migration(
            version="003",
            name="add_performance_monitoring",
            up_sql="""
        CREATE TABLE IF NOT EXISTS query_performance_log (
            id INTEGER PRIMARY KEY,
            query_hash VARCHAR(64) NOT NULL,
            query_text TEXT NOT NULL,
            execution_time_ms INTEGER NOT NULL,
            rows_affected INTEGER,
            query_plan TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS connection_pool_metrics (
            id INTEGER PRIMARY KEY,
            pool_size INTEGER NOT NULL,
            active_connections INTEGER NOT NULL,
            idle_connections INTEGER NOT NULL,
            wait_count INTEGER DEFAULT 0,
            timeout_count INTEGER DEFAULT 0,
            measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_query_performance_hash ON query_performance_log(query_hash);
        CREATE INDEX IF NOT EXISTS idx_query_performance_time ON query_performance_log(execution_time_ms);
        """,
            down_sql="""
        DROP TABLE IF EXISTS query_performance_log;
        DROP TABLE IF EXISTS connection_pool_metrics;
        """,
        )
    )

    # Migration 004: Add audit and security tables
    migrations.append(
        Migration(
            version="004",
            name="add_audit_security",
            up_sql="""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            action VARCHAR(100) NOT NULL,
            table_name VARCHAR(100),
            record_id INTEGER,
            old_values TEXT,
            new_values TEXT,
            ip_address VARCHAR(45),
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        
        CREATE TABLE IF NOT EXISTS database_health_checks (
            id INTEGER PRIMARY KEY,
            check_name VARCHAR(100) NOT NULL,
            check_status VARCHAR(20) NOT NULL,
            response_time_ms INTEGER,
            error_message TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_log_user_action ON audit_log(user_id, action);
        CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);
        """,
            down_sql="""
        DROP TABLE IF EXISTS audit_log;
        DROP TABLE IF EXISTS database_health_checks;
        """,
        )
    )

    return migrations
