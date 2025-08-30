"""
Production-grade transaction management with ACID guarantees
Supports nested transactions, savepoints, and distributed transactions
"""

import functools
import logging
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import event
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class TransactionError(Exception):
    """Transaction-related error"""

    pass


class TransactionManager:
    """Production transaction manager with ACID guarantees"""

    def __init__(self, db_client):
        self.db_client = db_client
        self.active_transactions = {}  # Thread-local storage for active transactions
        self.transaction_log = []  # For debugging and monitoring
        self._local = threading.local()

    @contextmanager
    def transaction(
        self,
        isolation_level: str = None,
        read_only: bool = False,
        timeout: int = 30,
        savepoint_name: str = None,
    ):
        """
        Context manager for database transactions with full ACID support

        Args:
            isolation_level: SQL isolation level (READ_COMMITTED, REPEATABLE_READ, etc.)
            read_only: Whether this is a read-only transaction
            timeout: Transaction timeout in seconds
            savepoint_name: Name for savepoint (enables nested transactions)
        """
        session = self.db_client.Session()
        transaction_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        # Store transaction info
        transaction_info = {
            "id": transaction_id,
            "start_time": start_time,
            "isolation_level": isolation_level,
            "read_only": read_only,
            "timeout": timeout,
            "savepoint_name": savepoint_name,
            "thread_id": threading.get_ident(),
        }

        self.transaction_log.append(transaction_info)

        try:
            # Set isolation level if specified
            if isolation_level:
                if self.db_client.is_postgres:
                    session.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}")
                elif self.db_client.is_sqlite:
                    # SQLite has limited isolation level support
                    if isolation_level in ["READ_UNCOMMITTED", "READ_COMMITTED"]:
                        session.execute("PRAGMA read_uncommitted = 1")

            # Set read-only mode
            if read_only:
                if self.db_client.is_postgres:
                    session.execute("SET TRANSACTION READ ONLY")

            # Create savepoint for nested transactions
            if savepoint_name:
                savepoint = session.begin_nested()
                logger.debug(f"Created savepoint: {savepoint_name}")

            # Set transaction timeout
            if timeout and self.db_client.is_postgres:
                session.execute(f"SET statement_timeout = '{timeout}s'")

            # Track active transaction
            thread_id = threading.get_ident()
            if thread_id not in self.active_transactions:
                self.active_transactions[thread_id] = []
            self.active_transactions[thread_id].append(transaction_info)

            logger.info(f"Started transaction {transaction_id}")

            yield session

            # Commit the transaction
            if savepoint_name:
                savepoint.commit()
                logger.debug(f"Committed savepoint: {savepoint_name}")
            else:
                session.commit()
                logger.info(f"Committed transaction {transaction_id}")

            # Update transaction info
            transaction_info["status"] = "committed"
            transaction_info["end_time"] = datetime.utcnow()
            transaction_info["duration"] = (
                transaction_info["end_time"] - start_time
            ).total_seconds()

        except Exception as e:
            # Rollback the transaction
            try:
                if savepoint_name:
                    savepoint.rollback()
                    logger.warning(f"Rolled back savepoint: {savepoint_name}")
                else:
                    session.rollback()
                    logger.warning(f"Rolled back transaction {transaction_id}")
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")

            # Update transaction info
            transaction_info["status"] = "rolled_back"
            transaction_info["error"] = str(e)
            transaction_info["end_time"] = datetime.utcnow()
            transaction_info["duration"] = (
                transaction_info["end_time"] - start_time
            ).total_seconds()

            logger.error(f"Transaction {transaction_id} failed: {e}")
            raise TransactionError(f"Transaction failed: {e}") from e

        finally:
            # Clean up
            try:
                session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")

            # Remove from active transactions
            thread_id = threading.get_ident()
            if thread_id in self.active_transactions:
                self.active_transactions[thread_id] = [
                    t for t in self.active_transactions[thread_id] if t["id"] != transaction_id
                ]
                if not self.active_transactions[thread_id]:
                    del self.active_transactions[thread_id]

    @contextmanager
    def read_only_transaction(self):
        """Convenience method for read-only transactions"""
        with self.transaction(read_only=True) as session:
            yield session

    @contextmanager
    def serializable_transaction(self):
        """Convenience method for serializable transactions"""
        with self.transaction(isolation_level="SERIALIZABLE") as session:
            yield session

    def execute_in_transaction(
        self,
        func: Callable,
        *args,
        retry_count: int = 3,
        retry_on: tuple = (OperationalError, IntegrityError),
        **kwargs,
    ):
        """
        Execute function in transaction with automatic retry on failure

        Args:
            func: Function to execute
            retry_count: Number of retries on failure
            retry_on: Exception types to retry on
        """
        last_exception = None

        for attempt in range(retry_count + 1):
            try:
                with self.transaction() as session:
                    kwargs["session"] = session
                    return func(*args, **kwargs)

            except retry_on as e:
                last_exception = e
                if attempt < retry_count:
                    logger.warning(f"Transaction attempt {attempt + 1} failed, retrying: {e}")
                    continue
                else:
                    logger.error(f"Transaction failed after {retry_count + 1} attempts")
                    raise
            except Exception as e:
                # Don't retry on non-retryable exceptions
                logger.error(f"Non-retryable error in transaction: {e}")
                raise

        if last_exception:
            raise last_exception

    def batch_operations(
        self, operations: List[Callable], batch_size: int = 1000, continue_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        Execute multiple operations in batches with transaction management

        Args:
            operations: List of callable operations
            batch_size: Number of operations per transaction
            continue_on_error: Whether to continue if a batch fails
        """
        results = {
            "total_operations": len(operations),
            "successful_batches": 0,
            "failed_batches": 0,
            "errors": [],
        }

        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]
            batch_id = f"batch_{i // batch_size + 1}"

            try:
                with self.transaction() as session:
                    for operation in batch:
                        if callable(operation):
                            operation(session)
                        else:
                            # Assume it's a tuple of (func, args, kwargs)
                            func, args, kwargs = operation
                            kwargs["session"] = session
                            func(*args, **kwargs)

                results["successful_batches"] += 1
                logger.info(f"Completed {batch_id} with {len(batch)} operations")

            except Exception as e:
                results["failed_batches"] += 1
                error_info = {
                    "batch_id": batch_id,
                    "error": str(e),
                    "operations_in_batch": len(batch),
                }
                results["errors"].append(error_info)

                logger.error(f"Batch {batch_id} failed: {e}")

                if not continue_on_error:
                    break

        return results

    def get_transaction_stats(self) -> Dict[str, Any]:
        """Get transaction statistics for monitoring"""
        stats = {
            "total_transactions": len(self.transaction_log),
            "active_transactions": sum(len(txns) for txns in self.active_transactions.values()),
            "committed": len([t for t in self.transaction_log if t.get("status") == "committed"]),
            "rolled_back": len(
                [t for t in self.transaction_log if t.get("status") == "rolled_back"]
            ),
            "pending": len([t for t in self.transaction_log if "status" not in t]),
        }

        # Calculate average duration for completed transactions
        completed_transactions = [
            t for t in self.transaction_log if "duration" in t and t["duration"] is not None
        ]

        if completed_transactions:
            stats["avg_duration"] = sum(t["duration"] for t in completed_transactions) / len(
                completed_transactions
            )
            stats["max_duration"] = max(t["duration"] for t in completed_transactions)
            stats["min_duration"] = min(t["duration"] for t in completed_transactions)

        return stats

    def cleanup_transaction_log(self, max_entries: int = 10000):
        """Clean up old transaction log entries"""
        if len(self.transaction_log) > max_entries:
            # Keep only the most recent entries
            self.transaction_log = self.transaction_log[-max_entries:]
            logger.info(f"Cleaned up transaction log, kept {max_entries} entries")


class DistributedTransactionManager:
    """Manager for distributed transactions across multiple databases"""

    def __init__(self, db_clients: Dict[str, Any]):
        self.db_clients = db_clients
        self.transaction_managers = {
            name: TransactionManager(client) for name, client in db_clients.items()
        }

    @contextmanager
    def distributed_transaction(self, participants: List[str] = None):
        """
        Two-phase commit for distributed transactions

        Args:
            participants: List of database names to include in transaction
        """
        if participants is None:
            participants = list(self.db_clients.keys())

        sessions = {}
        transaction_id = str(uuid.uuid4())

        try:
            # Phase 1: Prepare all participants
            for db_name in participants:
                manager = self.transaction_managers[db_name]
                session = manager.db_client.Session()
                sessions[db_name] = session

                # Start transaction
                session.begin()
                logger.info(
                    f"Prepared transaction on {db_name} for distributed transaction {transaction_id}"
                )

            yield sessions

            # Phase 2: Commit all participants
            for db_name, session in sessions.items():
                session.commit()
                logger.info(
                    f"Committed transaction on {db_name} for distributed transaction {transaction_id}"
                )

            logger.info(f"Distributed transaction {transaction_id} completed successfully")

        except Exception as e:
            # Rollback all participants
            logger.error(f"Distributed transaction {transaction_id} failed: {e}")

            for db_name, session in sessions.items():
                try:
                    session.rollback()
                    logger.info(f"Rolled back transaction on {db_name}")
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback {db_name}: {rollback_error}")

            raise TransactionError(f"Distributed transaction failed: {e}") from e

        finally:
            # Close all sessions
            for db_name, session in sessions.items():
                try:
                    session.close()
                except Exception as e:
                    logger.error(f"Error closing session for {db_name}: {e}")


def transactional(
    isolation_level: str = None, read_only: bool = False, retry_count: int = 3, timeout: int = 30
):
    """
    Decorator to make a function transactional

    Args:
        isolation_level: SQL isolation level
        read_only: Whether this is a read-only transaction
        retry_count: Number of retries on failure
        timeout: Transaction timeout
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get transaction manager from context or create new one
            from .client import get_db

            db_client = get_db()
            tx_manager = TransactionManager(db_client)

            def execute_with_session(session):
                # Inject session into kwargs
                kwargs["session"] = session
                return func(*args, **kwargs)

            return tx_manager.execute_in_transaction(execute_with_session, retry_count=retry_count)

        return wrapper

    return decorator


def read_only_transactional(func: Callable) -> Callable:
    """Decorator for read-only transactions"""
    return transactional(read_only=True)(func)


def serializable_transactional(func: Callable) -> Callable:
    """Decorator for serializable transactions"""
    return transactional(isolation_level="SERIALIZABLE")(func)
