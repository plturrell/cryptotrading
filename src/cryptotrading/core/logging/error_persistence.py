"""
Error Logging Database Persistence
Stores errors and exceptions for monitoring and debugging
"""

import logging
import traceback
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

from ...infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)


class ErrorLogPersistence:
    """
    Persists error logs to database for monitoring and analysis
    Provides centralized error tracking across the system
    """
    
    def __init__(self, db: Optional[UnifiedDatabase] = None):
        self.db = db or UnifiedDatabase()
        self._batch_buffer = defaultdict(list)
        self._batch_size = 50
        self._flush_interval = 60  # seconds
        self._flush_task = None
        
    async def start(self):
        """Start the error logging service"""
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Error logging persistence service started")
        
    async def stop(self):
        """Stop the service and flush remaining logs"""
        if self._flush_task:
            self._flush_task.cancel()
        await self._flush_buffer()
        logger.info("Error logging persistence service stopped")
        
    async def log_error(self,
                       error_type: str,
                       error_message: str,
                       error_details: Optional[Dict[str, Any]] = None,
                       component: Optional[str] = None,
                       severity: str = "error",
                       user_id: Optional[str] = None,
                       session_id: Optional[str] = None):
        """Log a single error to the database"""
        error_log = {
            "error_type": error_type,
            "error_message": error_message[:1000],  # Limit message length
            "error_details": json.dumps(error_details) if error_details else None,
            "stack_trace": traceback.format_exc() if severity == "critical" else None,
            "component": component or "unknown",
            "severity": severity,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.utcnow()
        }
        
        # Add to batch buffer
        self._batch_buffer[severity].append(error_log)
        
        # Flush if buffer is full or critical error
        if len(self._batch_buffer[severity]) >= self._batch_size or severity == "critical":
            await self._flush_buffer(severity)
            
    async def log_exception(self,
                          exception: Exception,
                          component: str,
                          context: Optional[Dict[str, Any]] = None,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None):
        """Log an exception with full stack trace"""
        error_details = {
            "exception_type": type(exception).__name__,
            "exception_args": str(exception.args),
            "context": context
        }
        
        await self.log_error(
            error_type=type(exception).__name__,
            error_message=str(exception),
            error_details=error_details,
            component=component,
            severity="error" if not isinstance(exception, (SystemError, MemoryError)) else "critical",
            user_id=user_id,
            session_id=session_id
        )
        
    async def _flush_buffer(self, severity: Optional[str] = None):
        """Flush error logs to database"""
        try:
            severities_to_flush = [severity] if severity else list(self._batch_buffer.keys())
            
            for sev in severities_to_flush:
                if not self._batch_buffer[sev]:
                    continue
                    
                logs = self._batch_buffer[sev]
                self._batch_buffer[sev] = []
                
                # Batch insert
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    cursor.executemany("""
                        INSERT INTO error_logs 
                        (error_type, error_message, error_details, stack_trace,
                         component, severity, user_id, session_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        (log['error_type'], log['error_message'], log['error_details'],
                         log['stack_trace'], log['component'], log['severity'],
                         log['user_id'], log['session_id'], log['timestamp'])
                        for log in logs
                    ])
                    
                    conn.commit()
                    
                logger.debug(f"Flushed {len(logs)} {sev} error logs to database")
                
        except Exception as e:
            logger.error(f"Failed to flush error logs: {e}")
            # Re-add logs to buffer
            if severity:
                self._batch_buffer[severity].extend(logs)
                
    async def _periodic_flush(self):
        """Periodically flush error logs"""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
                
    async def get_errors(self,
                        component: Optional[str] = None,
                        severity: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Query errors from database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM error_logs WHERE 1=1"
                params = []
                
                if component:
                    query += " AND component = ?"
                    params.append(component)
                    
                if severity:
                    query += " AND severity = ?"
                    params.append(severity)
                    
                if start_time:
                    query += " AND created_at >= ?"
                    params.append(start_time)
                    
                if end_time:
                    query += " AND created_at <= ?"
                    params.append(end_time)
                    
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                errors = []
                for row in cursor.fetchall():
                    errors.append({
                        'id': row[0],
                        'error_type': row[1],
                        'error_message': row[2],
                        'error_details': json.loads(row[3]) if row[3] else {},
                        'stack_trace': row[4],
                        'component': row[5],
                        'severity': row[6],
                        'user_id': row[7],
                        'session_id': row[8],
                        'timestamp': row[9],
                        'resolved': bool(row[10])
                    })
                    
                return errors
                
        except Exception as e:
            logger.error(f"Failed to get errors: {e}")
            return []
            
    async def get_error_statistics(self,
                                 days: int = 7,
                                 group_by: str = "component") -> Dict[str, Any]:
        """Get error statistics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total errors
                cursor.execute("""
                    SELECT COUNT(*) FROM error_logs
                    WHERE created_at >= ?
                """, (start_date,))
                total_errors = cursor.fetchone()[0]
                
                # Errors by severity
                cursor.execute("""
                    SELECT severity, COUNT(*) FROM error_logs
                    WHERE created_at >= ?
                    GROUP BY severity
                """, (start_date,))
                errors_by_severity = dict(cursor.fetchall())
                
                # Errors by group
                cursor.execute(f"""
                    SELECT {group_by}, COUNT(*) FROM error_logs
                    WHERE created_at >= ?
                    GROUP BY {group_by}
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """, (start_date,))
                errors_by_group = dict(cursor.fetchall())
                
                # Error trends (hourly for last 24 hours)
                cursor.execute("""
                    SELECT 
                        strftime('%Y-%m-%d %H:00:00', created_at) as hour,
                        COUNT(*) as count
                    FROM error_logs
                    WHERE created_at >= ?
                    GROUP BY hour
                    ORDER BY hour DESC
                    LIMIT 24
                """, (datetime.utcnow() - timedelta(hours=24),))
                
                hourly_trends = []
                for row in cursor.fetchall():
                    hourly_trends.append({
                        'hour': row[0],
                        'count': row[1]
                    })
                
                return {
                    'period_days': days,
                    'total_errors': total_errors,
                    'errors_by_severity': errors_by_severity,
                    f'errors_by_{group_by}': errors_by_group,
                    'hourly_trends': hourly_trends,
                    'avg_errors_per_day': round(total_errors / days, 2)
                }
                
        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {}
            
    async def mark_resolved(self, error_id: int, resolved_by: Optional[str] = None) -> bool:
        """Mark an error as resolved"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE error_logs 
                    SET resolved = 1, resolved_at = ?, resolved_by = ?
                    WHERE id = ?
                """, (datetime.utcnow(), resolved_by, error_id))
                
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to mark error resolved: {e}")
            return False
            
    async def purge_expired_errors(self, days: int = 30) -> int:
        """Remove expired error logs"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM error_logs 
                    WHERE created_at < ?
                    AND resolved = 1
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old error logs")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old errors: {e}")
            return 0


# Custom error handler that integrates with persistence
class DatabaseErrorHandler(logging.Handler):
    """Logging handler that persists errors to database"""
    
    def __init__(self, persistence: ErrorLogPersistence, component: str):
        super().__init__()
        self.persistence = persistence
        self.component = component
        self.loop = None
        
    def emit(self, record):
        """Handle log record"""
        if record.levelno >= logging.ERROR:
            try:
                # Get or create event loop
                try:
                    self.loop = asyncio.get_running_loop()
                except RuntimeError:
                    return  # No event loop available
                
                # Schedule error logging
                asyncio.create_task(self._log_error(record))
                
            except Exception:
                self.handleError(record)
                
    async def _log_error(self, record):
        """Log error to database"""
        error_details = {
            'filename': record.filename,
            'lineno': record.lineno,
            'funcName': record.funcName,
            'module': record.module
        }
        
        severity = "critical" if record.levelno >= logging.CRITICAL else "error"
        
        await self.persistence.log_error(
            error_type=record.name,
            error_message=record.getMessage(),
            error_details=error_details,
            component=self.component,
            severity=severity
        )


# Global error persistence instance
_error_persistence: Optional[ErrorLogPersistence] = None

async def get_error_persistence() -> ErrorLogPersistence:
    """Get global error persistence instance"""
    global _error_persistence
    if _error_persistence is None:
        _error_persistence = ErrorLogPersistence()
        await _error_persistence.start()
    return _error_persistence

def setup_database_error_logging(component: str):
    """Setup database error logging for a component"""
    # This would be called during component initialization
    # Example usage:
    # setup_database_error_logging("ml_inference")
    pass