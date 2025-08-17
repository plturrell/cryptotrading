"""
Database Adapter for Code Management
Provides database operations for issues, metrics, and code files with unified interface
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import asdict

from ..database import UnifiedDatabase
from .intelligent_code_manager import CodeIssue, IssueType, FixStatus

logger = logging.getLogger(__name__)

class EnumJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for enum types"""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

class CodeManagementDatabaseAdapter:
    """Database adapter for code management operations"""
    
    def __init__(self, database: UnifiedDatabase):
        self.db = database
    
    # Issue Management
    async def save_issue(self, issue: CodeIssue) -> str:
        """Save issue to database and return ID"""
        issue_id = str(uuid.uuid4())
        
        # Create metadata with initial lifecycle state
        issue_metadata = {
            "lifecycle_state": "detected",
            "priority": "medium",
            "created_by": "system"
        }
        
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                # SQLite
                cursor.execute("""
                    INSERT INTO issues (
                        id, issue_type, severity, state, priority, file_path, 
                        line_number, description, suggested_fix, auto_fixable, 
                        fix_status, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    issue_id, issue.type.value, issue.severity, 
                    "detected", "medium", issue.file_path, issue.line_number,
                    issue.description, issue.suggested_fix, 
                    1 if issue.auto_fixable else 0, issue.fix_status.value,
                    json.dumps(issue_metadata, cls=EnumJSONEncoder), datetime.now(), datetime.now()
                ))
            else:
                # PostgreSQL
                cursor.execute("""
                    INSERT INTO issues (
                        id, issue_type, severity, state, priority, file_path, 
                        line_number, description, suggested_fix, auto_fixable, 
                        fix_status, metadata, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    issue_id, issue.type.value, issue.severity,
                    "detected", "medium", issue.file_path, issue.line_number,
                    issue.description, issue.suggested_fix, issue.auto_fixable,
                    issue.fix_status.value, json.dumps(issue_metadata, cls=EnumJSONEncoder), 
                    datetime.now(), datetime.now()
                ))
            
            self.db.db_conn.commit()
            logger.info("Saved issue %s to database", issue_id)
            return issue_id
            
        except Exception as e:
            logger.error("Failed to save issue: %s", e)
            self.db.db_conn.rollback()
            raise
        finally:
            cursor.close()
    
    async def get_issues(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get issues with optional filters"""
        cursor = self.db.db_conn.cursor()
        try:
            query = "SELECT * FROM issues"
            params = []
            
            if filters:
                conditions = []
                if "issue_type" in filters:
                    conditions.append("issue_type = ?")
                    params.append(filters["issue_type"])
                if "severity" in filters:
                    conditions.append("severity = ?")
                    params.append(filters["severity"])
                if "state" in filters:
                    conditions.append("state = ?")
                    params.append(filters["state"])
                if "file_path" in filters:
                    conditions.append("file_path LIKE ?")
                    params.append(f"%{filters['file_path']}%")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            if self.db.config.mode.value == "local":
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                # Use proper PostgreSQL parameterized query
                postgres_query = query.replace("?", "%s")
                cursor.execute(postgres_query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error("Failed to get issues: %s", e)
            return []
        finally:
            cursor.close()
    
    async def update_issue_status(self, issue_id: str, status: FixStatus, fixed_at: datetime = None) -> bool:
        """Update issue fix status"""
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    UPDATE issues 
                    SET fix_status = ?, fixed_at = ?, updated_at = ?
                    WHERE id = ?
                """, (status.value, fixed_at, datetime.now(), issue_id))
            else:
                cursor.execute("""
                    UPDATE issues 
                    SET fix_status = %s, fixed_at = %s, updated_at = %s
                    WHERE id = %s
                """, (status.value, fixed_at, datetime.now(), issue_id))
            
            self.db.db_conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error("Failed to update issue status: %s", e)
            self.db.db_conn.rollback()
            return False
        finally:
            cursor.close()
    
    async def update_issue_metadata(self, issue_id: str, metadata: Dict[str, Any]) -> bool:
        """Update issue metadata"""
        cursor = self.db.db_conn.cursor()
        try:
            metadata_json = json.dumps(metadata, cls=EnumJSONEncoder)
            
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    UPDATE issues 
                    SET metadata = ?, updated_at = ?
                    WHERE id = ?
                """, (metadata_json, datetime.now(), issue_id))
            else:
                cursor.execute("""
                    UPDATE issues 
                    SET metadata = %s, updated_at = %s
                    WHERE id = %s
                """, (metadata_json, datetime.now(), issue_id))
            
            self.db.db_conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error("Failed to update issue metadata: %s", e)
            self.db.db_conn.rollback()
            return False
        finally:
            cursor.close()
    
    async def get_issue_stats(self) -> Dict[str, Any]:
        """Get issue statistics"""
        cursor = self.db.db_conn.cursor()
        try:
            stats = {}
            
            # Total issues
            cursor.execute("SELECT COUNT(*) as total FROM issues")
            stats["total_issues"] = cursor.fetchone()[0]
            
            # Issues by type
            cursor.execute("""
                SELECT issue_type, COUNT(*) as count 
                FROM issues 
                GROUP BY issue_type
            """)
            stats["by_type"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Issues by severity
            cursor.execute("""
                SELECT severity, COUNT(*) as count 
                FROM issues 
                GROUP BY severity
            """)
            stats["by_severity"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Issues by status
            cursor.execute("""
                SELECT fix_status, COUNT(*) as count 
                FROM issues 
                GROUP BY fix_status
            """)
            stats["by_status"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get issue stats: %s", e)
            return {}
        finally:
            cursor.close()
    
    # Code Files Management
    async def save_code_file(self, file_path: str, language: str, file_size: int, 
                           content_hash: str, facts_count: int = 0) -> str:
        """Save code file metadata"""
        file_id = str(uuid.uuid4())
        
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    INSERT OR REPLACE INTO code_files (
                        id, file_path, language, file_size, content_hash,
                        last_modified, indexed_at, facts_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_id, file_path, language, file_size, content_hash,
                    datetime.now(), datetime.now(), facts_count, "{}"
                ))
            else:
                cursor.execute("""
                    INSERT INTO code_files (
                        id, file_path, language, file_size, content_hash,
                        last_modified, indexed_at, facts_count, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (file_path) DO UPDATE SET
                        file_size = EXCLUDED.file_size,
                        content_hash = EXCLUDED.content_hash,
                        last_modified = EXCLUDED.last_modified,
                        indexed_at = EXCLUDED.indexed_at,
                        facts_count = EXCLUDED.facts_count
                """, (
                    file_id, file_path, language, file_size, content_hash,
                    datetime.now(), datetime.now(), facts_count, {}
                ))
            
            self.db.db_conn.commit()
            return file_id
            
        except Exception as e:
            logger.error("Failed to save code file: %s", e)
            self.db.db_conn.rollback()
            raise
        finally:
            cursor.close()
    
    async def get_code_files(self, language: str = None) -> List[Dict[str, Any]]:
        """Get code files with optional language filter"""
        cursor = self.db.db_conn.cursor()
        try:
            if language:
                if self.db.config.mode.value == "local":
                    cursor.execute("SELECT * FROM code_files WHERE language = ?", (language,))
                else:
                    cursor.execute("SELECT * FROM code_files WHERE language = %s", (language,))
            else:
                cursor.execute("SELECT * FROM code_files ORDER BY indexed_at DESC")
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error("Failed to get code files: %s", e)
            return []
        finally:
            cursor.close()
    
    # Metrics Management
    async def save_metric(self, file_path: str, metric_type: str, metric_value: float,
                         threshold_value: float = None, status: str = "ok") -> str:
        """Save code quality metric"""
        metric_id = str(uuid.uuid4())
        
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    INSERT INTO code_metrics (
                        id, file_path, metric_type, metric_value, threshold_value,
                        status, measured_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric_id, file_path, metric_type, metric_value, threshold_value,
                    status, datetime.now(), "{}"
                ))
            else:
                cursor.execute("""
                    INSERT INTO code_metrics (
                        id, file_path, metric_type, metric_value, threshold_value,
                        status, measured_at, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    metric_id, file_path, metric_type, metric_value, threshold_value,
                    status, datetime.now(), {}
                ))
            
            self.db.db_conn.commit()
            return metric_id
            
        except Exception as e:
            logger.error("Failed to save metric: %s", e)
            self.db.db_conn.rollback()
            raise
        finally:
            cursor.close()
    
    # Monitoring Events
    async def log_monitoring_event(self, event_type: str, component: str, 
                                 severity: str, message: str, details: Dict[str, Any] = None) -> str:
        """Log monitoring event"""
        event_id = str(uuid.uuid4())
        
        cursor = self.db.db_conn.cursor()
        try:
            if self.db.config.mode.value == "local":
                cursor.execute("""
                    INSERT INTO monitoring_events (
                        id, event_type, component, severity, message, details, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id, event_type, component, severity, message,
                    json.dumps(details or {}, cls=EnumJSONEncoder), datetime.now()
                ))
            else:
                cursor.execute("""
                    INSERT INTO monitoring_events (
                        id, event_type, component, severity, message, details, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    event_id, event_type, component, severity, message,
                    json.dumps(details or {}, cls=EnumJSONEncoder), datetime.now()
                ))
            
            self.db.db_conn.commit()
            return event_id
            
        except Exception as e:
            logger.error("Failed to log monitoring event: %s", e)
            self.db.db_conn.rollback()
            raise
        finally:
            cursor.close()
    
    # Cache Operations
    async def cache_health_metrics(self, metrics: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache health metrics"""
        cache_key = f"health_metrics:{datetime.now().strftime('%Y%m%d%H')}"
        return await self.db.cache_set(cache_key, metrics, ttl)
    
    async def get_cached_health_metrics(self) -> Optional[Dict[str, Any]]:
        """Get cached health metrics"""
        cache_key = f"health_metrics:{datetime.now().strftime('%Y%m%d%H')}"
        return await self.db.cache_get(cache_key)
