"""
Serverless Glean Storage Backend
Provides Glean-compatible fact storage without requiring Docker or a Glean server
Suitable for Vercel deployment - uses UnifiedDatabase for consistency
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import pickle

# Use unified database instead of direct SQLite
from ...infrastructure.database.unified_database import UnifiedDatabase

logger = logging.getLogger(__name__)


@dataclass
class GleanFact:
    """Represents a Glean fact"""
    predicate: str
    key: Dict[str, Any]
    value: Dict[str, Any]
    unit: str = ""
    
    def fact_id(self) -> str:
        """Generate unique ID for this fact"""
        content = f"{self.predicate}:{json.dumps(self.key, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class GleanStorage:
    """Serverless storage backend for Glean facts using UnifiedDatabase"""
    
    def __init__(self, db: Optional[UnifiedDatabase] = None):
        # Use unified database instead of direct SQLite
        self.db = db or UnifiedDatabase()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema using UnifiedDatabase"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS glean_facts (
                    fact_id TEXT PRIMARY KEY,
                    predicate TEXT NOT NULL,
                    key_json TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    unit TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_glean_predicate 
                ON glean_facts(predicate)
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS glean_units (
                    unit_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indexed_at TIMESTAMP,
                    facts_count INTEGER DEFAULT 0
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS glean_schemas (
                    predicate TEXT PRIMARY KEY,
                    schema_json TEXT NOT NULL,
                    version TEXT DEFAULT '1.0'
                )
            """)
            
            conn.commit()
    
    def store_facts(self, facts: List[Dict[str, Any]], unit: str = "default") -> int:
        """Store multiple facts in the database"""
        stored = 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create unit if not exists
            cursor.execute(
                "INSERT OR IGNORE INTO glean_units (unit_id, name) VALUES (?, ?)",
                (unit, unit)
            )
            
            for fact_dict in facts:
                fact = GleanFact(
                    predicate=fact_dict["predicate"],
                    key=fact_dict.get("key", {}),
                    value=fact_dict.get("value", {}),
                    unit=unit
                )
                
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO glean_facts 
                        (fact_id, predicate, key_json, value_json, unit)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        fact.fact_id(),
                        fact.predicate,
                        json.dumps(fact.key),
                        json.dumps(fact.value),
                        fact.unit
                    ))
                    stored += 1
                except Exception as e:
                    logger.error(f"Failed to store fact: {e}")
            
            # Update unit stats
            cursor.execute("""
                UPDATE glean_units 
                SET facts_count = (
                    SELECT COUNT(*) FROM glean_facts WHERE unit = ?
                ),
                indexed_at = CURRENT_TIMESTAMP
                WHERE unit_id = ?
            """, (unit, unit))
            
            conn.commit()
        
        logger.info(f"Stored {stored} facts in unit '{unit}'")
        return stored
    
    def query_facts(self, predicate: str, 
                   key_filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None) -> List[GleanFact]:
        """Query facts by predicate and optional key filters"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM glean_facts WHERE predicate = ?"
            params = [predicate]
            
            # Apply key filters if provided
            if key_filters:
                for k, v in key_filters.items():
                    query += f" AND json_extract(key_json, '$.{k}') = ?"
                    params.append(v)
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            
            facts = []
            for row in cursor.fetchall():
                fact = GleanFact(
                    predicate=row[1],  # predicate
                    key=json.loads(row[2]),  # key_json
                    value=json.loads(row[3]),  # value_json
                    unit=row[4]  # unit
                )
                facts.append(fact)
            
            return facts
    
    def get_predicates(self) -> List[str]:
        """Get all unique predicates in the database"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT predicate FROM glean_facts")
            return [row[0] for row in cursor.fetchall()]
    
    def get_units(self) -> List[Dict[str, Any]]:
        """Get all units with their metadata"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT unit_id, name, created_at, indexed_at, facts_count
                FROM glean_units
                ORDER BY indexed_at DESC
            """)
            return [
                {
                    "unit_id": row[0],
                    "name": row[1],
                    "created_at": row[2],
                    "indexed_at": row[3],
                    "facts_count": row[4]
                }
                for row in cursor.fetchall()
            ]
    
    def delete_unit(self, unit: str) -> int:
        """Delete all facts for a unit"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM glean_facts WHERE unit = ?", (unit,))
            deleted = cursor.rowcount
            cursor.execute("DELETE FROM glean_units WHERE unit_id = ?", (unit,))
            conn.commit()
        
        logger.info(f"Deleted {deleted} facts from unit '{unit}'")
        return deleted
    
    def export_unit(self, unit: str, output_path: Path) -> None:
        """Export all facts for a unit to JSON"""
        facts = []
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT predicate, key_json, value_json FROM glean_facts WHERE unit = ?", 
                (unit,)
            )
            
            for row in cursor.fetchall():
                facts.append({
                    "predicate": row[0],
                    "key": json.loads(row[1]),
                    "value": json.loads(row[2])
                })
        
        with open(output_path, 'w') as f:
            json.dump({
                "unit": unit,
                "facts": facts,
                "exported_at": datetime.utcnow().isoformat()
            }, f, indent=2)
        
        logger.info(f"Exported {len(facts)} facts to {output_path}")
    
    def import_facts(self, input_path: Path, unit: Optional[str] = None) -> int:
        """Import facts from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        facts = data.get("facts", [])
        import_unit = unit or data.get("unit", "imported")
        
        return self.store_facts(facts, import_unit)
    
    def register_schema(self, predicate: str, schema: Dict[str, Any]) -> None:
        """Register a schema for a predicate"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO glean_schemas (predicate, schema_json)
                VALUES (?, ?)
            """, (predicate, json.dumps(schema)))
            conn.commit()
    
    def get_schema(self, predicate: str) -> Optional[Dict[str, Any]]:
        """Get schema for a predicate"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT schema_json FROM glean_schemas WHERE predicate = ?",
                (predicate,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Total facts
            cursor.execute("SELECT COUNT(*) FROM glean_facts")
            total_facts = cursor.fetchone()[0]
            
            # Facts by predicate
            cursor.execute("""
                SELECT predicate, COUNT(*) as count
                FROM glean_facts
                GROUP BY predicate
                ORDER BY count DESC
            """)
            predicate_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "total_facts": total_facts,
                "total_predicates": len(predicate_counts),
                "predicate_counts": predicate_counts
            }


# Predefined schemas for Python predicates
PYTHON_SCHEMAS = {
    "python.Declaration": {
        "key": {
            "name": "string",
            "file": "string"
        },
        "value": {
            "kind": "enum:function|class|variable|import",
            "signature": "string",
            "documentation": "string"
        }
    },
    
    "python.Reference": {
        "key": {
            "target": "string",
            "file": "string",
            "span": "array:int"
        },
        "value": {}
    },
    
    "python.Import": {
        "key": {
            "module": "string",
            "file": "string"
        },
        "value": {
            "imported_names": "array:string",
            "is_relative": "boolean"
        }
    },
    
    "src.File": {
        "key": {
            "path": "string"
        },
        "value": {
            "language": "string",
            "size_bytes": "int",
            "last_modified": "timestamp"
        }
    }
}


def initialize_python_schemas(storage: GleanStorage) -> None:
    """Initialize storage with Python-specific schemas"""
    for predicate, schema in PYTHON_SCHEMAS.items():
        storage.register_schema(predicate, schema)
    logger.info(f"Registered {len(PYTHON_SCHEMAS)} Python schemas")