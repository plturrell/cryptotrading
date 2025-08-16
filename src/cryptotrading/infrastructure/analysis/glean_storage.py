"""
Serverless Glean Storage Backend
Provides Glean-compatible fact storage without requiring Docker or a Glean server
Suitable for Vercel deployment
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import pickle

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
    """Serverless storage backend for Glean facts using SQLite"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".glean" / "cryptotrading.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY,
                    predicate TEXT NOT NULL,
                    key_json TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    unit TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predicate 
                ON facts(predicate)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS units (
                    unit_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indexed_at TIMESTAMP,
                    facts_count INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schemas (
                    predicate TEXT PRIMARY KEY,
                    schema_json TEXT NOT NULL,
                    version TEXT DEFAULT '1.0'
                )
            """)
    
    def store_facts(self, facts: List[Dict[str, Any]], unit: str = "default") -> int:
        """Store multiple facts in the database"""
        stored = 0
        
        with sqlite3.connect(str(self.db_path)) as conn:
            # Create unit if not exists
            conn.execute(
                "INSERT OR IGNORE INTO units (unit_id, name) VALUES (?, ?)",
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
                    conn.execute("""
                        INSERT OR REPLACE INTO facts 
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
            conn.execute("""
                UPDATE units 
                SET facts_count = (
                    SELECT COUNT(*) FROM facts WHERE unit = ?
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
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM facts WHERE predicate = ?"
            params = [predicate]
            
            # Apply key filters if provided
            if key_filters:
                for k, v in key_filters.items():
                    query += f" AND json_extract(key_json, '$.{k}') = ?"
                    params.append(v)
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = conn.execute(query, params)
            
            facts = []
            for row in cursor:
                fact = GleanFact(
                    predicate=row["predicate"],
                    key=json.loads(row["key_json"]),
                    value=json.loads(row["value_json"]),
                    unit=row["unit"]
                )
                facts.append(fact)
            
            return facts
    
    def get_predicates(self) -> List[str]:
        """Get all unique predicates in the database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("SELECT DISTINCT predicate FROM facts")
            return [row[0] for row in cursor]
    
    def get_units(self) -> List[Dict[str, Any]]:
        """Get all units with their metadata"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT unit_id, name, created_at, indexed_at, facts_count
                FROM units
                ORDER BY indexed_at DESC
            """)
            return [dict(row) for row in cursor]
    
    def delete_unit(self, unit: str) -> int:
        """Delete all facts for a unit"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM facts WHERE unit = ?", (unit,))
            deleted = cursor.rowcount
            conn.execute("DELETE FROM units WHERE unit_id = ?", (unit,))
            conn.commit()
        
        logger.info(f"Deleted {deleted} facts from unit '{unit}'")
        return deleted
    
    def export_unit(self, unit: str, output_path: Path) -> None:
        """Export all facts for a unit to JSON"""
        facts = []
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM facts WHERE unit = ?", 
                (unit,)
            )
            
            for row in cursor:
                facts.append({
                    "predicate": row["predicate"],
                    "key": json.loads(row["key_json"]),
                    "value": json.loads(row["value_json"])
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
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO schemas (predicate, schema_json)
                VALUES (?, ?)
            """, (predicate, json.dumps(schema)))
            conn.commit()
    
    def get_schema(self, predicate: str) -> Optional[Dict[str, Any]]:
        """Get schema for a predicate"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT schema_json FROM schemas WHERE predicate = ?",
                (predicate,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Total facts
            total_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            
            # Facts by predicate
            cursor = conn.execute("""
                SELECT predicate, COUNT(*) as count
                FROM facts
                GROUP BY predicate
                ORDER BY count DESC
            """)
            predicate_counts = {row[0]: row[1] for row in cursor}
            
            # Storage size
            db_size = self.db_path.stat().st_size
            
            return {
                "total_facts": total_facts,
                "total_predicates": len(predicate_counts),
                "predicate_counts": predicate_counts,
                "storage_size_bytes": db_size,
                "storage_size_mb": round(db_size / (1024 * 1024), 2)
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