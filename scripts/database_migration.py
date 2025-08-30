#!/usr/bin/env python3
"""
Database Migration Script
Handles consolidation and cleanup of duplicate tables across databases
"""

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class DatabaseMigration:
    def __init__(self, db_dir: str = "data"):
        self.db_dir = Path(db_dir)
        self.main_db = "cryptotrading.db"
        self.market_db = "real_market_data.db"
        self.backup_db = "rex.db"

    def backup_databases(self) -> Dict[str, str]:
        """Create backups of all databases before migration"""
        backup_dir = self.db_dir / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backups = {}

        for db_name in [self.main_db, self.market_db, self.backup_db]:
            db_path = self.db_dir / db_name
            if db_path.exists():
                backup_name = f"{db_name}.backup_{timestamp}"
                backup_path = backup_dir / backup_name
                shutil.copy2(db_path, backup_path)
                backups[db_name] = str(backup_path)
                print(f"✅ Backed up {db_name} to {backup_path}")

        return backups

    def analyze_table_data(self, db_name: str, table_name: str) -> Dict:
        """Analyze data in a specific table"""
        db_path = self.db_dir / db_name
        if not db_path.exists():
            return {}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        # Get latest timestamp if available
        timestamp_cols = ["created_at", "timestamp", "updated_at"]
        latest_timestamp = None
        for col in timestamp_cols:
            try:
                cursor.execute(f"SELECT MAX({col}) FROM {table_name}")
                result = cursor.fetchone()[0]
                if result:
                    latest_timestamp = result
                    break
            except:
                pass

        conn.close()

        return {"row_count": row_count, "latest_timestamp": latest_timestamp}

    def consolidate_duplicate_tables(self, dry_run: bool = True) -> List[Dict]:
        """Consolidate duplicate tables, keeping data in main database"""
        actions = []

        # Tables that exist in multiple databases
        duplicate_tables = {
            "a2a_agents": ["cryptotrading.db", "rex.db"],
            "a2a_connections": ["cryptotrading.db", "rex.db"],
            "a2a_messages": ["cryptotrading.db", "rex.db"],
            "a2a_workflow_executions": ["cryptotrading.db", "rex.db"],
            "a2a_workflows": ["cryptotrading.db", "rex.db"],
            "agent_contexts": ["cryptotrading.db", "rex.db"],
            "aggregated_market_data": ["cryptotrading.db", "rex.db"],
            "ai_analyses": ["cryptotrading.db", "rex.db"],
            "conversation_messages": ["cryptotrading.db", "rex.db"],
            "conversation_sessions": ["cryptotrading.db", "rex.db"],
            "data_ingestion_jobs": ["cryptotrading.db", "rex.db"],
            "data_quality_metrics": ["cryptotrading.db", "rex.db"],
            "encryption_key_metadata": ["cryptotrading.db", "rex.db"],
            "factor_data": ["cryptotrading.db", "rex.db"],
            "macro_data": ["cryptotrading.db", "rex.db"],
            "market_data": ["cryptotrading.db", "real_market_data.db", "rex.db"],
            "market_data_sources": ["cryptotrading.db", "rex.db"],
            "memory_fragments": ["cryptotrading.db", "rex.db"],
            "onchain_data": ["cryptotrading.db", "rex.db"],
            "semantic_memory": ["cryptotrading.db", "rex.db"],
            "sentiment_data": ["cryptotrading.db", "rex.db"],
            "time_series": ["cryptotrading.db", "rex.db"],
            "users": ["cryptotrading.db", "rex.db"],
        }

        for table_name, databases in duplicate_tables.items():
            # Analyze data in each database
            table_info = {}
            for db_name in databases:
                info = self.analyze_table_data(db_name, table_name)
                if info:
                    table_info[db_name] = info

            # Determine primary database (with most recent data)
            primary_db = None
            max_timestamp = None
            max_rows = 0

            for db_name, info in table_info.items():
                if info["row_count"] > max_rows:
                    max_rows = info["row_count"]
                    primary_db = db_name
                elif info["row_count"] == max_rows and info["latest_timestamp"]:
                    if not max_timestamp or info["latest_timestamp"] > max_timestamp:
                        max_timestamp = info["latest_timestamp"]
                        primary_db = db_name

            # Create action plan
            action = {
                "table": table_name,
                "primary_db": primary_db,
                "primary_rows": table_info.get(primary_db, {}).get("row_count", 0),
                "action": "keep" if primary_db == self.main_db else "migrate",
                "secondary_dbs": [db for db in databases if db != primary_db],
                "details": table_info,
            }
            actions.append(action)

            # Execute migration if not dry run
            if not dry_run and action["action"] == "migrate":
                self._migrate_table_data(table_name, primary_db, self.main_db)

        return actions

    def _migrate_table_data(self, table_name: str, source_db: str, target_db: str):
        """Migrate table data from source to target database"""
        source_path = self.db_dir / source_db
        target_path = self.db_dir / target_db

        if not source_path.exists() or not target_path.exists():
            print(f"⚠️  Cannot migrate {table_name}: database files not found")
            return

        source_conn = sqlite3.connect(source_path)
        target_conn = sqlite3.connect(target_path)

        try:
            # Get table schema from source
            cursor = source_conn.cursor()
            cursor.execute(
                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            )
            create_sql = cursor.fetchone()[0]

            # Drop existing table in target if exists
            target_conn.execute(f"DROP TABLE IF EXISTS {table_name}_old")
            target_conn.execute(f"ALTER TABLE {table_name} RENAME TO {table_name}_old")

            # Create new table in target
            target_conn.execute(create_sql)

            # Copy data
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()

            if rows:
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]

                # Insert data into target
                placeholders = ",".join(["?" for _ in columns])
                target_conn.executemany(
                    f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})", rows
                )

            # Drop old table
            target_conn.execute(f"DROP TABLE {table_name}_old")

            target_conn.commit()
            print(
                f"✅ Migrated {len(rows)} rows from {source_db} to {target_db} for table {table_name}"
            )

        except Exception as e:
            print(f"❌ Error migrating {table_name}: {e}")
            target_conn.rollback()
        finally:
            source_conn.close()
            target_conn.close()

    def optimize_databases(self):
        """Run VACUUM and ANALYZE on all databases"""
        for db_name in [self.main_db, self.market_db]:
            db_path = self.db_dir / db_name
            if db_path.exists():
                conn = sqlite3.connect(db_path)
                print(f"🔧 Optimizing {db_name}...")
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                conn.close()
                print(f"✅ {db_name} optimized")

    def generate_migration_report(self, actions: List[Dict]) -> str:
        """Generate a migration report"""
        report = []
        report.append("=" * 60)
        report.append("DATABASE MIGRATION ANALYSIS")
        report.append("=" * 60)
        report.append("")

        # Group by action type
        keep_count = sum(1 for a in actions if a["action"] == "keep")
        migrate_count = sum(1 for a in actions if a["action"] == "migrate")

        report.append(f"Tables to keep in {self.main_db}: {keep_count}")
        report.append(f"Tables to migrate to {self.main_db}: {migrate_count}")
        report.append("")

        # Detailed table analysis
        report.append("TABLE ANALYSIS:")
        report.append("-" * 40)

        for action in actions:
            report.append(f"\n📊 {action['table']}")
            report.append(f"   Primary DB: {action['primary_db']} ({action['primary_rows']} rows)")
            report.append(f"   Action: {action['action'].upper()}")

            for db_name, info in action["details"].items():
                if db_name != action["primary_db"]:
                    report.append(f"   {db_name}: {info['row_count']} rows")

        report.append("")
        report.append("=" * 60)
        report.append("RECOMMENDATION:")
        report.append("=" * 60)

        if migrate_count > 0:
            report.append(f"⚠️  Found {migrate_count} tables that need migration")
            report.append("   Run with --execute flag to perform migration")
        else:
            report.append("✅ All tables are properly consolidated in main database")
            report.append("   Consider removing rex.db if it's no longer needed")

        return "\n".join(report)


def main():
    """Main entry point"""
    import sys

    dry_run = "--execute" not in sys.argv

    migrator = DatabaseMigration()

    print("🚀 Starting Database Migration Analysis\n")

    # Backup databases first
    if not dry_run:
        print("📦 Creating backups...")
        backups = migrator.backup_databases()
        print(f"✅ Backups created in data/backups/\n")

    # Analyze and potentially migrate
    print("🔍 Analyzing duplicate tables...")
    actions = migrator.consolidate_duplicate_tables(dry_run=dry_run)

    # Generate report
    report = migrator.generate_migration_report(actions)
    print("\n" + report)

    # Save report
    report_file = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_file}")

    # Optimize if migration was performed
    if not dry_run:
        print("\n🔧 Optimizing databases...")
        migrator.optimize_databases()
        print("\n✨ Migration complete!")
    else:
        print("\n💡 This was a dry run. To execute migration, run:")
        print("   python3 scripts/database_migration.py --execute")


if __name__ == "__main__":
    main()
