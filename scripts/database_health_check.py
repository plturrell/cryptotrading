#!/usr/bin/env python3
"""
Database Health Check and Optimization Script
Analyzes database structure and provides consolidation recommendations
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple

class DatabaseHealthCheck:
    def __init__(self, db_dir: str = "data"):
        self.db_dir = Path(db_dir)
        self.databases = {
            "cryptotrading.db": "Main application database",
            "real_market_data.db": "High-frequency market data",
            "rex.db": "Parallel/backup database"
        }
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "databases": {},
            "recommendations": []
        }
    
    def check_database_exists(self, db_name: str) -> bool:
        """Check if database file exists"""
        db_path = self.db_dir / db_name
        return db_path.exists()
    
    def get_database_size(self, db_name: str) -> int:
        """Get database file size in bytes"""
        db_path = self.db_dir / db_name
        if db_path.exists():
            return db_path.stat().st_size
        return 0
    
    def get_table_info(self, db_name: str) -> List[Dict]:
        """Get information about all tables in database"""
        db_path = self.db_dir / db_name
        if not db_path.exists():
            return []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        table_info = []
        for table in tables:
            table_name = table[0]
            if table_name == 'sqlite_sequence':
                continue
                
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # Get column count
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_count = len(columns)
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            index_count = len(indexes)
            
            table_info.append({
                "name": table_name,
                "row_count": row_count,
                "column_count": column_count,
                "index_count": index_count
            })
        
        conn.close()
        return table_info
    
    def check_table_overlap(self) -> Dict[str, List[str]]:
        """Check for duplicate tables across databases"""
        all_tables = {}
        overlaps = {}
        
        for db_name in self.databases:
            if self.check_database_exists(db_name):
                tables = self.get_table_info(db_name)
                for table in tables:
                    table_name = table['name']
                    if table_name not in all_tables:
                        all_tables[table_name] = []
                    all_tables[table_name].append(db_name)
        
        # Find tables that exist in multiple databases
        for table_name, databases in all_tables.items():
            if len(databases) > 1:
                overlaps[table_name] = databases
        
        return overlaps
    
    def analyze_database_usage(self, db_name: str) -> Dict:
        """Analyze database usage patterns"""
        db_path = self.db_dir / db_name
        if not db_path.exists():
            return {}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get database statistics
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        
        cursor.execute("PRAGMA freelist_count")
        free_pages = cursor.fetchone()[0]
        
        # Calculate usage
        total_size = page_count * page_size
        used_size = (page_count - free_pages) * page_size
        usage_percent = (used_size / total_size * 100) if total_size > 0 else 0
        
        conn.close()
        
        return {
            "total_pages": page_count,
            "free_pages": free_pages,
            "page_size": page_size,
            "total_size_bytes": total_size,
            "used_size_bytes": used_size,
            "usage_percent": round(usage_percent, 2)
        }
    
    def generate_consolidation_recommendations(self) -> List[str]:
        """Generate database consolidation recommendations"""
        recommendations = []
        
        # Check database sizes
        sizes = {}
        for db_name in self.databases:
            if self.check_database_exists(db_name):
                sizes[db_name] = self.get_database_size(db_name)
        
        # Check for overlapping tables
        overlaps = self.check_table_overlap()
        
        # Analyze recommendations
        if overlaps:
            recommendations.append(
                f"DUPLICATION: Found {len(overlaps)} tables that exist in multiple databases. "
                "Consider consolidating to avoid data inconsistency."
            )
            for table, dbs in overlaps.items():
                recommendations.append(f"  - Table '{table}' exists in: {', '.join(dbs)}")
        
        # Size-based recommendations
        total_size = sum(sizes.values())
        if total_size < 100 * 1024 * 1024:  # Less than 100MB total
            recommendations.append(
                "OPTIMIZATION: Total database size is small (<100MB). "
                "Current 3-database structure is fine for separation of concerns."
            )
        elif total_size > 1024 * 1024 * 1024:  # More than 1GB
            recommendations.append(
                "SCALING: Large database size detected (>1GB). "
                "Consider implementing table partitioning or archival strategy."
            )
        
        # Check if rex.db is being used
        if self.check_database_exists("rex.db"):
            rex_tables = self.get_table_info("rex.db")
            rex_rows = sum(t['row_count'] for t in rex_tables)
            if rex_rows == 0:
                recommendations.append(
                    "CLEANUP: rex.db appears to be unused (0 rows). "
                    "Consider removing if it's not needed for backup."
                )
        
        # Performance recommendations
        if self.check_database_exists("real_market_data.db"):
            market_tables = self.get_table_info("real_market_data.db")
            if market_tables:
                for table in market_tables:
                    if table['row_count'] > 1000000 and table['index_count'] < 2:
                        recommendations.append(
                            f"PERFORMANCE: Table '{table['name']}' in real_market_data.db has "
                            f"{table['row_count']:,} rows but only {table['index_count']} indexes. "
                            "Consider adding indexes for better query performance."
                        )
        
        if not recommendations:
            recommendations.append(
                "HEALTHY: Database structure appears optimal. "
                "No immediate consolidation needed."
            )
        
        return recommendations
    
    def run_health_check(self) -> Dict:
        """Run complete health check"""
        print("🔍 Starting Database Health Check...\n")
        
        for db_name, description in self.databases.items():
            print(f"Checking {db_name} ({description})...")
            
            if not self.check_database_exists(db_name):
                print(f"  ⚠️  Database not found")
                self.report["databases"][db_name] = {
                    "exists": False,
                    "description": description
                }
                continue
            
            size = self.get_database_size(db_name)
            tables = self.get_table_info(db_name)
            usage = self.analyze_database_usage(db_name)
            
            self.report["databases"][db_name] = {
                "exists": True,
                "description": description,
                "size_mb": round(size / (1024 * 1024), 2),
                "table_count": len(tables),
                "total_rows": sum(t['row_count'] for t in tables),
                "usage": usage,
                "tables": tables
            }
            
            print(f"  ✅ Size: {size / (1024 * 1024):.2f} MB")
            print(f"  ✅ Tables: {len(tables)}")
            print(f"  ✅ Total rows: {sum(t['row_count'] for t in tables):,}")
            print(f"  ✅ Usage: {usage.get('usage_percent', 0):.1f}%")
        
        # Generate recommendations
        print("\n📊 Analyzing consolidation needs...")
        self.report["recommendations"] = self.generate_consolidation_recommendations()
        
        # Check for overlaps
        overlaps = self.check_table_overlap()
        self.report["table_overlaps"] = overlaps
        
        return self.report
    
    def print_recommendations(self):
        """Print recommendations in a formatted way"""
        print("\n" + "="*60)
        print("📋 RECOMMENDATIONS")
        print("="*60)
        
        for i, rec in enumerate(self.report["recommendations"], 1):
            if rec.startswith("  -"):
                print(rec)
            else:
                print(f"\n{i}. {rec}")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Summary statistics
        total_size = sum(
            db.get("size_mb", 0) 
            for db in self.report["databases"].values() 
            if db.get("exists", False)
        )
        total_tables = sum(
            db.get("table_count", 0) 
            for db in self.report["databases"].values() 
            if db.get("exists", False)
        )
        total_rows = sum(
            db.get("total_rows", 0) 
            for db in self.report["databases"].values() 
            if db.get("exists", False)
        )
        
        print(f"Total database size: {total_size:.2f} MB")
        print(f"Total tables: {total_tables}")
        print(f"Total rows: {total_rows:,}")
        
        if self.report.get("table_overlaps"):
            print(f"Duplicate tables: {len(self.report['table_overlaps'])}")
        
        # Save report to file
        report_file = f"database_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.report, f, indent=2)
        print(f"\n📄 Full report saved to: {report_file}")


def main():
    """Main entry point"""
    checker = DatabaseHealthCheck()
    checker.run_health_check()
    checker.print_recommendations()
    
    # Offer to run optimization
    print("\n" + "="*60)
    response = input("Would you like to run database optimization? (y/n): ")
    if response.lower() == 'y':
        print("\n🔧 Running optimization...")
        for db_name in checker.databases:
            if checker.check_database_exists(db_name):
                db_path = checker.db_dir / db_name
                print(f"  Optimizing {db_name}...")
                conn = sqlite3.connect(db_path)
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                conn.close()
                print(f"  ✅ {db_name} optimized")
        print("\n✨ Database optimization complete!")


if __name__ == "__main__":
    main()