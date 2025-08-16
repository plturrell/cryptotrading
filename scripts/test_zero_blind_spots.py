#!/usr/bin/env python3
"""
Zero Blind Spots Test - Comprehensive Multi-Language Indexing Validation
Tests the complete elimination of knowledge blind spots across all languages
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from cryptotrading.infrastructure.analysis.multi_language_indexer import index_multi_language_project
from cryptotrading.infrastructure.analysis.enhanced_angle_queries import (
    EnhancedAngleQueryEngine, 
    find_cap_entities,
    find_ui5_controllers, 
    find_javascript_functions,
    find_typescript_interfaces,
    find_typescript_classes,
    find_typescript_functions,
    analyze_project_architecture
)

class ZeroBlindSpotsValidator:
    """Validates complete elimination of knowledge blind spots"""
    
    def __init__(self):
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "indexing_results": {},
            "language_coverage": {},
            "blind_spots_analysis": {},
            "query_validation": {},
            "final_score": 0.0
        }
    
    async def run_comprehensive_test(self):
        """Run comprehensive zero blind spots validation"""
        print("🔍 ZERO BLIND SPOTS VALIDATION")
        print("=" * 50)
        
        # Step 1: Multi-language indexing
        print("\n📚 Step 1: Multi-Language Indexing...")
        indexing_start = time.time()
        
        indexing_results = index_multi_language_project(str(project_root))
        indexing_duration = time.time() - indexing_start
        
        self.results["indexing_results"] = indexing_results
        
        # Print indexing summary
        summary = indexing_results.get("indexing_summary", {})
        print(f"   ✅ Files indexed: {summary.get('total_files_indexed', 0)}")
        print(f"   ✅ Facts generated: {summary.get('total_facts_generated', 0):,}")
        print(f"   ✅ Languages supported: {summary.get('languages_supported', 0)}")
        print(f"   ⏱️ Duration: {indexing_duration:.2f}s")
        
        # Step 2: Language coverage analysis
        print("\n🌐 Step 2: Language Coverage Analysis...")
        coverage = indexing_results.get("coverage_analysis", {})
        coverage_pct = coverage.get("coverage_percentage", 0)
        
        print(f"   📊 Coverage: {coverage_pct:.1f}%")
        print(f"   📁 Total files: {coverage.get('total_relevant_files', 0)}")
        print(f"   ✅ Indexed files: {coverage.get('indexed_files', 0)}")
        print(f"   ❌ Unindexed files: {coverage.get('unindexed_files', 0)}")
        
        self.results["language_coverage"] = coverage
        
        # Step 3: Blind spots analysis
        print("\n🎯 Step 3: Blind Spots Analysis...")
        blind_spots = indexing_results.get("blind_spots_eliminated", {})
        remaining_spots = blind_spots.get("blind_spots_count", 0)
        
        print(f"   🔍 Remaining blind spots: {remaining_spots}")
        if remaining_spots > 0:
            for spot in blind_spots.get("remaining_blind_spots", []):
                print(f"      • {spot}")
        else:
            print("   🎉 NO BLIND SPOTS DETECTED!")
        
        self.results["blind_spots_analysis"] = blind_spots
        
        # Step 4: Query validation
        print("\n🔎 Step 4: Enhanced Query Validation...")
        query_results = await self._validate_enhanced_queries(indexing_results["glean_facts"])
        self.results["query_validation"] = query_results
        
        # Step 5: Architecture analysis
        print("\n🏗️ Step 5: Cross-Language Architecture Analysis...")
        architecture = self._analyze_architecture(indexing_results["glean_facts"])
        
        # Calculate final score
        final_score = self._calculate_final_score()
        self.results["final_score"] = final_score
        
        # Generate final report
        self._generate_final_report()
        
        return self.results
    
    async def _validate_enhanced_queries(self, facts):
        """Validate enhanced multi-language queries"""
        engine = EnhancedAngleQueryEngine(facts)
        query_results = {}
        
        try:
            # Test CAP queries
            cap_entities = find_cap_entities(engine)
            query_results["cap_entities_found"] = len(cap_entities)
            print(f"   🏢 CAP entities found: {len(cap_entities)}")
            
            # Test UI5 queries  
            ui5_controllers = find_ui5_controllers(engine)
            query_results["ui5_controllers_found"] = len(ui5_controllers)
            print(f"   📱 UI5 controllers found: {len(ui5_controllers)}")
            
            # Test JavaScript queries
            js_functions = find_javascript_functions(engine)
            query_results["js_functions_found"] = len(js_functions)
            print(f"   ⚡ JavaScript functions found: {len(js_functions)}")
            
            # Test TypeScript queries
            ts_interfaces = find_typescript_interfaces(engine)
            ts_classes = find_typescript_classes(engine)
            ts_functions = find_typescript_functions(engine)
            query_results["ts_interfaces_found"] = len(ts_interfaces)
            query_results["ts_classes_found"] = len(ts_classes)
            query_results["ts_functions_found"] = len(ts_functions)
            print(f"   ⚡ TypeScript interfaces found: {len(ts_interfaces)}")
            print(f"   ⚡ TypeScript classes found: {len(ts_classes)}")
            print(f"   ⚡ TypeScript functions found: {len(ts_functions)}")
            
            # Test comprehensive stats
            stats = engine.get_comprehensive_stats()
            query_results["comprehensive_stats"] = stats
            print(f"   📊 Total predicates: {len(stats['predicate_distribution'])}")
            
            # Test cross-language relationships
            relationships = engine.find_cross_language_relationships()
            query_results["cross_language_relationships"] = len(relationships)
            print(f"   🔗 Cross-language relationships: {len(relationships)}")
            
            query_results["query_engine_success"] = True
            
        except Exception as e:
            print(f"   ❌ Query validation failed: {e}")
            query_results["query_engine_success"] = False
            query_results["error"] = str(e)
        
        return query_results
    
    def _analyze_architecture(self, facts):
        """Analyze cross-language architecture"""
        engine = EnhancedAngleQueryEngine(facts)
        return analyze_project_architecture(engine)
    
    def _calculate_final_score(self):
        """Calculate final zero blind spots score"""
        score = 0.0
        
        # Coverage score (40 points)
        coverage_pct = self.results["language_coverage"].get("coverage_percentage", 0)
        score += (coverage_pct / 100) * 40
        
        # Blind spots elimination (30 points)
        blind_spots_count = self.results["blind_spots_analysis"].get("blind_spots_count", 999)
        if blind_spots_count == 0:
            score += 30
        else:
            score += max(0, 30 - (blind_spots_count * 5))
        
        # Query functionality (20 points)
        if self.results["query_validation"].get("query_engine_success", False):
            score += 20
        
        # Facts generation (10 points)
        facts_count = self.results["indexing_results"].get("indexing_summary", {}).get("total_facts_generated", 0)
        if facts_count > 50000:
            score += 10
        elif facts_count > 30000:
            score += 8
        elif facts_count > 10000:
            score += 6
        else:
            score += max(0, facts_count / 10000 * 6)
        
        return min(100.0, score)
    
    def _generate_final_report(self):
        """Generate final validation report"""
        print(f"\n{'='*60}")
        print("🎯 ZERO BLIND SPOTS VALIDATION REPORT")
        print(f"{'='*60}")
        
        # Overall score
        score = self.results["final_score"]
        if score >= 95:
            status = "🏆 PERFECT - ZERO BLIND SPOTS ACHIEVED"
            color = "🟢"
        elif score >= 85:
            status = "✅ EXCELLENT - MINIMAL BLIND SPOTS"
            color = "🟡"
        elif score >= 70:
            status = "⚠️ GOOD - SOME BLIND SPOTS REMAIN"
            color = "🟠"
        else:
            status = "❌ NEEDS WORK - SIGNIFICANT BLIND SPOTS"
            color = "🔴"
        
        print(f"\n{color} Final Score: {score:.1f}/100 - {status}")
        
        # Language breakdown
        lang_dist = self.results["indexing_results"].get("language_distribution", {})
        print(f"\n📊 Language Coverage:")
        for language, count in lang_dist.items():
            print(f"   • {language}: {count} files")
        
        # Predicate breakdown
        pred_dist = self.results["indexing_results"].get("predicate_distribution", {})
        print(f"\n🔍 Knowledge Types Captured:")
        for predicate, count in list(pred_dist.items())[:10]:  # Top 10
            print(f"   • {predicate}: {count:,} facts")
        
        # Blind spots status
        blind_spots = self.results["blind_spots_analysis"]
        if blind_spots.get("coverage_complete", False):
            print(f"\n🎉 ZERO BLIND SPOTS CONFIRMED!")
            print("   All supported file types have been indexed")
        else:
            remaining = blind_spots.get("remaining_blind_spots", [])
            print(f"\n⚠️ Remaining Blind Spots: {len(remaining)}")
            for spot in remaining[:3]:  # Show top 3
                print(f"   • {spot}")
        
        # Performance metrics
        summary = self.results["indexing_results"].get("indexing_summary", {})
        print(f"\n⚡ Performance Metrics:")
        print(f"   • Total facts: {summary.get('total_facts_generated', 0):,}")
        print(f"   • Files indexed: {summary.get('total_files_indexed', 0)}")
        print(f"   • Duration: {summary.get('indexing_duration_seconds', 0):.2f}s")
        print(f"   • Facts/second: {summary.get('total_facts_generated', 0) / max(1, summary.get('indexing_duration_seconds', 1)):,.0f}")
        
        print(f"\n{'='*60}")
        if score >= 95:
            print("🏆 CONGRATULATIONS! ZERO BLIND SPOTS ACHIEVED!")
            print("Your knowledge collector now has comprehensive coverage")
            print("across Python, SAP CAP, JavaScript/UI5, and configuration files.")
        else:
            print("💪 Keep improving to achieve zero blind spots!")
        print(f"{'='*60}")

async def main():
    """Main validation function"""
    validator = ZeroBlindSpotsValidator()
    
    try:
        results = await validator.run_comprehensive_test()
        
        # Save results
        output_file = project_root / "data" / "zero_blind_spots_validation.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {output_file}")
        
        # Exit with appropriate code
        final_score = results["final_score"]
        sys.exit(0 if final_score >= 95 else 1)
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
