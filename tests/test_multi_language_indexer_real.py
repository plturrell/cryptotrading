#!/usr/bin/env python3
"""
Real functionality test for Multi-Language Indexer
Tests all indexers with actual code samples to verify no fake features
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cryptotrading.infrastructure.analysis.multi_language_indexer import (
    UnifiedLanguageIndexer, 
    index_multi_language_project
)

class MultiLanguageIndexerTester:
    """Comprehensive tester for multi-language indexer"""
    
    def __init__(self):
        self.temp_dir = None
        self.test_results = {}
        
    def setup_test_project(self) -> Path:
        """Create a temporary test project with real code samples"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create Python files
        python_dir = self.temp_dir / "src" / "python"
        python_dir.mkdir(parents=True)
        
        (python_dir / "main.py").write_text("""
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str
    age: int = 0

class UserManager:
    def __init__(self):
        self.users: List[User] = []
    
    async def add_user(self, user: User) -> bool:
        \"\"\"Add a new user\"\"\"
        self.users.append(user)
        return True
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        for user in self.users:
            if user.email == email:
                return user
        return None

def main():
    manager = UserManager()
    user = User("John", "john@example.com", 25)
    asyncio.run(manager.add_user(user))

if __name__ == "__main__":
    main()
""")
        
        # Create JavaScript files
        js_dir = self.temp_dir / "webapp" / "controller"
        js_dir.mkdir(parents=True)
        
        (js_dir / "App.controller.js").write_text("""
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast"
], function (Controller, MessageToast) {
    "use strict";

    return Controller.extend("crypto.trading.controller.App", {
        onInit: function () {
            this.getView().setModel(new sap.ui.model.json.JSONModel({
                title: "Crypto Trading Platform"
            }));
        },

        onPress: function (oEvent) {
            const sMessage = "Button pressed!";
            MessageToast.show(sMessage);
        },

        async loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                this.getModel().setData(data);
            } catch (error) {
                console.error("Failed to load data:", error);
            }
        }
    });
});
""")
        
        # Create TypeScript files
        ts_dir = self.temp_dir / "api"
        ts_dir.mkdir(parents=True)
        
        (ts_dir / "types.ts").write_text("""
export interface CryptoPrice {
    symbol: string;
    price: number;
    timestamp: Date;
    volume?: number;
}

export class TradingEngine {
    private prices: Map<string, CryptoPrice> = new Map();
    
    constructor(private apiKey: string) {}
    
    public async updatePrice(symbol: string): Promise<CryptoPrice | null> {
        try {
            const response = await fetch(`/api/price/${symbol}`);
            const data: CryptoPrice = await response.json();
            this.prices.set(symbol, data);
            return data;
        } catch (error) {
            console.error(`Failed to update price for ${symbol}:`, error);
            return null;
        }
    }
    
    public getPrice(symbol: string): CryptoPrice | undefined {
        return this.prices.get(symbol);
    }
}

export enum OrderType {
    BUY = "buy",
    SELL = "sell",
    LIMIT = "limit",
    MARKET = "market"
}

export type OrderStatus = "pending" | "filled" | "cancelled";
""")
        
        # Create SAP CAP files
        cap_dir = self.temp_dir / "db"
        cap_dir.mkdir(parents=True)
        
        (cap_dir / "schema.cds").write_text("""
namespace crypto.trading;

using { Currency, managed } from '@sap/cds/common';

entity Users : managed {
    key ID       : UUID;
    name         : String(100);
    email        : String(255);
    balance      : Decimal(15,2);
    currency     : Currency;
    trades       : Composition of many Trades on trades.user = $self;
}

entity Trades : managed {
    key ID       : UUID;
    user         : Association to Users;
    symbol       : String(10);
    amount       : Decimal(15,8);
    price        : Decimal(15,2);
    type         : String(10);
    status       : String(20);
}

service TradingService {
    entity Users as projection on crypto.trading.Users;
    entity Trades as projection on crypto.trading.Trades;
    
    action executeTrade(userID: UUID, symbol: String, amount: Decimal, type: String) returns String;
    function getUserBalance(userID: UUID) returns Decimal;
}
""")
        
        # Create XML view files
        xml_dir = self.temp_dir / "webapp" / "view"
        xml_dir.mkdir(parents=True)
        
        (xml_dir / "App.view.xml").write_text("""
<mvc:View
    controllerName="crypto.trading.controller.App"
    xmlns:mvc="sap.ui.core.mvc"
    xmlns="sap.m">
    <Page title="{/title}">
        <content>
            <VBox class="sapUiMediumMargin">
                <Title text="Trading Dashboard" level="H2"/>
                <Button text="Load Data" press="onPress"/>
                <Table id="pricesTable" items="{/prices}">
                    <columns>
                        <Column>
                            <Text text="Symbol"/>
                        </Column>
                        <Column>
                            <Text text="Price"/>
                        </Column>
                        <Column>
                            <Text text="Volume"/>
                        </Column>
                    </columns>
                    <items>
                        <ColumnListItem>
                            <cells>
                                <Text text="{symbol}"/>
                                <Text text="{price}"/>
                                <Text text="{volume}"/>
                            </cells>
                        </ColumnListItem>
                    </items>
                </Table>
            </VBox>
        </content>
    </Page>
</mvc:View>
""")
        
        # Create JSON configuration files
        (self.temp_dir / "package.json").write_text("""
{
    "name": "crypto-trading-platform",
    "version": "1.0.0",
    "description": "Enterprise crypto trading platform",
    "main": "index.js",
    "scripts": {
        "start": "node server.js",
        "build": "webpack --mode=production",
        "test": "jest"
    },
    "dependencies": {
        "@sap/cds": "^6.0.0",
        "express": "^4.18.0",
        "ws": "^8.0.0"
    },
    "devDependencies": {
        "webpack": "^5.0.0",
        "jest": "^29.0.0"
    }
}
""")
        
        (self.temp_dir / "manifest.json").write_text("""
{
    "sap.app": {
        "id": "crypto.trading",
        "type": "application"
    },
    "sap.ui5": {
        "dependencies": {
            "minUI5Version": "1.108.0",
            "libs": {
                "sap.m": {},
                "sap.ui.core": {}
            }
        }
    }
}
""")
        
        # Create YAML files
        (self.temp_dir / "docker-compose.yml").write_text("""
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
  database:
    image: postgres:14
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secret
""")
        
        return self.temp_dir
    
    def test_python_indexing(self, indexer: UnifiedLanguageIndexer) -> Dict[str, Any]:
        """Test Python file indexing"""
        print("🐍 Testing Python indexing...")
        
        result = indexer._index_python()
        
        # Verify Python facts
        python_facts = [f for f in result.get('glean_facts', []) 
                       if f.get('predicate') in ['src.File', 'python.Declaration', 'python.Reference']]
        
        # Check for expected symbols
        declarations = [f for f in python_facts if f.get('predicate') == 'python.Declaration']
        declaration_names = [f['key']['name'] for f in declarations if 'key' in f and 'name' in f['key']]
        
        expected_symbols = ['User', 'UserManager', 'add_user', 'get_user_by_email', 'main']
        found_symbols = [name for name in expected_symbols if any(name in decl_name for decl_name in declaration_names)]
        
        return {
            'files_processed': result.get('stats', {}).get('files_indexed', 0),
            'facts_generated': len(python_facts),
            'declarations_found': len(declarations),
            'expected_symbols_found': len(found_symbols),
            'expected_symbols': expected_symbols,
            'found_symbols': found_symbols,
            'success': len(found_symbols) >= 3  # At least 3 major symbols should be found
        }
    
    def test_javascript_indexing(self, indexer: UnifiedLanguageIndexer) -> Dict[str, Any]:
        """Test JavaScript/UI5 file indexing"""
        print("📱 Testing JavaScript/UI5 indexing...")
        
        result = indexer._index_javascript_ui5()
        
        # Verify JavaScript facts
        js_facts = [f for f in result.get('glean_facts', []) 
                   if f.get('predicate') in ['src.File', 'javascript.Function', 'ui5.Controller']]
        
        # Check for UI5 controller
        controllers = [f for f in js_facts if f.get('predicate') == 'ui5.Controller']
        functions = [f for f in js_facts if f.get('predicate') == 'javascript.Function']
        
        return {
            'files_processed': result.get('stats', {}).get('files_indexed', 0),
            'facts_generated': len(js_facts),
            'controllers_found': len(controllers),
            'functions_found': len(functions),
            'success': len(controllers) > 0 and len(functions) > 0
        }
    
    def test_typescript_indexing(self, indexer: UnifiedLanguageIndexer) -> Dict[str, Any]:
        """Test TypeScript file indexing"""
        print("⚡ Testing TypeScript indexing...")
        
        result = indexer._index_typescript()
        
        # Verify TypeScript facts
        ts_facts = [f for f in result.get('glean_facts', []) 
                   if f.get('predicate') in ['src.File', 'typescript.Declaration', 'typescript.Import']]
        
        # Check for expected TypeScript symbols
        declarations = [f for f in ts_facts if f.get('predicate') == 'typescript.Declaration']
        
        expected_ts_symbols = ['CryptoPrice', 'TradingEngine', 'OrderType', 'OrderStatus']
        found_ts_symbols = []
        
        for decl in declarations:
            if 'key' in decl and 'name' in decl['key']:
                name = decl['key']['name']
                if name in expected_ts_symbols:
                    found_ts_symbols.append(name)
        
        return {
            'files_processed': result.get('stats', {}).get('files_indexed', 0),
            'facts_generated': len(ts_facts),
            'declarations_found': len(declarations),
            'expected_symbols_found': len(found_ts_symbols),
            'expected_symbols': expected_ts_symbols,
            'found_symbols': found_ts_symbols,
            'success': len(found_ts_symbols) >= 2  # At least 2 TS symbols should be found
        }
    
    def test_cap_indexing(self, indexer: UnifiedLanguageIndexer) -> Dict[str, Any]:
        """Test SAP CAP file indexing"""
        print("🏢 Testing SAP CAP indexing...")
        
        result = indexer._index_cap()
        
        # Verify CAP facts
        cap_facts = [f for f in result.get('glean_facts', []) 
                    if f.get('predicate') in ['src.File', 'cap.Entity', 'cap.Service']]
        
        # Check for expected CAP entities
        entities = [f for f in cap_facts if f.get('predicate') == 'cap.Entity']
        services = [f for f in cap_facts if f.get('predicate') == 'cap.Service']
        
        expected_entities = ['Users', 'Trades']
        found_entities = []
        
        for entity in entities:
            if 'key' in entity and 'name' in entity['key']:
                name = entity['key']['name']
                if name in expected_entities:
                    found_entities.append(name)
        
        return {
            'files_processed': result.get('stats', {}).get('files_indexed', 0),
            'facts_generated': len(cap_facts),
            'entities_found': len(entities),
            'services_found': len(services),
            'expected_entities_found': len(found_entities),
            'expected_entities': expected_entities,
            'found_entities': found_entities,
            'success': len(found_entities) >= 1 and len(services) >= 1
        }
    
    def test_configuration_indexing(self, indexer: UnifiedLanguageIndexer) -> Dict[str, Any]:
        """Test configuration file indexing"""
        print("⚙️ Testing configuration file indexing...")
        
        result = indexer._index_configuration_files()
        
        # Verify config facts
        config_facts = result.get('glean_facts', [])
        
        # Check for JSON and YAML files
        json_files = [f for f in config_facts if f.get('predicate') == 'src.File' 
                     and f.get('value', {}).get('language') == 'JSON']
        yaml_files = [f for f in config_facts if f.get('predicate') == 'src.File' 
                     and f.get('value', {}).get('language') == 'YAML']
        
        return {
            'files_processed': result.get('files_indexed', 0),
            'facts_generated': len(config_facts),
            'json_files_found': len(json_files),
            'yaml_files_found': len(yaml_files),
            'success': len(json_files) >= 2 and len(yaml_files) >= 1  # Should find package.json, manifest.json, docker-compose.yml
        }
    
    def test_comprehensive_indexing(self, project_path: Path) -> Dict[str, Any]:
        """Test the complete multi-language indexing"""
        print("🔗 Testing comprehensive multi-language indexing...")
        
        result = index_multi_language_project(str(project_path))
        
        # Analyze results
        summary = result.get('indexing_summary', {})
        language_breakdown = result.get('language_breakdown', {})
        coverage = result.get('coverage_analysis', {})
        blind_spots = result.get('blind_spots_eliminated', {})
        
        return {
            'total_files_indexed': summary.get('total_files_indexed', 0),
            'total_facts_generated': summary.get('total_facts_generated', 0),
            'languages_supported': summary.get('languages_supported', 0),
            'indexing_duration': summary.get('indexing_duration_seconds', 0),
            'language_breakdown': language_breakdown,
            'coverage_percentage': coverage.get('coverage_percentage', 0),
            'blind_spots_count': blind_spots.get('blind_spots_count', 0),
            'coverage_complete': blind_spots.get('coverage_complete', False),
            'success': (
                summary.get('total_files_indexed', 0) > 0 and
                summary.get('total_facts_generated', 0) > 0 and
                coverage.get('coverage_percentage', 0) > 50
            )
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all indexer tests"""
        print("🚀 Starting Multi-Language Indexer Real Functionality Tests")
        print("=" * 60)
        
        try:
            # Setup test project
            project_path = self.setup_test_project()
            print(f"📁 Created test project at: {project_path}")
            
            # Initialize indexer
            indexer = UnifiedLanguageIndexer(project_path)
            
            # Run individual tests
            self.test_results['python'] = self.test_python_indexing(indexer)
            self.test_results['javascript'] = self.test_javascript_indexing(indexer)
            self.test_results['typescript'] = self.test_typescript_indexing(indexer)
            self.test_results['cap'] = self.test_cap_indexing(indexer)
            self.test_results['configuration'] = self.test_configuration_indexing(indexer)
            self.test_results['comprehensive'] = self.test_comprehensive_indexing(project_path)
            
            return self.test_results
            
        except Exception as e:
            print(f"❌ Test setup failed: {e}")
            return {'error': str(e)}
        
        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
    
    def print_results(self):
        """Print detailed test results"""
        print("\n" + "=" * 60)
        print("📊 MULTI-LANGUAGE INDEXER TEST RESULTS")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.test_results.items():
            if 'error' in result:
                continue
                
            total_tests += 1
            success = result.get('success', False)
            if success:
                passed_tests += 1
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"\n{status} {test_name.upper()} INDEXING:")
            
            if test_name == 'python':
                print(f"   Files processed: {result['files_processed']}")
                print(f"   Facts generated: {result['facts_generated']}")
                print(f"   Declarations found: {result['declarations_found']}")
                print(f"   Expected symbols found: {result['expected_symbols_found']}/{len(result['expected_symbols'])}")
                print(f"   Found symbols: {result['found_symbols']}")
                
            elif test_name == 'javascript':
                print(f"   Files processed: {result['files_processed']}")
                print(f"   Facts generated: {result['facts_generated']}")
                print(f"   Controllers found: {result['controllers_found']}")
                print(f"   Functions found: {result['functions_found']}")
                
            elif test_name == 'typescript':
                print(f"   Files processed: {result['files_processed']}")
                print(f"   Facts generated: {result['facts_generated']}")
                print(f"   Declarations found: {result['declarations_found']}")
                print(f"   Expected symbols found: {result['expected_symbols_found']}/{len(result['expected_symbols'])}")
                print(f"   Found symbols: {result['found_symbols']}")
                
            elif test_name == 'cap':
                print(f"   Files processed: {result['files_processed']}")
                print(f"   Facts generated: {result['facts_generated']}")
                print(f"   Entities found: {result['entities_found']}")
                print(f"   Services found: {result['services_found']}")
                print(f"   Expected entities found: {result['expected_entities_found']}/{len(result['expected_entities'])}")
                
            elif test_name == 'configuration':
                print(f"   Files processed: {result['files_processed']}")
                print(f"   Facts generated: {result['facts_generated']}")
                print(f"   JSON files found: {result['json_files_found']}")
                print(f"   YAML files found: {result['yaml_files_found']}")
                
            elif test_name == 'comprehensive':
                print(f"   Total files indexed: {result['total_files_indexed']}")
                print(f"   Total facts generated: {result['total_facts_generated']}")
                print(f"   Languages supported: {result['languages_supported']}")
                print(f"   Coverage percentage: {result['coverage_percentage']:.1f}%")
                print(f"   Blind spots count: {result['blind_spots_count']}")
                print(f"   Coverage complete: {result['coverage_complete']}")
        
        print(f"\n{'='*60}")
        print(f"🎯 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Multi-Language Indexer is fully functional!")
        else:
            print("⚠️  SOME TESTS FAILED - Issues detected in indexer functionality")
        
        print("=" * 60)

def main():
    """Main test execution"""
    tester = MultiLanguageIndexerTester()
    results = tester.run_all_tests()
    tester.print_results()
    
    # Return exit code based on results
    if 'error' in results:
        return 1
    
    all_passed = all(result.get('success', False) for result in results.values() if 'error' not in result)
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
