#!/usr/bin/env python3
"""
Test script for CDS Service endpoints
Tests all CRUD operations, actions, and functions
"""

import requests
import json
from datetime import datetime, timedelta
import sys
from typing import Dict, List, Any
from colorama import init, Fore, Back, Style

# Initialize colorama for colored output
init(autoreset=True)

# Base URL for services
BASE_URL = "http://localhost:5001"
TRADING_SERVICE = f"{BASE_URL}/api/odata/v4/TradingService"
CODE_ANALYSIS_SERVICE = f"{BASE_URL}/api/odata/v4/CodeAnalysisService"

# Test data
TEST_TRADING_PAIR = {
    "id": "TEST-001",
    "symbol": "BTC/USD",
    "base_currency": "BTC",
    "quote_currency": "USD",
    "status": "active"
}

TEST_PROJECT = {
    "ID": "PROJ-001",
    "name": "Test Crypto Project",
    "status": "active",
    "language": "Python",
    "created_at": datetime.now().isoformat()
}

class CDSServiceTester:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    
    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{Back.BLUE}{Fore.WHITE} {text} {Style.RESET_ALL}")
    
    def print_test(self, name: str, passed: bool, details: str = ""):
        """Print test result"""
        if passed:
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} {name}")
            self.results["passed"] += 1
        else:
            print(f"{Fore.RED}✗{Style.RESET_ALL} {name}")
            if details:
                print(f"  {Fore.YELLOW}→ {details}{Style.RESET_ALL}")
            self.results["failed"] += 1
            self.results["errors"].append(f"{name}: {details}")
    
    def make_request(self, method: str, url: str, data: Dict = None) -> tuple:
        """Make HTTP request and return (success, response_data)"""
        try:
            if method == "GET":
                response = self.session.get(url)
            elif method == "POST":
                response = self.session.post(url, json=data)
            elif method == "PUT":
                response = self.session.put(url, json=data)
            elif method == "DELETE":
                response = self.session.delete(url)
            else:
                return False, f"Unknown method: {method}"
            
            if response.status_code in [200, 201, 204]:
                try:
                    return True, response.json() if response.content else {}
                except:
                    return True, {}
            else:
                return False, f"Status {response.status_code}: {response.text[:100]}"
        except requests.exceptions.ConnectionError:
            return False, "Connection refused - is the server running?"
        except Exception as e:
            return False, str(e)
    
    # ============== TRADING SERVICE TESTS ==============
    
    def test_trading_service_entities(self):
        """Test Trading Service entity CRUD operations"""
        self.print_header("TRADING SERVICE - ENTITY TESTS")
        
        # Test 1: GET all trading pairs
        success, data = self.make_request("GET", f"{TRADING_SERVICE}/TradingPairs")
        self.print_test("GET /TradingPairs (list all)", success, data if not success else "")
        
        # Test 2: POST new trading pair
        success, data = self.make_request("POST", f"{TRADING_SERVICE}/TradingPairs", TEST_TRADING_PAIR)
        self.print_test("POST /TradingPairs (create)", success, data if not success else "")
        created_id = data.get("id") if success else "TEST-001"
        
        # Test 3: GET specific trading pair
        success, data = self.make_request("GET", f"{TRADING_SERVICE}/TradingPairs/{created_id}")
        self.print_test(f"GET /TradingPairs/{created_id} (read one)", success, data if not success else "")
        
        # Test 4: PUT update trading pair
        update_data = TEST_TRADING_PAIR.copy()
        update_data["status"] = "inactive"
        success, data = self.make_request("PUT", f"{TRADING_SERVICE}/TradingPairs/{created_id}", update_data)
        self.print_test(f"PUT /TradingPairs/{created_id} (update)", success, data if not success else "")
        
        # Test 5: DELETE trading pair
        success, data = self.make_request("DELETE", f"{TRADING_SERVICE}/TradingPairs/{created_id}")
        self.print_test(f"DELETE /TradingPairs/{created_id} (delete)", success, data if not success else "")
        
        # Test other entities
        entities = ["Orders", "Portfolio", "MarketData", "OrderBook"]
        for entity in entities:
            success, data = self.make_request("GET", f"{TRADING_SERVICE}/{entity}")
            self.print_test(f"GET /{entity}", success, data if not success else "")
    
    def test_trading_service_actions(self):
        """Test Trading Service actions"""
        self.print_header("TRADING SERVICE - ACTION TESTS")
        
        # Test submitOrder action
        order_data = {
            "tradingPair": "BTC/USD",
            "orderType": "LIMIT",
            "orderMethod": "BUY",
            "quantity": 0.1,
            "price": 50000,
            "stopPrice": None,
            "timeInForce": "GTC"
        }
        success, data = self.make_request("POST", f"{TRADING_SERVICE}/submitOrder", order_data)
        self.print_test("POST /submitOrder", success, data if not success else "")
        
        # Test cancelOrder action
        cancel_data = {"orderId": "ORD-123"}
        success, data = self.make_request("POST", f"{TRADING_SERVICE}/cancelOrder", cancel_data)
        self.print_test("POST /cancelOrder", success, data if not success else "")
        
        # Test quickTrade action
        trade_data = {
            "symbol": "BTC",
            "orderType": "BUY",
            "amount": 100
        }
        success, data = self.make_request("POST", f"{TRADING_SERVICE}/quickTrade", trade_data)
        self.print_test("POST /quickTrade", success, data if not success else "")
        
        # Test rebalancePortfolio action
        rebalance_data = {
            "targetAllocations": [
                {"symbol": "BTC", "targetPercent": 50},
                {"symbol": "ETH", "targetPercent": 30},
                {"symbol": "USDT", "targetPercent": 20}
            ]
        }
        success, data = self.make_request("POST", f"{TRADING_SERVICE}/rebalancePortfolio", rebalance_data)
        self.print_test("POST /rebalancePortfolio", success, data if not success else "")
    
    def test_trading_service_functions(self):
        """Test Trading Service functions"""
        self.print_header("TRADING SERVICE - FUNCTION TESTS")
        
        # Test getOrderBook function
        success, data = self.make_request("GET", f"{TRADING_SERVICE}/getOrderBook?tradingPair=BTC-USD")
        self.print_test("GET /getOrderBook", success, data if not success else "")
        
        # Test getPriceHistory function
        params = "?tradingPair=BTC-USD&timeframe=1d"
        success, data = self.make_request("GET", f"{TRADING_SERVICE}/getPriceHistory{params}")
        self.print_test("GET /getPriceHistory", success, data if not success else "")
        
        # Test getMarketSummary function
        success, data = self.make_request("GET", f"{TRADING_SERVICE}/getMarketSummary")
        self.print_test("GET /getMarketSummary", success, data if not success else "")
        
        # Test calculateRiskMetrics function
        success, data = self.make_request("GET", f"{TRADING_SERVICE}/calculateRiskMetrics?portfolioId=default")
        self.print_test("GET /calculateRiskMetrics", success, data if not success else "")
        
        # Test validateOrder function
        params = "?tradingPair=BTC-USD&orderType=LIMIT&quantity=1&price=50000"
        success, data = self.make_request("GET", f"{TRADING_SERVICE}/validateOrder{params}")
        self.print_test("GET /validateOrder", success, data if not success else "")
    
    # ============== CODE ANALYSIS SERVICE TESTS ==============
    
    def test_code_analysis_service_entities(self):
        """Test Code Analysis Service entity CRUD operations"""
        self.print_header("CODE ANALYSIS SERVICE - ENTITY TESTS")
        
        # Test 1: GET all projects
        success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/Projects")
        self.print_test("GET /Projects (list all)", success, data if not success else "")
        
        # Test 2: POST new project
        success, data = self.make_request("POST", f"{CODE_ANALYSIS_SERVICE}/Projects", TEST_PROJECT)
        self.print_test("POST /Projects (create)", success, data if not success else "")
        created_id = data.get("ID") if success else "PROJ-001"
        
        # Test 3: GET specific project
        success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/Projects/{created_id}")
        self.print_test(f"GET /Projects/{created_id} (read one)", success, data if not success else "")
        
        # Test 4: PUT update project
        update_data = TEST_PROJECT.copy()
        update_data["status"] = "completed"
        success, data = self.make_request("PUT", f"{CODE_ANALYSIS_SERVICE}/Projects/{created_id}", update_data)
        self.print_test(f"PUT /Projects/{created_id} (update)", success, data if not success else "")
        
        # Test 5: DELETE project
        success, data = self.make_request("DELETE", f"{CODE_ANALYSIS_SERVICE}/Projects/{created_id}")
        self.print_test(f"DELETE /Projects/{created_id} (delete)", success, data if not success else "")
        
        # Test other entities
        entities = ["IndexingSessions", "CodeFiles", "AnalysisResults", "BlindSpots"]
        for entity in entities:
            success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/{entity}")
            self.print_test(f"GET /{entity}", success, data if not success else "")
    
    def test_code_analysis_service_actions(self):
        """Test Code Analysis Service actions"""
        self.print_header("CODE ANALYSIS SERVICE - ACTION TESTS")
        
        # Test startIndexing action
        indexing_data = {
            "projectId": "PROJ-001",
            "sessionName": "Test Indexing Session"
        }
        success, data = self.make_request("POST", f"{CODE_ANALYSIS_SERVICE}/startIndexing", indexing_data)
        self.print_test("POST /startIndexing", success, data if not success else "")
        session_id = data.get("sessionId") if success else "IDX-001"
        
        # Test stopIndexing action
        stop_data = {"sessionId": session_id}
        success, data = self.make_request("POST", f"{CODE_ANALYSIS_SERVICE}/stopIndexing", stop_data)
        self.print_test("POST /stopIndexing", success, data if not success else "")
        
        # Test validateResults action
        validate_data = {"sessionId": session_id}
        success, data = self.make_request("POST", f"{CODE_ANALYSIS_SERVICE}/validateResults", validate_data)
        self.print_test("POST /validateResults", success, data if not success else "")
        
        # Test exportResults action
        export_data = {
            "sessionId": session_id,
            "format": "json"
        }
        success, data = self.make_request("POST", f"{CODE_ANALYSIS_SERVICE}/exportResults", export_data)
        self.print_test("POST /exportResults", success, data if not success else "")
    
    def test_code_analysis_service_functions(self):
        """Test Code Analysis Service functions"""
        self.print_header("CODE ANALYSIS SERVICE - FUNCTION TESTS")
        
        # Test getAnalytics function
        success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/getAnalytics")
        self.print_test("GET /getAnalytics", success, data if not success else "")
        
        # Test getBlindSpotAnalysis function
        success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/getBlindSpotAnalysis")
        self.print_test("GET /getBlindSpotAnalysis", success, data if not success else "")
        
        # Test getPerformanceMetrics function
        success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/getPerformanceMetrics")
        self.print_test("GET /getPerformanceMetrics", success, data if not success else "")
    
    def print_summary(self):
        """Print test summary"""
        total = self.results["passed"] + self.results["failed"]
        success_rate = (self.results["passed"] / total * 100) if total > 0 else 0
        
        print(f"\n{Back.MAGENTA}{Fore.WHITE} TEST SUMMARY {Style.RESET_ALL}")
        print(f"Total Tests: {total}")
        print(f"{Fore.GREEN}Passed: {self.results['passed']}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {self.results['failed']}{Style.RESET_ALL}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.results["errors"]:
            print(f"\n{Fore.YELLOW}Failed Tests:{Style.RESET_ALL}")
            for error in self.results["errors"][:5]:  # Show first 5 errors
                print(f"  • {error}")
            if len(self.results["errors"]) > 5:
                print(f"  ... and {len(self.results['errors']) - 5} more")
        
        if success_rate == 100:
            print(f"\n{Back.GREEN}{Fore.WHITE} ✨ ALL TESTS PASSED! ✨ {Style.RESET_ALL}")
        elif success_rate >= 80:
            print(f"\n{Back.YELLOW}{Fore.BLACK} ⚠️  MOSTLY PASSING {Style.RESET_ALL}")
        else:
            print(f"\n{Back.RED}{Fore.WHITE} ❌ NEEDS ATTENTION {Style.RESET_ALL}")

def main():
    """Main test runner"""
    print(f"{Back.CYAN}{Fore.WHITE} CDS SERVICE TEST SUITE {Style.RESET_ALL}")
    print(f"Testing endpoints at: {BASE_URL}")
    print("="*50)
    
    tester = CDSServiceTester()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        print(f"{Fore.GREEN}✓ Server is running{Style.RESET_ALL}")
    except:
        print(f"{Fore.RED}✗ Server is not running at {BASE_URL}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Start the server with: python app.py{Style.RESET_ALL}")
        sys.exit(1)
    
    # Run all tests
    try:
        # Trading Service Tests
        tester.test_trading_service_entities()
        tester.test_trading_service_actions()
        tester.test_trading_service_functions()
        
        # Code Analysis Service Tests
        tester.test_code_analysis_service_entities()
        tester.test_code_analysis_service_actions()
        tester.test_code_analysis_service_functions()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Tests interrupted by user{Style.RESET_ALL}")
    
    # Print summary
    tester.print_summary()
    
    # Return exit code
    sys.exit(0 if tester.results["failed"] == 0 else 1)

if __name__ == "__main__":
    main()