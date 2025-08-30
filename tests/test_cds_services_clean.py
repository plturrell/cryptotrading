#!/usr/bin/env python3
"""
Test script for REAL CDS Service endpoints only
No fake trading, portfolio, or risk management tests
"""

import requests
import json
from datetime import datetime
import sys
from typing import Dict, List, Any
from colorama import init, Fore, Back, Style

# Initialize colorama for colored output
init(autoreset=True)

# Base URL for services
BASE_URL = "http://localhost:5001"
MARKET_DATA_SERVICE = f"{BASE_URL}/api/odata/v4/MarketDataService"
CODE_ANALYSIS_SERVICE = f"{BASE_URL}/api/odata/v4/CodeAnalysisService"
TECHNICAL_ANALYSIS_SERVICE = f"{BASE_URL}/api/odata/v4/TechnicalAnalysisService"

# Test data
TEST_PROJECT = {
    "name": "Test Crypto Project",
    "language": "Python"
}

class RealCDSServiceTester:
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
    
    # ============== MARKET DATA SERVICE TESTS (READ ONLY) ==============
    
    def test_market_data_service(self):
        """Test Market Data Service (READ ONLY)"""
        self.print_header("MARKET DATA SERVICE - READ ONLY TESTS")
        
        # Test 1: GET market data
        success, data = self.make_request("GET", f"{MARKET_DATA_SERVICE}/MarketData")
        self.print_test("GET /MarketData (read-only)", success, data if not success else "")
        
        # Test 2: GET price history
        success, data = self.make_request("GET", f"{MARKET_DATA_SERVICE}/PriceHistory?symbol=BTC&timeframe=1d")
        self.print_test("GET /PriceHistory (read-only)", success, data if not success else "")
        
        # Test 3: GET technical indicators
        success, data = self.make_request("GET", f"{MARKET_DATA_SERVICE}/TechnicalIndicators?symbol=BTC")
        self.print_test("GET /TechnicalIndicators (read-only)", success, data if not success else "")
    
    # ============== CODE ANALYSIS SERVICE TESTS ==============
    
    def test_code_analysis_service(self):
        """Test Code Analysis Service (REAL functionality)"""
        self.print_header("CODE ANALYSIS SERVICE - REAL FUNCTIONALITY")
        
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
        
        # Test indexing sessions
        success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/IndexingSessions")
        self.print_test("GET /IndexingSessions", success, data if not success else "")
        
        # Test indexing actions
        indexing_data = {
            "projectId": "PROJ-001",
            "sessionName": "Test Indexing Session"
        }
        success, data = self.make_request("POST", f"{CODE_ANALYSIS_SERVICE}/startIndexing", indexing_data)
        self.print_test("POST /startIndexing", success, data if not success else "")
        
        session_id = data.get("sessionId") if success else "IDX-001"
        stop_data = {"sessionId": session_id}
        success, data = self.make_request("POST", f"{CODE_ANALYSIS_SERVICE}/stopIndexing", stop_data)
        self.print_test("POST /stopIndexing", success, data if not success else "")
        
        validate_data = {"sessionId": session_id}
        success, data = self.make_request("POST", f"{CODE_ANALYSIS_SERVICE}/validateResults", validate_data)
        self.print_test("POST /validateResults", success, data if not success else "")
        
        # Test analytics functions
        success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/getAnalytics")
        self.print_test("GET /getAnalytics", success, data if not success else "")
        
        success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/getBlindSpotAnalysis")
        self.print_test("GET /getBlindSpotAnalysis", success, data if not success else "")
        
        success, data = self.make_request("GET", f"{CODE_ANALYSIS_SERVICE}/getPerformanceMetrics")
        self.print_test("GET /getPerformanceMetrics", success, data if not success else "")
    
    # ============== TECHNICAL ANALYSIS SERVICE TESTS ==============
    
    def test_technical_analysis_service(self):
        """Test Technical Analysis Service (READ ONLY analysis)"""
        self.print_header("TECHNICAL ANALYSIS SERVICE - READ ONLY ANALYSIS")
        
        # Test 1: Analyze technical data
        analysis_data = {"symbol": "BTC"}
        success, data = self.make_request("POST", f"{TECHNICAL_ANALYSIS_SERVICE}/analyze", analysis_data)
        self.print_test("POST /analyze (technical analysis)", success, data if not success else "")
        
        # Test 2: Get patterns
        success, data = self.make_request("GET", f"{TECHNICAL_ANALYSIS_SERVICE}/patterns?symbol=BTC")
        self.print_test("GET /patterns (chart patterns)", success, data if not success else "")
    
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
    print(f"{Back.CYAN}{Fore.WHITE} REAL CDS SERVICE TEST SUITE {Style.RESET_ALL}")
    print(f"Testing endpoints at: {BASE_URL}")
    print("No fake trading, portfolio, or risk management tests")
    print("="*50)
    
    tester = RealCDSServiceTester()
    
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
        # Market Data Service Tests (READ ONLY)
        tester.test_market_data_service()
        
        # Code Analysis Service Tests (REAL)
        tester.test_code_analysis_service()
        
        # Technical Analysis Service Tests (READ ONLY)
        tester.test_technical_analysis_service()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Tests interrupted by user{Style.RESET_ALL}")
    
    # Print summary
    tester.print_summary()
    
    # Return exit code
    sys.exit(0 if tester.results["failed"] == 0 else 1)

if __name__ == "__main__":
    main()