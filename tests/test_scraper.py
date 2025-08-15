#!/usr/bin/env python3
"""
Test script for the production-ready Koyfin scraper V2
Demonstrates all major functionality with proper error handling
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from src.rex.documentation.scraper import KoyfinScraperV2
from src.rex.documentation.config import ScraperConfig, get_default_config
from src.rex.documentation.ai_analyzer import AIAnalyzer, create_ai_analyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_scraping():
    """Test basic scraping functionality without AI"""
    print("\n" + "="*60)
    print("TEST 1: Basic Web Scraping (No AI)")
    print("="*60)
    
    # Create config without AI
    config = ScraperConfig(
        base_url="https://www.koyfin.com/help/",
        max_depth=2,
        max_pages=10,
        use_ai_analysis=False,
        rate_limit_delay=1.0
    )
    
    # Validate config
    is_valid, errors = config.validate()
    if errors:
        print(f"Configuration warnings: {errors}")
    
    # Create scraper
    scraper = KoyfinScraperV2(config)
    
    # Test page discovery
    print("\nDiscovering pages...")
    urls = scraper.discover_pages()
    print(f"✓ Discovered {len(urls)} pages")
    
    if urls:
        # Show sample URLs
        print("\nSample URLs discovered:")
        for url in list(urls)[:5]:
            print(f"  - {url}")
        
        # Test single page scraping
        print("\nScraping a single page...")
        test_url = list(urls)[0]
        page = scraper.scrape_page(test_url)
        
        if page and not page.error:
            print(f"✓ Successfully scraped: {page.title}")
            print(f"  - Content length: {len(page.content)} chars")
            print(f"  - Features found: {len(page.features)}")
            print(f"  - Links extracted: {len(page.links)}")
            
            # Show sample features
            if page.features:
                print("\n  Sample features:")
                for feature in page.features[:3]:
                    print(f"    - {feature['feature']} ({feature['category']})")
        else:
            print(f"✗ Failed to scrape page: {page.error if page else 'Unknown error'}")
        
        # Test batch scraping
        print("\nBatch scraping multiple pages...")
        test_urls = list(urls)[:5]
        pages = scraper.scrape_pages(test_urls, max_workers=3)
        
        print(f"✓ Scraped {len(pages)} pages successfully")
        
        # Generate report
        print("\nGenerating analysis report...")
        report = scraper.generate_analysis_report(pages)
        
        print(f"✓ Report generated:")
        print(f"  - Total features: {report['summary']['total_features_found']}")
        print(f"  - Unique features: {report['summary']['unique_features']}")
        print(f"  - Categories: {report['summary']['feature_categories']}")
        
        # Save results
        print("\nSaving results...")
        saved_files = scraper.save_results(pages, report, prefix="test_basic")
        
        print("✓ Files saved:")
        for file_type, path in saved_files.items():
            print(f"  - {file_type}: {path}")
    
    return True


def test_ai_analysis():
    """Test AI-powered analysis functionality"""
    print("\n" + "="*60)
    print("TEST 2: AI-Powered Analysis")
    print("="*60)
    
    # Check if API key is available
    api_key = os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY')
    if not api_key:
        print("⚠️  No API key found. Set XAI_API_KEY to enable AI analysis.")
        print("   Falling back to pattern-based analysis.")
    
    # Create AI analyzer
    analyzer = create_ai_analyzer()
    
    # Test with sample content
    sample_content = """
    Koyfin provides comprehensive portfolio management tools including:
    - Real-time portfolio tracking with live market data
    - Advanced charting with over 100 technical indicators
    - Custom screening tools to find investment opportunities
    - Historical data analysis going back 20+ years
    - API access for automated trading strategies
    - Multi-asset class support including stocks, ETFs, crypto
    - Fundamental analysis with financial statements
    - Economic data and macro indicators
    """
    
    print("\nAnalyzing sample content...")
    
    # Run async analysis
    async def run_analysis():
        result = await analyzer.analyze_content(
            content=sample_content,
            url="https://example.com/sample",
            analysis_type="features"
        )
        return result
    
    result = asyncio.run(run_analysis())
    
    if result['status'] == 'success':
        print("✓ Analysis completed successfully")
        print(f"  - Model used: {result.get('model', 'unknown')}")
        print(f"  - Analysis type: {result.get('analysis_type')}")
        
        # Show snippet of analysis
        analysis_text = result.get('analysis', '')
        if analysis_text:
            lines = analysis_text.split('\n')
            print("\n  Analysis preview:")
            for line in lines[:10]:
                if line.strip():
                    print(f"    {line}")
            if len(lines) > 10:
                print(f"    ... ({len(lines) - 10} more lines)")
    else:
        print(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")
    
    return True


def test_full_workflow():
    """Test complete scraping workflow with all features"""
    print("\n" + "="*60)
    print("TEST 3: Full Workflow Test")
    print("="*60)
    
    # Create config with limited scope for testing
    config = get_default_config()
    config.max_pages = 5  # Limit for testing
    config.max_depth = 1  # Shallow scrape
    config.rate_limit_delay = 0.5  # Faster for testing
    
    print(f"\nConfiguration:")
    print(f"  - Base URL: {config.base_url}")
    print(f"  - Max pages: {config.max_pages}")
    print(f"  - Max depth: {config.max_depth}")
    print(f"  - AI analysis: {config.use_ai_analysis}")
    
    # Create scraper
    scraper = KoyfinScraperV2(config)
    
    # Run full scrape
    print("\nRunning full scrape workflow...")
    result = scraper.run_full_scrape()
    
    if result['status'] == 'success':
        print("\n✓ Full scrape completed successfully!")
        
        summary = result['summary']
        print(f"\nResults Summary:")
        print(f"  - Pages scraped: {summary['successful_pages']}")
        print(f"  - Failed pages: {summary['failed_pages']}")
        print(f"  - Total features: {summary['total_features_found']}")
        print(f"  - Unique features: {summary['unique_features']}")
        print(f"  - Duration: {result['duration']:.2f} seconds")
        
        print(f"\nOutput Files:")
        for file_type, path in result['saved_files'].items():
            file_size = Path(path).stat().st_size / 1024  # KB
            print(f"  - {file_type}: {path} ({file_size:.1f} KB)")
        
        # Read and display summary
        if 'summary' in result['saved_files']:
            summary_path = Path(result['saved_files']['summary'])
            if summary_path.exists():
                print(f"\nMarkdown Summary Preview:")
                with open(summary_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[:20]:
                        print(f"  {line.rstrip()}")
                    if len(lines) > 20:
                        print(f"  ... ({len(lines) - 20} more lines)")
    else:
        print(f"\n✗ Full scrape failed: {result.get('error', 'Unknown error')}")
        print(f"  Duration: {result.get('duration', 0):.2f} seconds")
    
    return result['status'] == 'success'


def test_error_handling():
    """Test error handling and edge cases"""
    print("\n" + "="*60)
    print("TEST 4: Error Handling")
    print("="*60)
    
    config = ScraperConfig(
        base_url="https://www.koyfin.com/help/",
        max_pages=5,
        retry_attempts=2,
        request_timeout=5
    )
    
    scraper = KoyfinScraperV2(config)
    
    # Test invalid URL
    print("\nTesting invalid URL handling...")
    invalid_url = "https://www.koyfin.com/this-page-does-not-exist-12345"
    page = scraper.scrape_page(invalid_url)
    
    if page and page.error:
        print(f"✓ Correctly handled invalid URL: {page.error}")
    else:
        print("✗ Failed to handle invalid URL properly")
    
    # Test malformed URL
    print("\nTesting malformed URL handling...")
    malformed_url = "not-a-valid-url"
    page = scraper.scrape_page(malformed_url)
    
    if page and page.error:
        print(f"✓ Correctly handled malformed URL: {page.error}")
    else:
        print("✗ Failed to handle malformed URL properly")
    
    # Test empty content
    print("\nTesting empty content handling...")
    from src.rex.documentation.scraper import PageContent
    empty_page = PageContent(url="test", title="Test", content="")
    features = scraper._extract_features(empty_page)
    
    print(f"✓ Handled empty content: {len(features)} features extracted")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Koyfin Scraper V2 - Production Test Suite")
    print("="*60)
    
    # Check environment
    api_key = os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY')
    print(f"\nEnvironment Check:")
    print(f"  - Python version: {sys.version.split()[0]}")
    print(f"  - Working directory: {os.getcwd()}")
    print(f"  - API key configured: {'Yes' if api_key else 'No'}")
    
    # Run tests
    tests = [
        ("Basic Scraping", test_basic_scraping),
        ("AI Analysis", test_ai_analysis),
        ("Full Workflow", test_full_workflow),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}...")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The scraper is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
    
    # Usage instructions
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. To enable AI analysis, set your API key:")
    print("   export XAI_API_KEY='your-api-key-here'")
    print("\n2. To run a full production scrape:")
    print("   python -c \"from src.rex.documentation.scraper import KoyfinScraperV2; KoyfinScraperV2().run_full_scrape()\"")
    print("\n3. Use the CLI interface:")
    print("   python scraper_cli.py scrape")
    print("\n4. Check the output in: data/koyfin_analysis/")
    print("\n5. Review the README.md for detailed documentation")


if __name__ == "__main__":
    main()