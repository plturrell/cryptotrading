#!/usr/bin/env python3
"""
Command-line interface for Koyfin Documentation Scraper
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import json
from datetime import datetime

from .scraper import KoyfinScraperV2
from .config import ScraperConfig, get_default_config
from .ai_analyzer import create_ai_analyzer


def setup_logging(level: str = "INFO"):
    """Configure logging for CLI"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'koyfin_scraper_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )


def cmd_scrape(args):
    """Execute scraping command"""
    # Load configuration
    if args.config:
        config = ScraperConfig.from_file(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.url:
        config.base_url = args.url
    if args.max_pages:
        config.max_pages = args.max_pages
    if args.max_depth:
        config.max_depth = args.max_depth
    if args.no_ai:
        config.use_ai_analysis = False
    if args.output_dir:
        config.data_dir = args.output_dir
    
    # Validate configuration
    is_valid, errors = config.validate()
    if not is_valid:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    # Show configuration
    print(f"Scraping Configuration:")
    print(f"  Base URL: {config.base_url}")
    print(f"  Max pages: {config.max_pages}")
    print(f"  Max depth: {config.max_depth}")
    print(f"  AI analysis: {'Enabled' if config.use_ai_analysis else 'Disabled'}")
    print(f"  Output dir: {config.data_dir}")
    print()
    
    # Create scraper
    scraper = KoyfinScraperV2(config)
    
    # Run scraping
    if args.discover_only:
        print("Discovering pages...")
        urls = scraper.discover_pages()
        print(f"\nDiscovered {len(urls)} pages:")
        for url in sorted(urls):
            print(f"  {url}")
        return 0
    
    print("Starting full scrape...")
    result = scraper.run_full_scrape()
    
    if result['status'] == 'success':
        summary = result['summary']
        print(f"\n✓ Scraping completed successfully!")
        print(f"  Pages scraped: {summary['successful_pages']}")
        print(f"  Features found: {summary['total_features_found']}")
        print(f"  Duration: {result['duration']:.2f} seconds")
        print(f"\nOutput files:")
        for file_type, path in result['saved_files'].items():
            print(f"  {file_type}: {path}")
        return 0
    else:
        print(f"\n✗ Scraping failed: {result.get('error', 'Unknown error')}")
        return 1


def cmd_analyze(args):
    """Execute analysis command on existing data"""
    # Find latest scraped data
    data_dir = Path(args.input_dir or "data/koyfin_analysis")
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    # Find pages files
    pages_files = sorted(data_dir.glob("*_pages_*.json"), reverse=True)
    
    if not pages_files:
        print(f"Error: No scraped data found in {data_dir}")
        return 1
    
    # Use latest or specified file
    if args.input_file:
        input_file = Path(args.input_file)
    else:
        input_file = pages_files[0]
        print(f"Using latest data file: {input_file.name}")
    
    # Load pages data
    try:
        with open(input_file, 'r') as f:
            pages_data = json.load(f)
        print(f"Loaded {len(pages_data)} pages")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Create AI analyzer
    analyzer = create_ai_analyzer()
    
    # Run analysis
    print(f"\nAnalyzing content with AI...")
    import asyncio
    
    async def run_analysis():
        # Convert to format expected by analyzer
        pages_for_analysis = [
            {
                'content': page.get('content', ''),
                'url': page.get('url', ''),
                'title': page.get('title', '')
            }
            for page in pages_data
            if page.get('content')
        ]
        
        # Limit pages if specified
        if args.max_pages:
            pages_for_analysis = pages_for_analysis[:args.max_pages]
        
        results = await analyzer.batch_analyze(
            pages_for_analysis,
            analysis_type=args.analysis_type,
            max_concurrent=3
        )
        
        return results
    
    results = asyncio.run(run_analysis())
    
    # Generate insights report
    insights = analyzer.generate_insights_report(results)
    
    # Save results
    output_file = data_dir / f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'analyses': results,
            'insights': insights,
            'metadata': {
                'input_file': str(input_file),
                'analysis_type': args.analysis_type,
                'pages_analyzed': len(results),
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)
    
    print(f"\n✓ Analysis completed!")
    print(f"  Pages analyzed: {insights['summary']['successful_analyses']}")
    print(f"  AI analyses: {insights['summary']['ai_analyses']}")
    print(f"  Key features: {len(insights['key_features'])}")
    print(f"\nResults saved to: {output_file}")
    
    # Show top features
    if insights['key_features']:
        print("\nTop features identified:")
        for feature, count in list(insights['key_features'].items())[:10]:
            print(f"  - {feature}: {count} occurrences")
    
    return 0


def cmd_config(args):
    """Manage configuration"""
    if args.generate:
        # Generate default config file
        config = get_default_config()
        output_file = args.output or "koyfin_scraper_config.json"
        config.save_to_file(output_file)
        print(f"✓ Configuration file generated: {output_file}")
        return 0
    
    if args.validate:
        # Validate config file
        if not args.file:
            print("Error: --file required for validation")
            return 1
        
        try:
            config = ScraperConfig.from_file(args.file)
            is_valid, errors = config.validate()
            
            if is_valid:
                print(f"✓ Configuration is valid")
                print(f"\nSettings:")
                for key, value in config.to_dict().items():
                    if key != 'feature_patterns':  # Skip long list
                        print(f"  {key}: {value}")
            else:
                print(f"✗ Configuration has errors:")
                for error in errors:
                    print(f"  - {error}")
                return 1
                
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return 1
    
    return 0


def cmd_test(args):
    """Run tests to verify scraper functionality"""
    print("Running Koyfin Scraper tests...\n")
    
    # Check environment
    api_key = os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY')
    print(f"Environment Check:")
    print(f"  API key configured: {'Yes' if api_key else 'No'}")
    print(f"  Working directory: {os.getcwd()}")
    
    # Test configuration
    print("\n1. Testing configuration...")
    config = get_default_config()
    is_valid, errors = config.validate()
    if is_valid:
        print("  ✓ Configuration is valid")
    else:
        print("  ✗ Configuration errors:", errors)
        if not args.continue_on_error:
            return 1
    
    # Test scraping (limited)
    print("\n2. Testing web scraping...")
    config.max_pages = 3
    config.max_depth = 1
    scraper = KoyfinScraperV2(config)
    
    urls = scraper.discover_pages()
    if urls:
        print(f"  ✓ Discovered {len(urls)} pages")
        
        # Test single page
        page = scraper.scrape_page(list(urls)[0])
        if page and not page.error:
            print(f"  ✓ Successfully scraped a page")
            print(f"    Title: {page.title}")
            print(f"    Content: {len(page.content)} chars")
            print(f"    Features: {len(page.features)}")
        else:
            print(f"  ✗ Failed to scrape page")
    else:
        print(f"  ✗ No pages discovered")
    
    # Test AI analysis
    print("\n3. Testing AI analysis...")
    analyzer = create_ai_analyzer()
    
    import asyncio
    async def test_ai():
        result = await analyzer.analyze_content(
            "Test content with portfolio and real-time data features",
            "https://test.com",
            "features"
        )
        return result
    
    result = asyncio.run(test_ai())
    if result['status'] == 'success':
        print(f"  ✓ AI analysis working")
        print(f"    Model: {result.get('model')}")
    else:
        print(f"  ⚠️  AI analysis not available (will use fallback)")
    
    print("\n✓ All tests completed!")
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Koyfin Documentation Scraper - Extract and analyze Koyfin features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full scrape with default settings
  python scraper_cli.py scrape
  
  # Scrape with custom settings
  python scraper_cli.py scrape --url https://koyfin.com/help/ --max-pages 100 --max-depth 4
  
  # Discover pages only
  python scraper_cli.py scrape --discover-only
  
  # Analyze existing data with AI
  python scraper_cli.py analyze --input-dir data/koyfin_analysis
  
  # Generate configuration file
  python scraper_cli.py config --generate
  
  # Test scraper functionality
  python scraper_cli.py test
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape Koyfin documentation')
    scrape_parser.add_argument('--url', help='Base URL to start scraping')
    scrape_parser.add_argument('--max-pages', type=int, help='Maximum pages to scrape')
    scrape_parser.add_argument('--max-depth', type=int, help='Maximum crawl depth')
    scrape_parser.add_argument('--config', help='Configuration file path')
    scrape_parser.add_argument('--output-dir', help='Output directory for results')
    scrape_parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis')
    scrape_parser.add_argument('--discover-only', action='store_true', 
                              help='Only discover URLs without scraping')
    scrape_parser.set_defaults(func=cmd_scrape)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze scraped data with AI')
    analyze_parser.add_argument('--input-dir', help='Directory containing scraped data')
    analyze_parser.add_argument('--input-file', help='Specific file to analyze')
    analyze_parser.add_argument('--analysis-type', 
                               choices=['comprehensive', 'features', 'technical'],
                               default='comprehensive', help='Type of analysis')
    analyze_parser.add_argument('--max-pages', type=int, help='Limit pages to analyze')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--generate', action='store_true', 
                              help='Generate default configuration file')
    config_parser.add_argument('--validate', action='store_true', 
                              help='Validate configuration file')
    config_parser.add_argument('--file', help='Configuration file to validate')
    config_parser.add_argument('--output', help='Output file for generated config')
    config_parser.set_defaults(func=cmd_config)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test scraper functionality')
    test_parser.add_argument('--continue-on-error', action='store_true',
                            help='Continue tests even if some fail')
    test_parser.set_defaults(func=cmd_test)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command:
        try:
            return args.func(args)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            return 130
        except Exception as e:
            logging.error(f"Command failed: {e}", exc_info=args.verbose)
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())