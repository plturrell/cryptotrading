# Koyfin Documentation Scraper

A production-ready web scraper for extracting and analyzing Koyfin's documentation to understand their features and capabilities. Uses AI-powered analysis with Grok-4 for intelligent insights.

## Features

- **Robust Web Scraping**: Automatic discovery and crawling of documentation pages
- **Intelligent Analysis**: AI-powered content analysis using Grok-4 or fallback methods
- **Feature Extraction**: Pattern-based identification of key features and capabilities
- **Error Handling**: Comprehensive retry logic, rate limiting, and failure recovery
- **Flexible Configuration**: Environment variables and config file support
- **Multiple Output Formats**: JSON data, analysis reports, and Markdown summaries

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cryptotrading
```

2. Install dependencies:
```bash
pip install -r src/rex/documentation/requirements.txt
```

3. Set up environment variables:
```bash
export XAI_API_KEY="your-grok-api-key"  # Optional, for AI analysis
```

## Quick Start

### Using the Scraper

```python
from src.rex.documentation.scraper import KoyfinScraperV2
from src.rex.documentation.config import get_default_config

# Create scraper with default config
config = get_default_config()
scraper = KoyfinScraperV2(config)

# Run full scrape
result = scraper.run_full_scrape()

if result['status'] == 'success':
    print(f"Scraped successfully!")
    print(f"Summary: {result['summary']}")
    print(f"Files saved: {result['saved_files']}")
```

### Basic Usage

```python
from src.rex.documentation.scraper import KoyfinScraperV2

# Initialize scraper
scraper = KoyfinScraperV2()

# Discover pages
urls = scraper.discover_pages()
print(f"Found {len(urls)} pages")

# Scrape specific pages
pages = scraper.scrape_pages(list(urls)[:10])  # First 10 pages

# Generate analysis report
report = scraper.generate_analysis_report(pages)

# Save results
saved_files = scraper.save_results(pages, report)
```

## Configuration

### Environment Variables

- `XAI_API_KEY` or `GROK_API_KEY`: API key for Grok-4 AI analysis (optional)
- `KOYFIN_BASE_URL`: Base URL to start scraping (default: https://www.koyfin.com/help/)
- `KOYFIN_MAX_DEPTH`: Maximum crawl depth (default: 3)
- `KOYFIN_MAX_PAGES`: Maximum pages to scrape (default: 50)
- `KOYFIN_DATA_DIR`: Directory for saving results (default: data/koyfin_analysis)
- `KOYFIN_LOG_LEVEL`: Logging level (default: INFO)

### Configuration File

Create a `koyfin_scraper_config.json`:

```json
{
  "base_url": "https://www.koyfin.com/help/",
  "max_depth": 3,
  "max_pages": 50,
  "request_timeout": 30,
  "retry_attempts": 3,
  "rate_limit_delay": 1.0,
  "use_ai_analysis": true,
  "ai_model": "grok-beta",
  "feature_patterns": [
    "(?i)(portfolio|watchlist|screen|chart)",
    "(?i)(real-time|historical|data|api)"
  ]
}
```

## API Reference

### KoyfinScraperV2

Main scraper class with production-ready features.

#### Methods

- `discover_pages(start_url=None, max_depth=None)`: Discover documentation pages
- `scrape_page(url)`: Scrape a single page
- `scrape_pages(urls, max_workers=5)`: Scrape multiple pages concurrently
- `generate_analysis_report(pages)`: Generate comprehensive analysis from scraped data
- `save_results(pages, report, prefix="scrape")`: Save results to files
- `run_full_scrape()`: Execute complete scraping workflow

### ScraperConfig

Configuration management class.

#### Methods

- `from_env()`: Load configuration from environment variables
- `from_file(config_path)`: Load configuration from JSON file
- `validate()`: Validate configuration settings
- `setup_logging()`: Configure logging based on settings

### AIAnalyzer

AI-powered content analysis module.

#### Methods

- `analyze_content(content, url, analysis_type="comprehensive")`: Analyze single page
- `batch_analyze(pages, analysis_type="comprehensive", max_concurrent=3)`: Analyze multiple pages
- `generate_insights_report(analyses)`: Generate insights from analyses

## Output Files

The scraper generates three types of output files:

1. **Pages Data** (`scrape_pages_YYYYMMDD_HHMMSS.json`):
   - Raw scraped content from each page
   - Extracted features and metadata
   - Navigation structure and links

2. **Analysis Report** (`scrape_report_YYYYMMDD_HHMMSS.json`):
   - Summary statistics
   - Top features by frequency
   - Features organized by category
   - Failed URLs and errors

3. **Markdown Summary** (`scrape_summary_YYYYMMDD_HHMMSS.md`):
   - Executive summary
   - Feature tables and categories
   - Implementation recommendations
   - Technical architecture suggestions

## Example Analysis Output

```json
{
  "summary": {
    "total_pages_scraped": 25,
    "successful_pages": 23,
    "failed_pages": 2,
    "total_features_found": 456,
    "unique_features": 89,
    "feature_categories": 11
  },
  "top_features": [
    {
      "feature": "portfolio",
      "category": "portfolio",
      "count": 34,
      "sample_occurrences": [...]
    },
    {
      "feature": "real-time",
      "category": "data",
      "count": 28,
      "sample_occurrences": [...]
    }
  ],
  "features_by_category": {
    "data": ["real-time", "historical", "api", ...],
    "visualization": ["chart", "graph", "dashboard", ...],
    "portfolio": ["watchlist", "holdings", "performance", ...]
  }
}
```

## Troubleshooting

### No URLs Discovered

- Check if the base URL is accessible
- Verify your internet connection
- Increase `max_depth` in configuration
- Check if the site structure has changed

### AI Analysis Not Working

- Verify `XAI_API_KEY` is set correctly
- Check API key permissions
- The scraper will fall back to pattern-based analysis if AI is unavailable

### Rate Limiting

- Increase `rate_limit_delay` in configuration
- Reduce `max_workers` for concurrent scraping
- Add proxy support if needed

### Memory Issues

- Reduce `max_pages` limit
- Enable `save_intermediate_results` to save progress
- Process pages in smaller batches

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
python test_scraper.py

# Test via CLI
python scraper_cli.py test
```

### Adding New Features

1. Add feature patterns to `ScraperConfig.feature_patterns`
2. Update `_categorize_feature()` method for new categories
3. Extend `AIAnalyzer` prompts for specific analysis types

## License

See LICENSE file in the project root.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues or questions:
- Create an issue on GitHub
- Check existing documentation
- Review test files for usage examples