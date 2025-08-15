# Documentation Module File Structure

This document describes the cleaned-up file structure for the documentation scraping module.

## Current Files

### Core Module (`src/rex/documentation/`)

```
src/rex/documentation/
├── __init__.py              # Module exports and imports
├── scraper.py               # Main scraper class (KoyfinScraperV2)
├── config.py                # Configuration management
├── ai_analyzer.py           # AI-powered content analysis  
├── cli.py                   # Command-line interface
├── requirements.txt         # Python dependencies
└── README.md               # Documentation
```

### Root Level Files

```
/
├── scraper                  # Main CLI entry point (executable)
├── scraper_cli.py          # Alternative CLI wrapper
├── test_scraper.py         # Comprehensive test suite
└── FILE_STRUCTURE.md       # This file
```

### Data Directory

```
data/
└── scraping_analysis/      # Output directory for scraper results
    ├── *.json              # Raw data and analysis reports
    └── *.md                # Markdown summaries
```

## File Naming Conventions

### Standard Python Conventions
- **Module files**: `snake_case.py` (e.g., `ai_analyzer.py`)
- **Class names**: `PascalCase` (e.g., `KoyfinScraperV2`)
- **Function names**: `snake_case` (e.g., `run_full_scrape`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_CONFIG`)

### Output File Conventions
- **Pages data**: `scrape_pages_YYYYMMDD_HHMMSS.json`
- **Reports**: `scrape_report_YYYYMMDD_HHMMSS.json`
- **Summaries**: `scrape_summary_YYYYMMDD_HHMMSS.md`
- **AI analysis**: `ai_analysis_YYYYMMDD_HHMMSS.json`

## Removed Files

The following old/duplicate files have been removed:

### Removed from `src/rex/documentation/`
- `a2a_koyfin_scraper.py` (old implementation)

### Removed from root
- `test_koyfin_scraper.py` (old test for v1)
- `test_direct_koyfin_scraper.py` (old direct test)
- `test_koyfin_tools_direct.py` (old tools test)
- `koyfin-scraper` (replaced by `scraper`)

### Moved/Renamed
- `koyfin_scraper_v2.py` → `scraper.py`
- `test_koyfin_scraper_v2.py` → `test_scraper.py`
- `data/koyfin_analysis/` → `data/scraping_analysis/`

## Usage Examples

### Import the module
```python
from src.rex.documentation import KoyfinScraperV2, ScraperConfig
```

### Use the CLI
```bash
# Main CLI
./scraper scrape --max-pages 50

# Alternative CLI
python3 scraper_cli.py scrape --max-pages 50

# Run tests
python3 test_scraper.py
```

### Environment Variables
```bash
# Renamed from KOYFIN_* to more generic names
export XAI_API_KEY="your-api-key"
export SCRAPER_DATA_DIR="custom/output/path"
export SCRAPER_LOG_LEVEL="DEBUG"
```

## Benefits of Cleanup

1. **Clearer naming**: No more version numbers in filenames
2. **Standard conventions**: Follows Python PEP 8 naming
3. **No duplicates**: Removed old/unused files
4. **Better organization**: Logical file structure
5. **Easier maintenance**: Clear responsibility for each file
6. **Generic naming**: Not tied to specific websites

## Migration Notes

If you have existing code that imports the old files:

```python
# OLD (no longer works)
from src.rex.documentation.koyfin_scraper_v2 import KoyfinScraperV2
from src.rex.documentation.a2a_koyfin_scraper import A2AKoyfinDocumentationScraper

# NEW (current)
from src.rex.documentation.scraper import KoyfinScraperV2
from src.rex.documentation import KoyfinScraperV2  # or via module
```