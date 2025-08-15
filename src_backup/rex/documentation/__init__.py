"""
Rex Documentation Analysis Module
Provides tools for web scraping and competitive analysis
"""

from .scraper import KoyfinScraperV2, PageContent
from .config import ScraperConfig, get_default_config
from .ai_analyzer import AIAnalyzer, create_ai_analyzer
from .cli import main as cli_main

__all__ = [
    'KoyfinScraperV2',
    'PageContent',
    'ScraperConfig',
    'get_default_config',
    'AIAnalyzer',
    'create_ai_analyzer',
    'cli_main'
]
