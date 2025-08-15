"""
Configuration management for Koyfin scraper
Handles API keys, scraper settings, and environment variables
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ScraperConfig:
    """Configuration for Koyfin scraper"""
    # API Configuration
    xai_api_key: Optional[str] = None
    xai_api_base: str = "https://api.x.ai/v1"
    
    # Scraper Settings
    base_url: str = "https://www.koyfin.com/help/"
    max_depth: int = 3
    max_pages: int = 50
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 2
    rate_limit_delay: float = 1.0  # Seconds between requests
    
    # User Agent
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # Data Storage
    data_dir: str = "data/scraping_analysis"
    save_raw_html: bool = False
    save_intermediate_results: bool = True
    
    # Analysis Settings
    use_ai_analysis: bool = True
    ai_model: str = "grok-beta"
    ai_temperature: float = 0.3
    ai_max_tokens: int = 4096
    
    # Feature Extraction
    feature_patterns: list = field(default_factory=lambda: [
        r'(?i)(watchlist|portfolio|screen|chart|graph|dashboard|analysis|data|metric|indicator)',
        r'(?i)(filter|search|sort|export|import|alert|notification)',
        r'(?i)(historical|real-time|live|streaming|update)',
        r'(?i)(financial|economic|market|trading|investment)',
        r'(?i)(custom|template|model|formula|calculation)',
        r'(?i)(api|integration|webhook|data feed|connection)',
        r'(?i)(equity|stock|bond|commodity|crypto|forex|etf)',
        r'(?i)(fundamental|technical|quantitative|qualitative)',
        r'(?i)(earnings|revenue|profit|valuation|ratio)',
        r'(?i)(backtesting|simulation|optimization|strategy)'
    ])
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_env(cls) -> 'ScraperConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Load API key
        config.xai_api_key = os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY')
        
        # Load other settings from env
        if os.getenv('KOYFIN_BASE_URL'):
            config.base_url = os.getenv('KOYFIN_BASE_URL')
        if os.getenv('KOYFIN_MAX_DEPTH'):
            config.max_depth = int(os.getenv('KOYFIN_MAX_DEPTH'))
        if os.getenv('KOYFIN_MAX_PAGES'):
            config.max_pages = int(os.getenv('KOYFIN_MAX_PAGES'))
        if os.getenv('SCRAPER_DATA_DIR'):
            config.data_dir = os.getenv('SCRAPER_DATA_DIR')
        if os.getenv('SCRAPER_LOG_LEVEL'):
            config.log_level = os.getenv('SCRAPER_LOG_LEVEL')
            
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ScraperConfig':
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return cls.from_env()
            
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Create config from file data
            config = cls(**data)
            
            # Override with env vars if present
            env_key = os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY')
            if env_key:
                config.xai_api_key = env_key
                
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls.from_env()
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration settings"""
        errors = []
        
        # Check required settings
        if self.use_ai_analysis and not self.xai_api_key:
            errors.append("XAI_API_KEY is required when use_ai_analysis is True")
            
        if self.max_depth < 1:
            errors.append("max_depth must be at least 1")
            
        if self.max_pages < 1:
            errors.append("max_pages must be at least 1")
            
        if self.request_timeout < 5:
            errors.append("request_timeout should be at least 5 seconds")
            
        if self.rate_limit_delay < 0:
            errors.append("rate_limit_delay cannot be negative")
            
        # Validate URLs
        if not self.base_url.startswith(('http://', 'https://')):
            errors.append("base_url must start with http:// or https://")
            
        return len(errors) == 0, errors
    
    def setup_logging(self):
        """Configure logging based on settings"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=self.log_format
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'xai_api_key': '***' if self.xai_api_key else None,  # Mask API key
            'xai_api_base': self.xai_api_base,
            'base_url': self.base_url,
            'max_depth': self.max_depth,
            'max_pages': self.max_pages,
            'request_timeout': self.request_timeout,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'rate_limit_delay': self.rate_limit_delay,
            'user_agent': self.user_agent,
            'data_dir': self.data_dir,
            'save_raw_html': self.save_raw_html,
            'save_intermediate_results': self.save_intermediate_results,
            'use_ai_analysis': self.use_ai_analysis,
            'ai_model': self.ai_model,
            'ai_temperature': self.ai_temperature,
            'ai_max_tokens': self.ai_max_tokens,
            'feature_patterns': self.feature_patterns,
            'log_level': self.log_level
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't save API key to file
        data = self.to_dict()
        data['xai_api_key'] = None
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Configuration saved to {config_path}")


def get_default_config() -> ScraperConfig:
    """Get default configuration, trying multiple sources"""
    # Try loading from file first
    config_files = [
        'koyfin_scraper_config.json',
        'config/koyfin_scraper.json',
        '.koyfin_scraper_config.json',
        os.path.expanduser('~/.koyfin_scraper_config.json')
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            logger.info(f"Loading config from {config_file}")
            return ScraperConfig.from_file(config_file)
    
    # Fall back to environment variables
    logger.info("Loading config from environment variables")
    return ScraperConfig.from_env()