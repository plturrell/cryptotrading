"""
Production-ready Koyfin Documentation Scraper V2
Robust web scraping with fallback mechanisms and proper error handling
"""

import asyncio
import aiohttp
import requests
from typing import Optional, List, Dict, Any, Set, Tuple
from datetime import datetime
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import re
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from .config import ScraperConfig, get_default_config

logger = logging.getLogger(__name__)

@dataclass
class PageContent:
    """Scraped page content with metadata"""
    url: str
    title: str
    content: str
    html: Optional[str] = None
    features: List[Dict[str, Any]] = None
    navigation: List[str] = None
    images: List[Dict[str, str]] = None
    links: List[Dict[str, str]] = None
    metadata: Dict[str, Any] = None
    scrape_timestamp: str = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []
        if self.navigation is None:
            self.navigation = []
        if self.images is None:
            self.images = []
        if self.links is None:
            self.links = []
        if self.metadata is None:
            self.metadata = {}
        if self.scrape_timestamp is None:
            self.scrape_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def get_text_hash(self) -> str:
        """Get hash of content for deduplication"""
        return hashlib.md5(self.content.encode()).hexdigest()


class KoyfinScraperV2:
    """
    Production-ready Koyfin documentation scraper with robust error handling
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or get_default_config()
        self.config.setup_logging()
        
        # Validate configuration
        is_valid, errors = self.config.validate()
        if not is_valid:
            logger.warning(f"Configuration validation errors: {errors}")
        
        # Initialize data directory
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for synchronous requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Tracking
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.scraped_pages: List[PageContent] = []
        self.discovered_urls: Set[str] = set()
        
        # Rate limiting
        self.last_request_time = 0
        
        logger.info(f"KoyfinScraperV2 initialized with config: {self.config.to_dict()}")
    
    def _rate_limit(self):
        """Implement rate limiting between requests"""
        if self.config.rate_limit_delay > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.config.rate_limit_delay:
                sleep_time = self.config.rate_limit_delay - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def _is_valid_help_url(self, url: str) -> bool:
        """Check if URL is a valid Koyfin help URL"""
        try:
            parsed = urlparse(url)
            # Check if it's a Koyfin domain
            if 'koyfin.com' not in parsed.netloc:
                return False
            # Check if it's in the help section
            if '/help' in parsed.path or '/support' in parsed.path or '/docs' in parsed.path:
                return True
            # Also accept main site features pages
            if any(keyword in parsed.path.lower() for keyword in ['features', 'product', 'solutions']):
                return True
            return False
        except Exception:
            return False
    
    def _extract_urls_from_page(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Extract all valid URLs from a page"""
        urls = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            
            # Clean up URL (remove fragments and trailing slashes)
            parsed = urlparse(full_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if clean_url.endswith('/'):
                clean_url = clean_url[:-1]
            
            if self._is_valid_help_url(clean_url):
                urls.add(clean_url)
        
        return urls
    
    def _fetch_page_with_retry(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch page content with retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                self._rate_limit()
                
                response = self.session.get(
                    url,
                    timeout=self.config.request_timeout,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return response.text, None
                elif response.status_code == 404:
                    return None, f"Page not found (404)"
                elif response.status_code == 429:
                    # Rate limited, wait longer
                    wait_time = self.config.retry_delay * (attempt + 1) * 2
                    logger.warning(f"Rate limited on {url}, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    error = f"HTTP {response.status_code}"
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    return None, error
                    
            except requests.exceptions.Timeout:
                error = "Request timeout"
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(f"Timeout on {url}, retry {attempt + 1}/{self.config.retry_attempts}")
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                return None, error
                
            except Exception as e:
                error = str(e)
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(f"Error fetching {url}: {error}, retry {attempt + 1}/{self.config.retry_attempts}")
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                return None, error
        
        return None, "Max retries exceeded"
    
    def _extract_page_content(self, html: str, url: str) -> PageContent:
        """Extract structured content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = "No Title"
        if soup.title:
            title = soup.title.get_text().strip()
        else:
            # Try meta title
            meta_title = soup.find('meta', property='og:title')
            if meta_title and meta_title.get('content'):
                title = meta_title['content']
        
        # Extract main content with multiple strategies
        content = ""
        content_selectors = [
            'main',
            '[role="main"]',
            '.help-content',
            '.documentation',
            '.content',
            'article',
            '.article-content',
            '#content',
            '.main-content',
            '.doc-content',
            '.page-content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # Get the element with the most text
                best_element = max(elements, key=lambda x: len(x.get_text(strip=True)))
                content = best_element.get_text(separator='\n', strip=True)
                if len(content) > 100:  # Minimum content threshold
                    break
        
        # Fallback: get body text but remove navigation/footer
        if len(content) < 100:
            # Remove unwanted elements
            for elem in soup.select('header, nav, footer, script, style'):
                elem.decompose()
            content = soup.get_text(separator='\n', strip=True)
        
        # Extract navigation
        navigation = []
        nav_selectors = ['nav', '.nav', '.navigation', '.sidebar', '.toc', '.table-of-contents']
        for selector in nav_selectors:
            for nav in soup.select(selector):
                nav_text = nav.get_text(separator=' | ', strip=True)
                if nav_text and len(nav_text) > 10:
                    navigation.append(nav_text)
        
        # Extract images with context
        images = []
        for img in soup.find_all('img', src=True):
            img_data = {
                'src': urljoin(url, img['src']),
                'alt': img.get('alt', ''),
                'title': img.get('title', ''),
                'width': img.get('width', ''),
                'height': img.get('height', '')
            }
            # Get surrounding text for context
            parent = img.parent
            if parent:
                context = parent.get_text(strip=True)[:200]
                img_data['context'] = context
            images.append(img_data)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            link_url = urljoin(url, link['href'])
            link_text = link.get_text(strip=True)
            if link_text:  # Only include links with text
                links.append({
                    'url': link_url,
                    'text': link_text,
                    'title': link.get('title', '')
                })
        
        # Extract metadata
        metadata = {
            'description': '',
            'keywords': '',
            'author': '',
            'last_modified': ''
        }
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', property='og:description')
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        # Last modified
        meta_modified = soup.find('meta', attrs={'name': 'last-modified'})
        if meta_modified:
            metadata['last_modified'] = meta_modified.get('content', '')
        
        return PageContent(
            url=url,
            title=title,
            content=content,
            html=html if self.config.save_raw_html else None,
            navigation=navigation,
            images=images,
            links=links,
            metadata=metadata
        )
    
    def _extract_features(self, page: PageContent) -> List[Dict[str, Any]]:
        """Extract features from page content using patterns"""
        features = []
        content = page.content
        
        if not content:
            return features
        
        # Track unique features to avoid duplicates
        seen_features = set()
        
        for pattern in self.config.feature_patterns:
            try:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    feature_text = match.group().lower()
                    
                    # Skip if we've seen this exact feature
                    if feature_text in seen_features:
                        continue
                    seen_features.add(feature_text)
                    
                    # Get context around the match
                    start = max(0, match.start() - 150)
                    end = min(len(content), match.end() + 150)
                    context = content[start:end].strip()
                    
                    # Clean up context
                    context = re.sub(r'\s+', ' ', context)
                    
                    # Extract sentence containing the feature
                    sentences = re.split(r'[.!?]+', context)
                    feature_sentence = None
                    for sentence in sentences:
                        if feature_text in sentence.lower():
                            feature_sentence = sentence.strip()
                            break
                    
                    feature_data = {
                        'feature': match.group(),
                        'category': self._categorize_feature(feature_text),
                        'context': context,
                        'sentence': feature_sentence,
                        'position': match.start(),
                        'source_url': page.url,
                        'source_title': page.title
                    }
                    
                    features.append(feature_data)
                    
            except Exception as e:
                logger.error(f"Error extracting features with pattern {pattern}: {e}")
                continue
        
        # Sort features by position in document
        features.sort(key=lambda x: x['position'])
        
        # Limit features per page to avoid explosion
        return features[:50]
    
    def _categorize_feature(self, feature: str) -> str:
        """Categorize a feature based on keywords"""
        feature_lower = feature.lower()
        
        categories = {
            'data': ['data', 'historical', 'real-time', 'live', 'feed', 'api', 'streaming'],
            'visualization': ['chart', 'graph', 'plot', 'visualization', 'dashboard', 'display'],
            'analysis': ['analysis', 'metric', 'indicator', 'calculation', 'formula', 'model'],
            'portfolio': ['portfolio', 'watchlist', 'holdings', 'position', 'allocation'],
            'screening': ['screen', 'filter', 'search', 'scan', 'criteria', 'query'],
            'alerts': ['alert', 'notification', 'warning', 'trigger', 'condition'],
            'integration': ['integration', 'api', 'webhook', 'export', 'import', 'connection'],
            'market_data': ['equity', 'stock', 'bond', 'commodity', 'crypto', 'forex', 'etf'],
            'fundamental': ['earnings', 'revenue', 'profit', 'valuation', 'ratio', 'fundamental'],
            'technical': ['technical', 'indicator', 'oscillator', 'moving average', 'rsi', 'macd'],
            'tools': ['tool', 'calculator', 'simulator', 'backtesting', 'optimization']
        }
        
        for category, keywords in categories.items():
            if any(keyword in feature_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def discover_pages(self, start_url: Optional[str] = None, max_depth: Optional[int] = None) -> Set[str]:
        """Discover all help pages using BFS"""
        start_url = start_url or self.config.base_url
        max_depth = max_depth or self.config.max_depth
        
        logger.info(f"Starting page discovery from {start_url} with max_depth={max_depth}")
        
        discovered = set()
        queue = [(start_url, 0)]
        
        while queue and len(discovered) < self.config.max_pages:
            url, depth = queue.pop(0)
            
            if depth > max_depth:
                continue
                
            if url in discovered or url in self.failed_urls:
                continue
            
            logger.debug(f"Discovering: {url} (depth={depth})")
            
            # Fetch page
            html, error = self._fetch_page_with_retry(url)
            
            if error:
                logger.warning(f"Failed to fetch {url}: {error}")
                self.failed_urls.add(url)
                continue
            
            discovered.add(url)
            
            # Parse and extract URLs
            try:
                soup = BeautifulSoup(html, 'html.parser')
                found_urls = self._extract_urls_from_page(soup, url)
                
                # Add new URLs to queue
                for found_url in found_urls:
                    if found_url not in discovered:
                        queue.append((found_url, depth + 1))
                        
                logger.info(f"Discovered {len(found_urls)} URLs from {url}")
                
            except Exception as e:
                logger.error(f"Error parsing {url}: {e}")
                continue
        
        logger.info(f"Discovery complete. Found {len(discovered)} pages")
        self.discovered_urls = discovered
        return discovered
    
    def scrape_page(self, url: str) -> Optional[PageContent]:
        """Scrape a single page"""
        logger.info(f"Scraping: {url}")
        
        if url in self.visited_urls:
            logger.debug(f"Already scraped: {url}")
            return None
        
        # Fetch page
        html, error = self._fetch_page_with_retry(url)
        
        if error:
            logger.error(f"Failed to scrape {url}: {error}")
            page = PageContent(
                url=url,
                title=f"Error: {error}",
                content="",
                error=error
            )
            self.failed_urls.add(url)
            return page
        
        try:
            # Extract content
            page = self._extract_page_content(html, url)
            
            # Extract features
            if self.config.feature_patterns:
                page.features = self._extract_features(page)
                logger.info(f"Extracted {len(page.features)} features from {url}")
            
            self.visited_urls.add(url)
            return page
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            page = PageContent(
                url=url,
                title=f"Processing Error",
                content="",
                error=str(e)
            )
            return page
    
    def scrape_pages(self, urls: List[str], max_workers: int = 5) -> List[PageContent]:
        """Scrape multiple pages concurrently"""
        logger.info(f"Scraping {len(urls)} pages with {max_workers} workers")
        
        pages = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_url = {executor.submit(self.scrape_page, url): url for url in urls}
            
            # Process completed tasks
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    page = future.result()
                    if page and not page.error:
                        pages.append(page)
                        self.scraped_pages.append(page)
                except Exception as e:
                    logger.error(f"Exception scraping {url}: {e}")
        
        logger.info(f"Scraped {len(pages)} pages successfully")
        return pages
    
    def generate_analysis_report(self, pages: List[PageContent]) -> Dict[str, Any]:
        """Generate comprehensive analysis report from scraped pages"""
        logger.info(f"Generating analysis report from {len(pages)} pages")
        
        # Aggregate all features
        all_features = {}
        feature_by_category = {}
        unique_pages = set()
        total_content_length = 0
        
        for page in pages:
            if page.error:
                continue
                
            unique_pages.add(page.url)
            total_content_length += len(page.content)
            
            for feature in page.features:
                feature_name = feature['feature'].lower()
                category = feature['category']
                
                # Track by feature name
                if feature_name not in all_features:
                    all_features[feature_name] = {
                        'count': 0,
                        'category': category,
                        'occurrences': []
                    }
                
                all_features[feature_name]['count'] += 1
                all_features[feature_name]['occurrences'].append({
                    'url': feature['source_url'],
                    'title': feature['source_title'],
                    'context': feature['context'],
                    'sentence': feature['sentence']
                })
                
                # Track by category
                if category not in feature_by_category:
                    feature_by_category[category] = []
                feature_by_category[category].append(feature_name)
        
        # Sort features by frequency
        sorted_features = sorted(all_features.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Build report
        report = {
            'summary': {
                'total_pages_scraped': len(pages),
                'successful_pages': len([p for p in pages if not p.error]),
                'failed_pages': len([p for p in pages if p.error]),
                'unique_urls': len(unique_pages),
                'total_content_length': total_content_length,
                'total_features_found': sum(len(p.features) for p in pages),
                'unique_features': len(all_features),
                'feature_categories': len(feature_by_category),
                'scrape_timestamp': datetime.now().isoformat()
            },
            'top_features': [
                {
                    'feature': name,
                    'category': data['category'],
                    'count': data['count'],
                    'sample_occurrences': data['occurrences'][:3]  # Top 3 examples
                }
                for name, data in sorted_features[:20]  # Top 20 features
            ],
            'features_by_category': {
                category: list(set(features))[:10]  # Top 10 unique features per category
                for category, features in feature_by_category.items()
            },
            'page_details': [
                {
                    'url': page.url,
                    'title': page.title,
                    'content_length': len(page.content),
                    'features_found': len(page.features),
                    'navigation_items': len(page.navigation),
                    'images': len(page.images),
                    'links': len(page.links),
                    'error': page.error
                }
                for page in pages[:50]  # Limit details to 50 pages
            ],
            'failed_urls': list(self.failed_urls)[:20]  # Top 20 failed URLs
        }
        
        return report
    
    def save_results(self, pages: List[PageContent], report: Dict[str, Any], 
                     prefix: str = "scrape") -> Dict[str, str]:
        """Save scraping results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        saved_files = {}
        
        # Save raw pages data
        if self.config.save_intermediate_results:
            pages_file = self.data_dir / f"{prefix}_pages_{timestamp}.json"
            pages_data = [page.to_dict() for page in pages if not page.error]
            with open(pages_file, 'w', encoding='utf-8') as f:
                json.dump(pages_data, f, indent=2, ensure_ascii=False)
            saved_files['pages'] = str(pages_file)
            logger.info(f"Saved {len(pages_data)} pages to {pages_file}")
        
        # Save analysis report
        report_file = self.data_dir / f"{prefix}_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        saved_files['report'] = str(report_file)
        logger.info(f"Saved analysis report to {report_file}")
        
        # Save markdown summary
        md_file = self.data_dir / f"{prefix}_summary_{timestamp}.md"
        self._save_markdown_summary(report, pages, md_file)
        saved_files['summary'] = str(md_file)
        logger.info(f"Saved markdown summary to {md_file}")
        
        return saved_files
    
    def _save_markdown_summary(self, report: Dict[str, Any], pages: List[PageContent], 
                               output_file: Path):
        """Generate and save markdown summary"""
        summary = report['summary']
        
        md_content = f"""# Documentation Scraping Report

**Generated:** {summary['scrape_timestamp']}

## Executive Summary

- **Total Pages Discovered:** {len(self.discovered_urls)}
- **Pages Successfully Scraped:** {summary['successful_pages']}
- **Failed Pages:** {summary['failed_pages']}
- **Total Features Identified:** {summary['total_features_found']}
- **Unique Features:** {summary['unique_features']}
- **Content Analyzed:** {summary['total_content_length']:,} characters

## Top Features Discovered

"""
        
        # Add top features table
        if report['top_features']:
            md_content += "| Feature | Category | Occurrences | Sample Context |\n"
            md_content += "|---------|----------|-------------|----------------|\n"
            
            for feature in report['top_features'][:15]:
                name = feature['feature']
                category = feature['category']
                count = feature['count']
                
                # Get sample context
                sample = ""
                if feature['sample_occurrences']:
                    context = feature['sample_occurrences'][0]['sentence'] or feature['sample_occurrences'][0]['context']
                    sample = context[:100] + "..." if len(context) > 100 else context
                    sample = sample.replace('|', '\\|').replace('\n', ' ')
                
                md_content += f"| {name} | {category} | {count} | {sample} |\n"
        
        # Add features by category
        md_content += "\n## Features by Category\n\n"
        
        for category, features in report['features_by_category'].items():
            if features:
                md_content += f"### {category.replace('_', ' ').title()}\n"
                for feature in features[:10]:
                    md_content += f"- {feature}\n"
                md_content += "\n"
        
        # Add implementation recommendations
        md_content += """## Implementation Recommendations

Based on the analysis of Koyfin's documentation, here are the key features to implement:

### High Priority
1. **Real-time Data Infrastructure**
   - Live market data feeds
   - Historical data storage
   - Data normalization pipeline

2. **Advanced Charting Engine**
   - Interactive charts with multiple indicators
   - Custom timeframes and intervals
   - Drawing tools and annotations

3. **Portfolio Management**
   - Multi-portfolio support
   - Performance tracking
   - Risk analytics

4. **Screening Tools**
   - Custom screeners with complex criteria
   - Saved screens and alerts
   - Backtesting capabilities

### Medium Priority
1. **API Integration**
   - RESTful API for data access
   - Webhook support for alerts
   - Third-party integrations

2. **Custom Dashboards**
   - Drag-and-drop interface
   - Widget library
   - Saved layouts

3. **Analysis Tools**
   - Financial ratios and metrics
   - Peer comparison
   - Custom formulas

## Technical Architecture

Based on the features identified, the recommended architecture includes:

1. **Frontend**: React/Vue.js with advanced charting libraries (TradingView, D3.js)
2. **Backend**: FastAPI/Django with WebSocket support for real-time data
3. **Database**: TimescaleDB for time-series data, PostgreSQL for user data
4. **Cache**: Redis for real-time data and session management
5. **Message Queue**: RabbitMQ/Kafka for data pipeline
6. **Infrastructure**: Kubernetes for scalability, CDN for static assets

## Data Sources Required

- Level 1/2 market data feeds
- Fundamental data providers
- Economic indicators
- News and sentiment data
- Alternative data sources
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def run_full_scrape(self) -> Dict[str, Any]:
        """Run complete scraping workflow"""
        logger.info("Starting full documentation scrape")
        
        start_time = time.time()
        
        # Step 1: Discover pages
        logger.info("Step 1: Discovering pages...")
        discovered_urls = self.discover_pages()
        
        if not discovered_urls:
            logger.error("No URLs discovered. Aborting scrape.")
            return {
                'status': 'error',
                'error': 'No URLs discovered',
                'duration': time.time() - start_time
            }
        
        # Step 2: Scrape pages
        logger.info(f"Step 2: Scraping {len(discovered_urls)} pages...")
        pages = self.scrape_pages(list(discovered_urls), max_workers=5)
        
        if not pages:
            logger.error("No pages successfully scraped. Aborting.")
            return {
                'status': 'error',
                'error': 'No pages scraped successfully',
                'discovered_urls': len(discovered_urls),
                'duration': time.time() - start_time
            }
        
        # Step 3: Generate analysis report
        logger.info("Step 3: Generating analysis report...")
        report = self.generate_analysis_report(pages)
        
        # Step 4: Save results
        logger.info("Step 4: Saving results...")
        saved_files = self.save_results(pages, report)
        
        duration = time.time() - start_time
        logger.info(f"Scraping complete in {duration:.2f} seconds")
        
        return {
            'status': 'success',
            'summary': report['summary'],
            'saved_files': saved_files,
            'duration': duration
        }


def main():
    """Main entry point for standalone execution"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create scraper with default config
    config = get_default_config()
    
    # Override some settings for demo
    config.max_pages = 20  # Limit for demo
    config.max_depth = 2   # Shallow scrape for demo
    
    scraper = KoyfinScraperV2(config)
    
    # Run full scrape
    result = scraper.run_full_scrape()
    
    if result['status'] == 'success':
        print(f"\nScraping completed successfully!")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"\nSummary:")
        for key, value in result['summary'].items():
            print(f"  {key}: {value}")
        print(f"\nSaved files:")
        for file_type, path in result['saved_files'].items():
            print(f"  {file_type}: {path}")
    else:
        print(f"\nScraping failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()