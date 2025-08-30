#!/usr/bin/env python3
"""
Image Services for Crypto News Platform
Implements web scraping, chart generation, and image search for news articles
"""

import asyncio
import os
import logging
import ssl
import certifi
import aiohttp
import io
import base64
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import requests
from bs4 import BeautifulSoup
import yfinance as yf

try:
    from .news_service import NewsImage
except ImportError:
    # Fallback if relative import fails
    from src.cryptotrading.infrastructure.data.news_service import NewsImage

logger = logging.getLogger(__name__)

class WebImageScraper:
    """Scrapes images from news article URLs"""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def scrape_article_images(self, article_url: str, max_images: int = 5) -> List[NewsImage]:
        """Scrape images from a news article URL"""
        try:
            # Create SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector, headers=self.headers) as session:
                async with session.get(article_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch article: {response.status}")
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    images = []
                    
                    # Extract Open Graph image
                    og_image = soup.find('meta', property='og:image')
                    if og_image and og_image.get('content'):
                        img_url = self._resolve_url(og_image['content'], article_url)
                        if img_url:
                            images.append(NewsImage(
                                url=img_url,
                                alt_text="Article featured image",
                                type="featured",
                                source=urlparse(article_url).netloc
                            ))
                    
                    # Extract article images
                    article_images = soup.find_all('img', src=True)
                    for img in article_images[:max_images]:
                        img_url = self._resolve_url(img['src'], article_url)
                        if img_url and self._is_valid_image_url(img_url):
                            alt_text = img.get('alt', '')
                            
                            # Determine image type
                            img_type = self._classify_image_type(img_url, alt_text)
                            
                            images.append(NewsImage(
                                url=img_url,
                                alt_text=alt_text,
                                type=img_type,
                                source=urlparse(article_url).netloc,
                                caption=img.get('title', '')
                            ))
                    
                    # Remove duplicates
                    unique_images = []
                    seen_urls = set()
                    for img in images:
                        if img.url not in seen_urls:
                            unique_images.append(img)
                            seen_urls.add(img.url)
                    
                    logger.info(f"Scraped {len(unique_images)} images from {article_url}")
                    return unique_images[:max_images]
                    
        except Exception as e:
            logger.error(f"Error scraping images from {article_url}: {str(e)}")
            return []
    
    def _resolve_url(self, url: str, base_url: str) -> Optional[str]:
        """Resolve relative URLs to absolute URLs"""
        try:
            if url.startswith('http'):
                return url
            return urljoin(base_url, url)
        except Exception:
            return None
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL"""
        if not url:
            return False
        
        # Skip common non-content images
        skip_patterns = [
            'logo', 'avatar', 'icon', 'button', 'banner', 'ad',
            'tracking', 'pixel', '1x1', 'spacer', 'blank'
        ]
        
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in skip_patterns):
            return False
        
        # Check for image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
        return any(ext in url_lower for ext in image_extensions)
    
    def _classify_image_type(self, url: str, alt_text: str) -> str:
        """Classify image type based on URL and alt text"""
        url_lower = url.lower()
        alt_lower = alt_text.lower()
        
        if any(word in url_lower or word in alt_lower for word in ['chart', 'graph', 'price']):
            return 'chart'
        elif any(word in url_lower or word in alt_lower for word in ['infographic', 'diagram']):
            return 'infographic'
        elif any(word in url_lower or word in alt_lower for word in ['logo', 'brand']):
            return 'logo'
        else:
            return 'photo'

class CryptoPriceChartGenerator:
    """Generates price charts for cryptocurrency news articles"""
    
    def __init__(self):
        self.chart_cache = {}
        self.cache_duration = timedelta(hours=1)
    
    async def generate_price_chart(self, symbol: str, days: int = 7, chart_type: str = 'candlestick') -> Optional[NewsImage]:
        """Generate a price chart for a cryptocurrency symbol"""
        try:
            # Check cache
            cache_key = f"{symbol}_{days}_{chart_type}"
            if cache_key in self.chart_cache:
                cached_time, cached_image = self.chart_cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    return cached_image
            
            # Fetch price data
            price_data = await self._fetch_price_data(symbol, days)
            if price_data is None or price_data.empty:
                return None
            
            # Generate chart
            chart_buffer = self._create_chart(price_data, symbol, chart_type)
            if chart_buffer is None:
                return None
            
            # Convert to base64 data URL
            chart_b64 = base64.b64encode(chart_buffer.getvalue()).decode()
            data_url = f"data:image/png;base64,{chart_b64}"
            
            # Create NewsImage
            chart_image = NewsImage(
                url=data_url,
                alt_text=f"{symbol} {days}-day price chart",
                type="chart",
                source="generated",
                caption=f"{symbol} price movement over {days} days"
            )
            
            # Cache the result
            self.chart_cache[cache_key] = (datetime.now(), chart_image)
            
            logger.info(f"Generated {chart_type} chart for {symbol}")
            return chart_image
            
        except Exception as e:
            logger.error(f"Error generating chart for {symbol}: {str(e)}")
            return None
    
    async def _fetch_price_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency price data"""
        try:
            # Map crypto symbols to Yahoo Finance tickers
            symbol_map = {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD',
                'ADA': 'ADA-USD',
                'DOT': 'DOT-USD',
                'SOL': 'SOL-USD',
                'MATIC': 'MATIC-USD',
                'LINK': 'LINK-USD',
                'UNI': 'UNI-USD',
                'AVAX': 'AVAX-USD',
                'ATOM': 'ATOM-USD'
            }
            
            ticker = symbol_map.get(symbol.upper(), f"{symbol.upper()}-USD")
            
            # Fetch data using yfinance
            crypto = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = crypto.history(start=start_date, end=end_date, interval='1h' if days <= 7 else '1d')
            
            if data.empty:
                logger.warning(f"No price data found for {symbol}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {str(e)}")
            return None
    
    def _create_chart(self, data: pd.DataFrame, symbol: str, chart_type: str) -> Optional[io.BytesIO]:
        """Create price chart from data"""
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if chart_type == 'candlestick':
                self._create_candlestick_chart(ax, data, symbol)
            elif chart_type == 'line':
                self._create_line_chart(ax, data, symbol)
            elif chart_type == 'volume':
                self._create_volume_chart(ax, data, symbol)
            else:
                self._create_line_chart(ax, data, symbol)
            
            # Style the chart
            ax.set_facecolor('#1a1a1a')
            fig.patch.set_facecolor('#1a1a1a')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{symbol} Price Chart', fontsize=16, color='white', pad=20)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data) // 10)))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='#1a1a1a', edgecolor='none')
            buffer.seek(0)
            plt.close(fig)
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            return None
    
    def _create_candlestick_chart(self, ax, data: pd.DataFrame, symbol: str):
        """Create candlestick chart"""
        from matplotlib.patches import Rectangle
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            color = '#00ff88' if close_price >= open_price else '#ff4444'
            
            # Draw high-low line
            ax.plot([i, i], [low_price, high_price], color=color, linewidth=1)
            
            # Draw open-close rectangle
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            rect = Rectangle((i-0.3, bottom), 0.6, height, facecolor=color, alpha=0.8)
            ax.add_patch(rect)
        
        ax.set_xlim(-0.5, len(data)-0.5)
        ax.set_ylabel('Price (USD)', color='white')
    
    def _create_line_chart(self, ax, data: pd.DataFrame, symbol: str):
        """Create line chart"""
        ax.plot(range(len(data)), data['Close'], color='#00ff88', linewidth=2)
        ax.fill_between(range(len(data)), data['Close'], alpha=0.3, color='#00ff88')
        ax.set_ylabel('Price (USD)', color='white')
    
    def _create_volume_chart(self, ax, data: pd.DataFrame, symbol: str):
        """Create volume chart"""
        colors = ['#00ff88' if close >= open else '#ff4444' 
                 for close, open in zip(data['Close'], data['Open'])]
        ax.bar(range(len(data)), data['Volume'], color=colors, alpha=0.7)
        ax.set_ylabel('Volume', color='white')

class CryptoImageSearcher:
    """Searches for cryptocurrency-related images using web APIs"""
    
    def __init__(self, unsplash_access_key: str = None):
        self.unsplash_api_key = unsplash_access_key or os.getenv('UNSPLASH_ACCESS_KEY')
        if not self.unsplash_api_key:
            logger.warning("No Unsplash API key configured - image search will be disabled")
        self.session = None
    
    async def search_crypto_images(self, query: str, symbols: List[str] = None, max_results: int = 3) -> List[NewsImage]:
        """Search for cryptocurrency-related images"""
        try:
            images = []
            
            # Search Unsplash for crypto images
            unsplash_images = await self._search_unsplash(query, max_results)
            images.extend(unsplash_images)
            
            # Search for symbol-specific images
            if symbols:
                for symbol in symbols[:2]:  # Limit to 2 symbols
                    symbol_images = await self._search_symbol_images(symbol, 1)
                    images.extend(symbol_images)
            
            return images[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching crypto images: {str(e)}")
            return []
    
    async def _search_unsplash(self, query: str, max_results: int) -> List[NewsImage]:
        """Search Unsplash for images using real API"""
        try:
            if not self.unsplash_api_key:
                logger.warning("Unsplash API key not configured")
                return []
            
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Client-ID {self.unsplash_api_key}'}
                params = {
                    'query': query,
                    'per_page': min(max_results, 10),
                    'orientation': 'landscape'
                }
                
                async with session.get(
                    'https://api.unsplash.com/search/photos',
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        images = []
                        
                        for photo in data.get('results', []):
                            images.append(NewsImage(
                                url=photo['urls']['regular'],
                                alt_text=photo.get('alt_description', f'Image related to {query}'),
                                type="photo",
                                source="unsplash",
                                width=photo['width'],
                                height=photo['height'],
                                caption=photo.get('description', '')
                            ))
                        
                        return images
                    else:
                        logger.error(f"Unsplash API error: {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"Error searching Unsplash: {str(e)}")
            return []
    
    async def _search_symbol_images(self, symbol: str, max_results: int) -> List[NewsImage]:
        """Search for symbol-specific images from real sources"""
        try:
            # Use real crypto icon APIs
            symbol_lower = symbol.lower()
            images = []
            
            # Try CoinGecko API for real crypto logos
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f'https://api.coingecko.com/api/v3/coins/{symbol_lower}') as response:
                        if response.status == 200:
                            data = await response.json()
                            logo_url = data.get('image', {}).get('large')
                            if logo_url:
                                images.append(NewsImage(
                                    url=logo_url,
                                    alt_text=f"{symbol} cryptocurrency logo",
                                    type="logo",
                                    source="coingecko",
                                    width=200,
                                    height=200,
                                    caption=f"Official {symbol} logo"
                                ))
                except Exception as e:
                    logger.warning(f"Could not fetch CoinGecko logo for {symbol}: {e}")
            
            # Fallback to CryptoIcons if available
            fallback_url = f"https://cryptoicons.org/api/icon/{symbol_lower}/200"
            
            if not images:
                images.append(NewsImage(
                    url=fallback_url,
                    alt_text=f"{symbol} cryptocurrency logo",
                    type="logo",
                    source="cryptoicons",
                    width=200,
                    height=200,
                    caption=f"{symbol} official logo"
                ))
            
            return images
            
        except Exception as e:
            logger.error(f"Error searching symbol images for {symbol}: {str(e)}")
            return []

class NewsImageEnhancer:
    """Main service that combines all image enhancement methods"""
    
    def __init__(self, unsplash_key: str = None):
        self.web_scraper = WebImageScraper()
        self.chart_generator = CryptoPriceChartGenerator()
        self.image_searcher = CryptoImageSearcher(unsplash_key)
    
    async def enhance_article_with_images(self, article, max_images: int = 5) -> List[NewsImage]:
        """Enhance a news article with images from all sources"""
        try:
            all_images = []
            
            # 1. Scrape images from article URL
            if article.url:
                scraped_images = await self.web_scraper.scrape_article_images(article.url, 3)
                all_images.extend(scraped_images)
            
            # 2. Generate price charts for mentioned symbols
            if article.symbols:
                for symbol in article.symbols[:2]:  # Limit to 2 charts
                    chart_image = await self.chart_generator.generate_price_chart(symbol, days=7)
                    if chart_image:
                        all_images.append(chart_image)
            
            # 3. Search for related crypto images
            search_query = f"{article.title} cryptocurrency"
            search_images = await self.image_searcher.search_crypto_images(
                search_query, article.symbols, 2
            )
            all_images.extend(search_images)
            
            # Remove duplicates and limit results
            unique_images = []
            seen_urls = set()
            for img in all_images:
                if img.url not in seen_urls:
                    unique_images.append(img)
                    seen_urls.add(img.url)
            
            result_images = unique_images[:max_images]
            logger.info(f"Enhanced article with {len(result_images)} images")
            
            return result_images
            
        except Exception as e:
            logger.error(f"Error enhancing article with images: {str(e)}")
            return []
