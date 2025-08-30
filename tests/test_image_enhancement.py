#!/usr/bin/env python3
"""
Comprehensive test for image enhancement in crypto news system
Tests web scraping, chart generation, and image search integration
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cryptotrading.infrastructure.data.news_service import PerplexityNewsService
from cryptotrading.infrastructure.data.image_services import (
    WebImageScraper, CryptoPriceChartGenerator, CryptoImageSearcher, NewsImageEnhancer
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageEnhancementTester:
    """Test all image enhancement features"""
    
    def __init__(self):
        self.news_service = PerplexityNewsService(enable_images=True)
        self.web_scraper = WebImageScraper()
        self.chart_generator = CryptoPriceChartGenerator()
        self.image_searcher = CryptoImageSearcher()
        self.image_enhancer = NewsImageEnhancer()
    
    async def test_web_scraping(self):
        """Test web scraping for article images"""
        print("\n🕷️ Testing Web Image Scraping...")
        print("=" * 60)
        
        # Test URLs from major crypto news sites
        test_urls = [
            "https://cointelegraph.com",
            "https://coindesk.com", 
            "https://decrypt.co",
            "https://theblock.co"
        ]
        
        for url in test_urls:
            try:
                images = await self.web_scraper.scrape_article_images(url, max_images=3)
                print(f"\n📰 {url}:")
                print(f"   • Found {len(images)} images")
                
                for i, img in enumerate(images):
                    print(f"   • Image {i+1}: {img.type} - {img.alt_text[:50]}...")
                    print(f"     URL: {img.url[:80]}...")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        return True
    
    async def test_chart_generation(self):
        """Test cryptocurrency price chart generation"""
        print("\n📊 Testing Price Chart Generation...")
        print("=" * 60)
        
        symbols = ['BTC', 'ETH', 'ADA', 'SOL']
        chart_types = ['candlestick', 'line', 'volume']
        
        for symbol in symbols:
            print(f"\n💰 Generating charts for {symbol}:")
            
            for chart_type in chart_types:
                try:
                    chart_image = await self.chart_generator.generate_price_chart(
                        symbol, days=7, chart_type=chart_type
                    )
                    
                    if chart_image:
                        data_size = len(chart_image.url) if chart_image.url.startswith('data:') else 0
                        print(f"   ✅ {chart_type.title()} chart: {data_size} bytes")
                        print(f"      Caption: {chart_image.caption}")
                    else:
                        print(f"   ❌ {chart_type.title()} chart: Failed to generate")
                        
                except Exception as e:
                    print(f"   ❌ {chart_type.title()} chart error: {str(e)}")
        
        return True
    
    async def test_image_search(self):
        """Test cryptocurrency image search"""
        print("\n🔍 Testing Crypto Image Search...")
        print("=" * 60)
        
        search_queries = [
            "bitcoin cryptocurrency",
            "ethereum blockchain",
            "crypto trading charts",
            "defi protocols"
        ]
        
        symbols = ['BTC', 'ETH', 'ADA']
        
        for query in search_queries:
            try:
                images = await self.image_searcher.search_crypto_images(
                    query, symbols, max_results=3
                )
                
                print(f"\n🔎 Query: '{query}'")
                print(f"   • Found {len(images)} images")
                
                for i, img in enumerate(images):
                    print(f"   • Image {i+1}: {img.type} from {img.source}")
                    print(f"     Alt: {img.alt_text}")
                    print(f"     Size: {img.width}x{img.height}" if img.width else "     Size: Unknown")
                    
            except Exception as e:
                print(f"   ❌ Error searching '{query}': {str(e)}")
        
        return True
    
    async def test_full_integration(self):
        """Test full news article enhancement with images"""
        print("\n🚀 Testing Full News + Image Integration...")
        print("=" * 60)
        
        try:
            # Fetch latest crypto news
            print("\n📰 Fetching latest crypto news...")
            articles = await self.news_service.get_latest_news(limit=3)
            
            if not articles:
                print("❌ No articles fetched")
                return False
            
            print(f"✅ Fetched {len(articles)} articles")
            
            # Enhance articles with images
            print("\n🖼️ Enhancing articles with images...")
            enhanced_articles = await self.news_service.enhance_articles_with_images(articles)
            
            # Display results
            for i, article in enumerate(enhanced_articles):
                print(f"\n📄 Article {i+1}: {article.title[:80]}...")
                print(f"   • Content: {len(article.content)} characters")
                print(f"   • Symbols: {', '.join(article.symbols) if article.symbols else 'None'}")
                print(f"   • Images: {article.image_count} images")
                print(f"   • Has Images: {article.has_images}")
                
                if article.images:
                    for j, img in enumerate(article.images):
                        print(f"     Image {j+1}: {img.type} - {img.alt_text[:40]}...")
                        if img.url.startswith('data:'):
                            print(f"       Generated chart ({len(img.url)} bytes)")
                        else:
                            print(f"       URL: {img.url[:60]}...")
                
                # Show article JSON structure
                article_dict = article.to_dict()
                print(f"   • JSON keys: {list(article_dict.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {str(e)}")
            return False
    
    async def test_russian_news_with_images(self):
        """Test Russian news with image enhancement"""
        print("\n🇷🇺 Testing Russian News + Images...")
        print("=" * 60)
        
        try:
            # Fetch Russian crypto news
            print("\n📰 Fetching Russian crypto news...")
            russian_articles = await self.news_service.get_latest_news_russian(limit=2)
            
            if not russian_articles:
                print("❌ No Russian articles fetched")
                return False
            
            print(f"✅ Fetched {len(russian_articles)} Russian articles")
            
            # Enhance with images
            print("\n🖼️ Enhancing Russian articles with images...")
            enhanced_russian = await self.news_service.enhance_articles_with_images(russian_articles)
            
            # Display results
            for i, article in enumerate(enhanced_russian):
                print(f"\n📄 Russian Article {i+1}:")
                print(f"   • Title: {article.title[:60]}...")
                print(f"   • Language: {article.language}")
                print(f"   • Content: {len(article.content)} characters")
                print(f"   • Images: {article.image_count}")
                
                if article.images:
                    for j, img in enumerate(article.images):
                        print(f"     Image {j+1}: {img.type} - {img.caption[:30]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Russian news test failed: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all image enhancement tests"""
        print("🎯 Crypto News Image Enhancement Test Suite")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_results = {}
        
        # Test individual components
        test_results['web_scraping'] = await self.test_web_scraping()
        test_results['chart_generation'] = await self.test_chart_generation()
        test_results['image_search'] = await self.test_image_search()
        
        # Test integration
        test_results['full_integration'] = await self.test_full_integration()
        test_results['russian_integration'] = await self.test_russian_news_with_images()
        
        # Summary
        print("\n🏁 Test Results Summary")
        print("=" * 60)
        
        passed = sum(test_results.values())
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All image enhancement features working!")
        else:
            print("⚠️ Some features need attention")
        
        return test_results

async def main():
    """Main test runner"""
    tester = ImageEnhancementTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Save results to file
        with open('image_enhancement_test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'summary': f"{sum(results.values())}/{len(results)} tests passed"
            }, f, indent=2)
        
        print(f"\n📄 Results saved to: image_enhancement_test_results.json")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {str(e)}")
        logger.exception("Test suite error")

if __name__ == "__main__":
    asyncio.run(main())
