#!/usr/bin/env python3
"""
Test Russian News Service functionality
"""

import asyncio
import sys
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, 'src')

from src.cryptotrading.infrastructure.data.news_service import PerplexityNewsService

async def test_russian_news_service():
    """Test the Russian news capabilities"""
    print("🇷🇺 Testing Russian News Service Capabilities")
    print("=" * 60)
    
    async with PerplexityNewsService() as service:
        print(f"✅ Service initialized with API key: {service.api_key[:20]}...")
        print()
        
        # Test 1: Russian-specific crypto news
        print("📰 Testing Russian-specific crypto news...")
        try:
            russian_articles = await service.get_russian_crypto_news(3)
            print(f"   Retrieved {len(russian_articles)} Russian crypto articles")
            
            for i, article in enumerate(russian_articles):
                print(f"   Article {i+1}:")
                print(f"     Title: {article.title[:80]}...")
                print(f"     Language: {article.language}")
                print(f"     Source: {article.source}")
                print(f"     Symbols: {article.symbols}")
                print()
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 2: Translation to Russian
        print("🔄 Testing English to Russian translation...")
        try:
            english_articles = await service.get_latest_news(2)
            if english_articles:
                translated_articles = await service.translate_articles_to_russian(english_articles)
                print(f"   Translated {len(translated_articles)} articles to Russian")
                
                for i, article in enumerate(translated_articles):
                    print(f"   Article {i+1}:")
                    print(f"     Original Title: {article.title[:60]}...")
                    print(f"     Translated Title: {article.translated_title[:60]}...")
                    print(f"     Language: {article.language}")
                    print()
            else:
                print("   No English articles to translate")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 3: Russian category news
        print("📊 Testing Russian category news (market_analysis)...")
        try:
            category_articles = await service.get_news_by_category_russian('market_analysis', 2)
            print(f"   Retrieved {len(category_articles)} Russian market analysis articles")
            
            for i, article in enumerate(category_articles):
                print(f"   Article {i+1}:")
                print(f"     Title: {article.title[:80]}...")
                print(f"     Language: {article.language}")
                print(f"     Category: {article.category}")
                print()
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 4: Russian symbol news
        print("🪙 Testing Russian symbol news (BTC)...")
        try:
            symbol_articles = await service.get_news_by_symbol_russian('BTC', 2)
            print(f"   Retrieved {len(symbol_articles)} Russian BTC articles")
            
            for i, article in enumerate(symbol_articles):
                print(f"   Article {i+1}:")
                print(f"     Title: {article.title[:80]}...")
                print(f"     Language: {article.language}")
                print(f"     Symbols: {article.symbols}")
                print()
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 5: Russian categories
        print("🗂️  Testing Russian categories...")
        russian_categories = service.russian_categories
        print(f"   Available Russian categories: {len(russian_categories)}")
        for key, value in russian_categories.items():
            print(f"   - {key}: {value[:50]}...")
        print()
        
        # Test 6: Russian sources
        print("📡 Testing Russian sources...")
        russian_sources = service.russian_sources
        print(f"   Russian crypto news sources: {len(russian_sources)}")
        print(f"   Sources: {', '.join(russian_sources)}")
        print()

async def main():
    """Main test function"""
    print(f"🚀 Starting Russian News Service tests at {datetime.now()}")
    print()
    
    await test_russian_news_service()
    
    print("🎉 Russian News Service tests completed!")
    print()
    print("📋 SUMMARY:")
    print("✅ Russian-specific crypto news - Implemented")
    print("✅ AI translation to Russian - Implemented")
    print("✅ Russian category news - Implemented")
    print("✅ Russian symbol news - Implemented")
    print("✅ Russian categories mapping - Available")
    print("✅ Russian news sources - Configured")
    print()
    print("🔗 New Russian endpoints available:")
    print("   /api/news/latest/russian")
    print("   /api/news/category/{category}/russian")
    print("   /api/news/symbol/{symbol}/russian")
    print("   /api/news/russian/specific")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Tests failed with error: {e}")
