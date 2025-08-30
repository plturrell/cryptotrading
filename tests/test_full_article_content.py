#!/usr/bin/env python3
"""
Test to verify if we actually pull full article content from Perplexity API
"""

import asyncio
import sys
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, 'src')

from src.cryptotrading.infrastructure.data.news_service import PerplexityNewsService

async def test_full_article_content():
    """Test what depth of content we actually get from Perplexity"""
    print("🔍 Testing Full Article Content Retrieval")
    print("=" * 60)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    service = PerplexityNewsService()
    
    async with service:
        print("📰 TESTING: Latest Crypto News Content Depth")
        print("-" * 60)
        
        # Get latest news
        articles = await service.get_latest_news(limit=2)
        print(f"📥 Retrieved {len(articles)} articles")
        
        for i, article in enumerate(articles, 1):
            print(f"\n📄 ARTICLE {i} ANALYSIS:")
            print(f"   Title: {article.title}")
            print(f"   Title Length: {len(article.title)} characters")
            print()
            
            # Analyze content depth
            content = article.content or ""
            print(f"   Content Length: {len(content)} characters")
            print(f"   Content Word Count: {len(content.split())} words")
            print(f"   Content Lines: {len(content.splitlines())} lines")
            print()
            
            # Show content sample
            print("   Content Sample (first 300 chars):")
            print(f"   '{content[:300]}{'...' if len(content) > 300 else ''}'")
            print()
            
            # Check if it's actually full content or just summary
            if len(content) > 1000:
                print("   ✅ APPEARS TO BE FULL ARTICLE (>1000 chars)")
            elif len(content) > 500:
                print("   ⚠️  MEDIUM LENGTH CONTENT (500-1000 chars)")
            elif len(content) > 100:
                print("   ❌ SHORT CONTENT - LIKELY SUMMARY ONLY (<500 chars)")
            else:
                print("   ❌ VERY SHORT - LIKELY JUST HEADLINE/SNIPPET")
            
            print(f"   Source: {article.source}")
            print(f"   URL: {article.url}")
            print(f"   Published: {article.published_at}")
            print()
        
        print("=" * 60)
        print("🇷🇺 TESTING: Russian News Content Depth")
        print("-" * 60)
        
        # Test Russian news
        russian_articles = await service.get_russian_crypto_news(limit=1)
        print(f"📥 Retrieved {len(russian_articles)} Russian articles")
        
        if russian_articles:
            article = russian_articles[0]
            content = article.content or ""
            
            print(f"\n📄 RUSSIAN ARTICLE ANALYSIS:")
            print(f"   Title: {article.title}")
            print(f"   Content Length: {len(content)} characters")
            print(f"   Content Word Count: {len(content.split())} words")
            print()
            print("   Content Sample (first 300 chars):")
            print(f"   '{content[:300]}{'...' if len(content) > 300 else ''}'")
            print()
            
            # Check for Cyrillic characters
            has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in content)
            print(f"   Contains Cyrillic: {'✅ YES' if has_cyrillic else '❌ NO'}")
        
        print("\n" + "=" * 60)
        print("🔍 TESTING: Search Function Content Depth")
        print("-" * 60)
        
        # Test search function
        search_articles = await service.search_news("Bitcoin price analysis", limit=1)
        print(f"📥 Retrieved {len(search_articles)} search articles")
        
        if search_articles:
            article = search_articles[0]
            content = article.content or ""
            
            print(f"\n📄 SEARCH RESULT ANALYSIS:")
            print(f"   Query: 'Bitcoin price analysis'")
            print(f"   Title: {article.title}")
            print(f"   Content Length: {len(content)} characters")
            print(f"   Content Word Count: {len(content.split())} words")
            print()
            print("   Content Sample (first 400 chars):")
            print(f"   '{content[:400]}{'...' if len(content) > 400 else ''}'")
        
        print("\n" + "=" * 60)
        print("📊 FINAL ASSESSMENT:")
        print("-" * 60)
        
        total_articles = len(articles) + len(russian_articles) + len(search_articles)
        if total_articles > 0:
            avg_content_length = sum([
                len(a.content or "") for a in articles + russian_articles + search_articles
            ]) / total_articles
            
            print(f"   Total Articles Tested: {total_articles}")
            print(f"   Average Content Length: {avg_content_length:.0f} characters")
            print()
            
            if avg_content_length > 1000:
                print("   ✅ VERDICT: We ARE pulling substantial article content")
                print("   ✅ Content appears to be full articles, not just headlines")
            elif avg_content_length > 500:
                print("   ⚠️  VERDICT: We get moderate content - likely summaries")
                print("   ⚠️  May not be full articles, but more than headlines")
            else:
                print("   ❌ VERDICT: We get minimal content - likely just headlines/snippets")
                print("   ❌ Not pulling full article content")
            
            print()
            print("   CONTENT BREAKDOWN:")
            for i, article in enumerate(articles + russian_articles + search_articles, 1):
                content_len = len(article.content or "")
                content_type = "FULL" if content_len > 1000 else "MEDIUM" if content_len > 500 else "SHORT"
                print(f"   Article {i}: {content_len} chars ({content_type})")
        
        print("\n🎯 RECOMMENDATION:")
        if avg_content_length > 1000:
            print("   The current system IS pulling full article content.")
            print("   No changes needed - we have substantial content depth.")
        else:
            print("   The current system may NOT be pulling full articles.")
            print("   Consider enhancing the content extraction process.")

if __name__ == "__main__":
    try:
        asyncio.run(test_full_article_content())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
