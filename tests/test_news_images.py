#!/usr/bin/env python3
"""
Test script to check if Perplexity API can return images with news articles
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List

class NewsImageTester:
    def __init__(self):
        self.api_key = "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"
        self.base_url = "https://api.perplexity.ai/chat/completions"
    
    async def test_news_with_images(self) -> Dict[str, Any]:
        """Test if we can get images along with news articles"""
        
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a news aggregator. Provide recent cryptocurrency news with any available images, charts, or visual content."
                },
                {
                    "role": "user", 
                    "content": "Find the latest 3 cryptocurrency news articles with images, charts, or visual content. Include image URLs if available."
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
            "return_images": True,  # Request images in response
            "search_recency_filter": "day"
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Create SSL context that doesn't verify certificates (for testing)
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                async with session.post(self.base_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "response": data,
                            "has_images": self._check_for_images(data),
                            "content_length": len(data.get("choices", [{}])[0].get("message", {}).get("content", "")),
                            "search_results": data.get("search_results", [])
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Request failed: {str(e)}"
                }
    
    def _check_for_images(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the response contains any image references"""
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        search_results = data.get("search_results", [])
        
        # Look for image URLs in content
        image_urls_in_content = []
        lines = content.split('\n')
        for line in lines:
            if any(ext in line.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                if 'http' in line:
                    image_urls_in_content.append(line.strip())
        
        # Check search results for image fields
        search_images = []
        for result in search_results:
            if 'image' in result or 'images' in result:
                search_images.append(result)
        
        return {
            "image_urls_in_content": image_urls_in_content,
            "search_results_with_images": search_images,
            "total_search_results": len(search_results),
            "content_mentions_images": any(word in content.lower() for word in ['image', 'chart', 'graph', 'photo', 'picture'])
        }

async def main():
    """Test news image capabilities"""
    print("🔍 Testing Perplexity API for News Images...")
    print("=" * 60)
    
    tester = NewsImageTester()
    
    # Test news with images
    print("\n📰 Testing News Articles with Images...")
    result = await tester.test_news_with_images()
    
    if result["success"]:
        print(f"✅ Request successful!")
        print(f"📝 Content length: {result['content_length']} characters")
        print(f"🔍 Search results: {len(result.get('search_results', []))}")
        
        # Check for images
        image_check = result["has_images"]
        print(f"\n🖼️ Image Analysis:")
        print(f"   • Image URLs in content: {len(image_check['image_urls_in_content'])}")
        print(f"   • Search results with images: {len(image_check['search_results_with_images'])}")
        print(f"   • Content mentions images: {image_check['content_mentions_images']}")
        
        if image_check['image_urls_in_content']:
            print(f"\n🔗 Found Image URLs:")
            for url in image_check['image_urls_in_content']:
                print(f"   • {url}")
        
        # Show sample content
        content = result["response"].get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"\n📄 Sample Content (first 500 chars):")
        print(f"{content[:500]}...")
        
        # Show search results structure
        if result.get("search_results"):
            print(f"\n🔍 Search Results Structure:")
            for i, sr in enumerate(result["search_results"][:2]):  # Show first 2
                print(f"   Result {i+1} keys: {list(sr.keys())}")
                if 'url' in sr:
                    print(f"   URL: {sr['url']}")
    else:
        print(f"❌ Request failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
