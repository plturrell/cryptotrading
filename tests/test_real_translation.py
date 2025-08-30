#!/usr/bin/env python3
"""
Test if the translation service is actually making real API calls
"""

import asyncio
import sys
import aiohttp
import ssl
import certifi

# Add src to path for imports
sys.path.insert(0, 'src')

from src.cryptotrading.infrastructure.data.news_service import AITranslationClient

async def test_real_translation():
    """Test if translation actually works with real API calls"""
    print("🔍 Testing Real Translation Service")
    print("=" * 50)
    
    # Test text to translate
    test_text = "Bitcoin price has reached a new all-time high of $95,000 today."
    
    # Create SSL context
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    connector = aiohttp.TCPConnector(
        ssl=ssl_context,
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300,
        use_dns_cache=True
    )
    
    async with aiohttp.ClientSession(
        connector=connector,
        headers={
            'Authorization': f'Bearer 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',
            'Content-Type': 'application/json',
            'User-Agent': 'CryptoTrading-NewsService/1.0'
        },
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        
        print(f"📝 Original text: {test_text}")
        print()
        
        # Test the translation client
        translator = AITranslationClient()
        
        print("🔄 Calling Perplexity API for translation...")
        translated_text = await translator.translate_to_russian(test_text, session)
        
        print(f"🇷🇺 Translated text: {translated_text}")
        print()
        
        # Check if translation actually happened
        if translated_text == test_text:
            print("❌ FAKE: Translation returned original text - API call failed or not made")
            return False
        elif len(translated_text) > 0 and translated_text != test_text:
            print("✅ REAL: Translation service is working - received different text")
            
            # Additional check - does it contain Cyrillic characters?
            has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in translated_text)
            if has_cyrillic:
                print("✅ REAL: Contains Cyrillic characters - genuine Russian translation")
                return True
            else:
                print("⚠️  SUSPICIOUS: No Cyrillic characters found in translation")
                return False
        else:
            print("❌ FAKE: Empty or invalid translation response")
            return False

async def test_direct_api_call():
    """Test direct API call to verify the service works"""
    print("\n🎯 Testing Direct Perplexity API Call")
    print("=" * 50)
    
    api_key = "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"
    
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a professional translator. Translate the given text to Russian."
            },
            {
                "role": "user",
                "content": "Translate this to Russian: Ethereum blockchain technology is revolutionary"
            }
        ],
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    # Create SSL context
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                print(f"📡 Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"✅ API Response: {content}")
                    
                    # Check if it's actually Russian
                    has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in content)
                    if has_cyrillic:
                        print("✅ REAL: API is working and returning Russian text")
                        return True
                    else:
                        print("⚠️  SUSPICIOUS: API working but not returning Russian")
                        return False
                else:
                    error_text = await response.text()
                    print(f"❌ API Error {response.status}: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Testing Translation Service Authenticity")
    print()
    
    # Test 1: Translation client
    result1 = await test_real_translation()
    
    # Test 2: Direct API call
    result2 = await test_direct_api_call()
    
    print("\n📊 FINAL VERDICT:")
    if result1 and result2:
        print("✅ REAL: Translation service is authentic and functional")
    elif result1 or result2:
        print("⚠️  PARTIAL: Some functionality working, needs investigation")
    else:
        print("❌ FAKE: Translation service is not working or fake")
    
    return result1 or result2

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1)
