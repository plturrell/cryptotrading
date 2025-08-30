# Rex Crypto Trading Platform - Complete Deployment Guide

## 🚀 Three Unified Frameworks

### Framework 1: Build and Deploy New System
```bash
# Complete system build and deployment
python3 deploy/build_deploy_framework.py build-deploy
```

**What it does:**
- ✅ Installs all dependencies (Python + Node.js)
- ✅ Creates complete database schema with migrations
- ✅ Builds SAP UI5 frontend
- ✅ Creates deployment package
- ✅ Sets up GitHub Actions workflow
- ✅ Configures Vercel deployment

### Framework 2: Full System Startup
```bash
# Complete system startup
python3 startup.py

# Or using the framework directly
python3 deploy/build_deploy_framework.py startup
```

**What it does:**
- ✅ Initializes database with full schema
- ✅ Starts backend API services
- ✅ Prepares frontend services
- ✅ Runs comprehensive health check
- ✅ Reports system status

### Framework 3: GitHub and Vercel Deployment
```bash
# Deploy to GitHub and Vercel
./deploy.sh

# Or using the framework directly
python3 deploy/build_deploy_framework.py deploy
```

**What it does:**
- ✅ Sets up GitHub repository and Actions
- ✅ Configures Vercel deployment
- ✅ Creates environment files
- ✅ Generates deployment documentation

## 🎯 Quick Start Commands

### One-Command Deployment
```bash
# Complete deployment pipeline
./deploy.sh
```

### Individual Operations
```bash
# System health check
python3 startup.py health

# Build only
python3 startup.py build

# Deploy only
python3 startup.py deploy
```

## 🏗️ System Architecture

### Complete Database Schema
```sql
-- News Articles with Russian translation and images
CREATE TABLE news_articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    source TEXT,
    published_at TIMESTAMP,
    language TEXT DEFAULT 'en',
    
    -- Russian translation
    translated_title TEXT,
    translated_content TEXT,
    translation_status TEXT,
    
    -- Image support
    images TEXT, -- JSON array
    has_images BOOLEAN DEFAULT FALSE,
    image_count INTEGER DEFAULT 0,
    
    -- Metadata
    sentiment TEXT DEFAULT 'NEUTRAL',
    relevance_score DECIMAL(3,2),
    symbols TEXT, -- JSON array
    category TEXT
);

-- User searches and interactions
CREATE TABLE user_searches (...);
CREATE TABLE search_results (...);
CREATE TABLE user_interactions (...);
CREATE TABLE market_data (...);
CREATE TABLE system_config (...);
```

### API Endpoints
- `/api/news/latest` - Latest crypto news
- `/api/news/latest/russian` - Russian crypto news  
- `/api/news/translate` - Translate articles
- `/api/search` - User news search
- `/api/health` - System health check

### Frontend Features
- SAP UI5 with Fiori design
- Russian/English language toggle
- Image display (scraped, charts, search)
- Category filtering
- User search and history

## 🌐 Deployment Targets

### GitHub Repository
```bash
# Setup repository
git remote add origin https://github.com/plturrell/cryptotrading.git
git push -u origin main

# GitHub Actions will automatically:
# - Run tests
# - Test news service
# - Test image enhancement
# - Deploy to Vercel on main branch
```

### Vercel Deployment
```bash
# Install Vercel CLI
npm i -g vercel

# Login and deploy
vercel login
vercel --prod
```

**Vercel Configuration:**
- Python backend with `app_vercel.py`
- Static frontend serving
- Environment variables for API keys
- 30-second function timeout

## 🔧 Environment Variables

### Required for Production
```env
PERPLEXITY_API_KEY=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
ENABLE_RUSSIAN_TRANSLATION=true
ENABLE_IMAGE_ENHANCEMENT=true
NODE_ENV=production
FLASK_ENV=production
```

## 📊 System Features

### ✅ Russian Crypto News
- Real Perplexity API integration
- Professional AI translation
- Russian-specific news sources
- Cyrillic text support

### ✅ Image Enhancement
- **Web scraping**: Extract images from article URLs
- **Chart generation**: Real-time crypto price charts
- **Image search**: Crypto-related stock photos and logos
- **Complete metadata**: Alt text, captions, dimensions

### ✅ Database Integration
- Complete CDS schema
- News storage with images
- User search history
- Market data for charts
- System configuration

### ✅ SAP UI5 Frontend
- Professional Fiori design
- Responsive layout
- Image display support
- Language switching
- Category filtering

## 🏥 Health Monitoring

### System Health Check
```bash
python3 startup.py health
```

**Monitors:**
- ✅ Database connectivity
- ✅ Backend services
- ✅ Frontend availability
- ✅ News service functionality
- ✅ Image service availability

### API Health Endpoint
```bash
curl http://localhost:5000/api/health
```

## 📱 Usage Examples

### Fetch Enhanced News
```python
from src.cryptotrading.infrastructure.data.news_service import PerplexityNewsService

# Initialize with image enhancement
service = PerplexityNewsService(enable_images=True)

# Fetch and enhance articles
articles = await service.get_latest_news(limit=10)
enhanced = await service.enhance_articles_with_images(articles)

# Each article now has:
# - article.images: List[NewsImage]
# - article.has_images: bool
# - article.image_count: int
```

### Generate Price Charts
```python
from src.cryptotrading.infrastructure.data.image_services import CryptoPriceChartGenerator

generator = CryptoPriceChartGenerator()
chart = await generator.generate_price_chart('BTC', days=7, chart_type='candlestick')

# Returns base64 encoded chart image
print(f"Chart size: {len(chart.url)} bytes")
```

## 🚨 Troubleshooting

### Common Issues
1. **Database not found**: Run `python3 startup.py` to initialize
2. **API key errors**: Check environment variables
3. **Import errors**: Run `pip3 install -r requirements.txt`
4. **Frontend build fails**: Install Node.js and run `npm install`

### Debug Commands
```bash
# Check system health
python3 startup.py health

# Test news service
python3 test_real_translation.py

# Test image enhancement
python3 test_image_enhancement.py

# Full integration test
python3 demo_russian_translation.py
```

## 🎉 Deployment Checklist

- [ ] Run `./deploy.sh` for complete setup
- [ ] Push code to GitHub repository
- [ ] Connect repository to Vercel
- [ ] Set environment variables in Vercel dashboard
- [ ] Deploy with `vercel --prod`
- [ ] Test deployed application
- [ ] Monitor system health

## 📈 System Capabilities

**Content Depth:**
- Full article content (1000+ characters average)
- Professional Russian translations
- Multiple image sources per article
- Real-time price charts
- Complete metadata storage

**Performance:**
- Async operations throughout
- Database indexing for fast queries
- Image caching for charts
- Efficient web scraping

**Scalability:**
- Modular architecture
- Configurable limits
- Health monitoring
- Error handling and recovery

Your Rex Crypto Trading Platform is now ready for production deployment with complete Russian news integration and comprehensive image enhancement capabilities!
