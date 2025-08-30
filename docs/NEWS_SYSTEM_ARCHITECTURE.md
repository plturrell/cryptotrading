# News System Architecture - Complete User Guide

## 🚀 News Loading Triggers & User Controls

### **1. Automatic News Loading Triggers**

**System-Initiated Loading:**
- **App Initialization**: News loads automatically when NewsPanel opens
- **Auto-Refresh**: Every 5 minutes (configurable)
- **Language Switch**: Triggers immediate reload
- **Category Change**: Triggers immediate reload
- **Page Refresh**: Reloads current view

**Database-First Strategy:**
```
1. Check database for cached articles
2. If insufficient articles → Fetch from Perplexity API
3. Store new articles in database
4. Display to user
```

### **2. User-Initiated Loading**

**Manual Controls Available:**
- ✅ **Refresh Button**: Force refresh current view
- ✅ **Fetch Fresh News Button**: Get latest from API
- ✅ **Load More Button**: Paginate through more articles
- ✅ **Language Toggle**: Switch between English/Russian
- ✅ **Category Filter**: Filter by news categories

**User Search Functionality:**
- ✅ **Custom Search**: Users can search for specific topics
- ✅ **Search History**: All searches are saved and can be rerun
- ✅ **Saved Searches**: Convert searches to recurring alerts
- ✅ **Search Results Storage**: All search results stored in database

## 🔍 User Search System

### **Search Capabilities**

**Search Types:**
```javascript
// Basic search
POST /api/search/news
{
  "user_id": 123,
  "query": "Bitcoin regulation",
  "language": "en",
  "limit": 10,
  "save_search": true
}

// Russian search
POST /api/search/news
{
  "user_id": 123,
  "query": "Ethereum DeFi",
  "language": "ru",
  "limit": 10,
  "save_search": true
}
```

**Search Features:**
- ✅ **Real-time Search**: Queries Perplexity API directly
- ✅ **Multi-language**: English and Russian support
- ✅ **Auto-save**: Searches saved to database by default
- ✅ **Search History**: View all previous searches
- ✅ **Rerun Searches**: Re-execute saved searches
- ✅ **Search Alerts**: Convert searches to recurring alerts

### **Search Storage & History**

**Database Tables:**
```sql
-- User searches table
user_searches:
- search_id (unique hash)
- user_id
- query
- language
- results_count
- search_timestamp
- parameters (JSON)

-- Search alerts table  
search_alerts:
- user_id
- alert_name
- search_query
- frequency (daily/weekly)
- is_active
- next_run
```

**Search History API:**
```javascript
// Get user's search history
GET /api/search/history/123?limit=20

// Rerun a saved search
POST /api/search/rerun
{
  "user_id": 123,
  "search_id": "abc123def456"
}

// Save search as alert
POST /api/search/save-alert
{
  "user_id": 123,
  "search_id": "abc123def456", 
  "alert_name": "Daily Bitcoin News"
}
```

## 📊 Data Depth & Content Storage

### **What Information We Load**

**Full Article Data (Not Just Headlines):**
```json
{
  "title": "Full article title",
  "content": "Complete article content (full text)",
  "summary": "AI-generated summary",
  "url": "Source URL",
  "source": "News source name",
  "author": "Article author",
  "published_at": "2025-08-30T10:51:14Z",
  "language": "en/ru",
  "category": "market_analysis",
  "symbols": ["BTC", "ETH"],
  "sentiment": "positive/negative/neutral",
  "relevance_score": 0.85,
  "translated_title": "Russian title",
  "translated_content": "Russian content",
  "tags": ["regulation", "SEC"],
  "view_count": 42,
  "reading_time": "3 min"
}
```

**Content Richness:**
- ✅ **Full Text**: Complete article content, not just headlines
- ✅ **AI Summaries**: Generated summaries for quick reading
- ✅ **Metadata**: Source, author, publish date, category
- ✅ **Crypto Context**: Related symbols, sentiment analysis
- ✅ **Translations**: Full Russian translations when requested
- ✅ **User Data**: View counts, bookmarks, interactions

### **Database Storage Strategy**

**Storage Process:**
1. **Fetch from Perplexity**: Get complete article data
2. **Process & Enhance**: Add sentiment, symbols, categories
3. **Store in Database**: Full content + metadata
4. **Cache for Performance**: Avoid repeated API calls
5. **Update UI**: Display rich, formatted content

**Storage Tables:**
```sql
news_articles:
- id, title, content, summary
- url, source, author, published_at
- language, category, symbols (JSON)
- sentiment, relevance_score
- translated_title, translated_content
- view_count, is_active
- search_id (if from user search)
```

## 🎯 User Experience Flow

### **Complete User Journey**

**1. Automatic Experience:**
```
User opens app → News loads from database/API → Displays in beautiful UI
Auto-refresh every 5 minutes → Always fresh content
```

**2. User-Controlled Experience:**
```
User clicks refresh → Fresh API call → New articles stored → UI updates
User changes language → Loads Russian content → Translations displayed
User filters category → Shows filtered view → Cached when possible
```

**3. Search Experience:**
```
User searches "DeFi regulation" → 
API call to Perplexity → 
Results stored in database → 
Search saved to history → 
User can rerun anytime → 
Can convert to recurring alert
```

**4. Personalization:**
```
User bookmarks articles → Stored in localStorage
User views articles → View count tracked
User search history → Saved in database
User preferences → Language, categories remembered
```

## 🔧 Technical Implementation

### **API Endpoints Summary**

**News Loading:**
- `GET /api/news/ui/latest` - UI-formatted latest news
- `GET /api/news/ui/russian` - Russian crypto news
- `POST /api/news/fetch/fresh` - Force fresh API fetch

**User Search:**
- `POST /api/search/news` - Perform user search
- `GET /api/search/history/{user_id}` - Get search history
- `POST /api/search/rerun` - Rerun saved search
- `POST /api/search/save-alert` - Save as recurring alert

**Article Management:**
- `POST /api/news/translate/{id}` - Translate article
- `GET /api/news/health` - Service health check

### **Data Flow Architecture**

```
User Action → 
Frontend (SAP UI5) → 
API Endpoint → 
Service Layer → 
Perplexity API (if needed) → 
Database Storage → 
Response to UI → 
Beautiful Display
```

**Key Features:**
- ✅ **Real API Integration**: Genuine Perplexity API calls
- ✅ **Full Content Storage**: Complete articles, not just headlines
- ✅ **User Search Control**: Custom searches with history
- ✅ **Automatic & Manual**: Both system and user-triggered loading
- ✅ **Multi-language**: English and Russian support
- ✅ **Rich UI**: Beautiful SAP Fiori interface with real-time updates

The system provides complete user control while maintaining automatic functionality for the best experience.
