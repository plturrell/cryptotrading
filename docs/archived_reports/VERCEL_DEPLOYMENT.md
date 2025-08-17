# Vercel ML Deployment Guide

This guide explains how to deploy the ML prediction system to Vercel with **REAL data only** - no mocks or dummy values.

## 🚀 Quick Start

### 1. Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Test locally
python scripts/test_local_ml.py
```

### 2. Train ML Models

```bash
# Train models locally (required for predictions to work)
python scripts/export_models_for_vercel.py
# Choose option 1 to train new models
```

### 3. Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Deploy to production
vercel --prod
```

## 📁 Project Structure

```
cryptotrading/
├── api/
│   └── ml/
│       ├── predict.py      # Single prediction endpoint
│       ├── batch.py        # Batch predictions endpoint
│       └── features.py     # Feature extraction endpoint
├── src/
│   └── cryptotrading/
│       └── core/
│           ├── ml/         # ML models and training
│           ├── storage/    # Vercel KV adapter
│           └── monitoring/ # Lightweight monitoring
├── vercel.json            # Vercel configuration
└── models/               # Trained model storage
```

## 🔧 Key Changes Made for Vercel

### 1. **Vercel KV Storage** (Replaces Redis)
- Automatic fallback to file-based cache for local development
- No configuration needed locally
- Production uses Vercel KV when deployed

### 2. **Lightweight ML Models**
- Heavy sklearn models trained locally
- Exported as simple coefficients for edge functions
- Fast inference with minimal dependencies

### 3. **Real Data Only**
- All predictions use live Yahoo Finance data
- No dummy values or mocks
- Fails with proper errors if data unavailable

### 4. **Simplified Monitoring**
- Removed heavy OpenTelemetry dependencies
- Simple logging that works in serverless
- Metrics printed to Vercel logs

## 🌐 API Endpoints

### GET `/api/ml/predict?symbol=BTC&horizon=24h`
Returns real price prediction or fails with error.

### POST `/api/ml/batch`
```json
{
  "symbols": ["BTC", "ETH"],
  "horizon": "24h"
}
```

### GET `/api/ml/features?symbol=BTC`
Returns real calculated features from market data.

## ⚠️ Important Notes

1. **Models Must Be Trained**: The system will fail (as designed) without trained models
2. **Real Data Required**: Yahoo Finance must be accessible for predictions
3. **No Fallbacks**: System prefers to fail rather than return fake data
4. **Local Testing**: Use `scripts/test_local_ml.py` before deployment

## 🔑 Environment Variables

Set in Vercel dashboard or `.env`:

```bash
# Optional - for Vercel KV (auto-configured on Vercel)
KV_REST_API_URL=your_kv_url
KV_REST_API_TOKEN=your_kv_token

# ML Models (created by export script)
ML_MODELS_CONFIG=base64_encoded_models
```

## 🧪 Testing

### Local Testing
```bash
# Test all components
python scripts/test_local_ml.py

# Test specific API endpoint
python api/ml/predict.py
```

### Production Testing
```bash
# Test deployed API
curl https://your-app.vercel.app/api/ml/predict?symbol=BTC
```

## 🚨 Troubleshooting

### "No trained model available"
- Run `python scripts/export_models_for_vercel.py` first
- Deploy the generated model configuration to Vercel

### "Unable to fetch real market data"
- Check internet connection
- Verify Yahoo Finance is accessible
- Symbol might not be supported

### "Import error" in Vercel logs
- Check `requirements.txt` is complete
- Verify `vercel.json` configuration
- Some dependencies might be too large for edge functions

## 📈 Production Readiness

- ✅ Real market data integration
- ✅ Actual ML predictions (no mocks)
- ✅ Proper error handling
- ✅ Lightweight for edge deployment
- ✅ Local development support
- ✅ Caching for performance

The system is now ready for production deployment on Vercel with genuine ML predictions!