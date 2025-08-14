"""
FastAPI server for Strands Agent Management UI
Serves both the webapp and API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import sys

# Add src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = FastAPI(title="Strands Agent Management", version="1.0.0")

# Include agent API routes
try:
    from api.agents.routes import router as agents_router
    app.include_router(agents_router)
    print("✅ Agent API routes loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load agent routes: {e}")
    
    # Create a fallback route to show the error
    @app.get("/api/agents/list")
    async def fallback_agents():
        raise HTTPException(status_code=503, detail=f"Agent registry unavailable: {str(e)}")

# Serve static files from webapp directory
app.mount("/static", StaticFiles(directory="webapp"), name="static")

@app.get("/")
async def serve_index():
    """Serve the main launchpad page"""
    return FileResponse("webapp/index-launchpad.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Strands Agent Management Server"}

if __name__ == "__main__":
    print("🚀 Starting Strands Agent Management Server...")
    print("📱 UI will be available at: http://localhost:8000")
    print("🔌 API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )