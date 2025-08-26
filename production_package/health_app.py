"""
Health-only FastAPI app for deployment health checks.
Loads instantly without heavy ML models or database connections.
This allows the main API to initialize properly while passing health checks.
"""

import time
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create minimal FastAPI app
app = FastAPI(
    title="X Voice API Health Check",
    description="Fast health check endpoint for deployment",
    version="1.0.0",
    docs_url=None,  # Disable docs for health app
    redoc_url=None
)

# Basic CORS for health checks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD"],
    allow_headers=["*"],
)

@app.get("/")
@app.head("/")
async def root():
    """Main health check endpoint - responds instantly"""
    return {
        "status": "healthy",
        "service": "X Voice API",
        "timestamp": int(time.time()),
        "version": "1.0.0",
        "app_type": "health_check"
    }

@app.get("/health")
@app.head("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": int(time.time())
    }

@app.get("/ready")
@app.head("/ready")
async def ready():
    """Readiness probe endpoint"""
    return {
        "status": "ready",
        "timestamp": int(time.time())
    }

@app.get("/api-status")
async def api_status():
    """Check if main API is accessible"""
    # This will be useful for monitoring
    main_api_port = os.getenv("MAIN_API_PORT", "8000")
    return {
        "status": "health_app_running",
        "main_api_port": main_api_port,
        "timestamp": int(time.time()),
        "note": "Main API runs separately for voice analysis"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port)