#!/usr/bin/env python3
"""
Production startup script for X Voice API.
Configures the FastAPI application for production deployment.
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add the production_package to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "production_package"))

def main():
    """Start the FastAPI application in production mode."""
    # Set production environment variables
    os.environ["APP_ENV"] = "production"
    os.environ.setdefault("PORT", "80")  # Default to port 80 for deployment
    
    # Get port from environment (deployment will set this)
    port = int(os.environ.get("PORT", "80"))
    
    print(f"🚀 Starting X Voice API in production mode")
    print(f"🌐 Server will listen on 0.0.0.0:{port}")
    print(f"📋 Environment: production")
    print(f"🔄 Reload: disabled")
    
    # Start the server pointing to the production app
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,              # Disable reload in production
        workers=1,                 # Single worker for deployment
        log_level="info",
        access_log=True,
        server_header=False,       # Don't expose server info
        date_header=False          # Don't expose date header
    )

if __name__ == "__main__":
    main()