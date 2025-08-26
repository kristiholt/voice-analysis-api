#!/usr/bin/env python3
"""
Production startup script for X Voice API.
This script is optimized for production deployment.
"""
import os
import uvicorn

def main():
    """Start the FastAPI application in production mode."""
    # Force production settings
    os.environ["APP_ENV"] = "production"
    
    # Get port from environment, default to 80 for external deployment
    port = int(os.environ.get("PORT", "80"))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 Starting X Voice API in production mode")
    print(f"🌐 Server will listen on {host}:{port}")
    print(f"📋 Environment: production")
    print(f"🔄 Reload: disabled")
    
    # Start the server with production settings optimized for cloud deployment
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,              # Disable reload in production
        workers=1,                 # Single worker for container deployment  
        log_level="info",
        access_log=True,
        server_header=False,       # Don't expose server info
        date_header=False,         # Don't expose date header
        timeout_keep_alive=30,     # Optimized for Cloud Run
        timeout_graceful_shutdown=10,  # Faster shutdown for deployments
        loop="auto",               # Use the best available event loop
        lifespan="on",             # Enable lifespan events for proper startup/shutdown
        proxy_headers=True,        # Trust proxy headers for Cloud Run
        forwarded_allow_ips="*"    # Allow all forwarded IPs for Cloud Run
    )

if __name__ == "__main__":
    main()