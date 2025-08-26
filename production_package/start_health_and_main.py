"""
Startup script that runs both health app and main API.
Health app handles deployment health checks on PORT.
Main API runs on MAIN_API_PORT for actual functionality.
"""

import os
import sys
import subprocess
import time
import signal
from multiprocessing import Process

def start_health_app():
    """Start the health check app on main deployment port"""
    port = int(os.getenv("PORT", "80"))
    subprocess.run([
        "uvicorn", "production_package.health_app:app",
        "--host", "0.0.0.0",
        "--port", str(port)
    ])

def start_main_api():
    """Start the main API on a different port"""
    main_port = int(os.getenv("MAIN_API_PORT", "8000"))
    # Set environment for main API
    env = os.environ.copy()
    env["PORT"] = str(main_port)
    
    # Give health app time to start first
    time.sleep(2)
    
    subprocess.run([
        "uvicorn", "production_package.app.main:app",
        "--host", "0.0.0.0",
        "--port", str(main_port)
    ], env=env)

def signal_handler(sig, frame):
    """Handle shutdown gracefully"""
    print("\nShutting down services...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting X Voice API with dual-app setup...")
    print(f"Health app will run on port {os.getenv('PORT', '80')}")
    print(f"Main API will run on port {os.getenv('MAIN_API_PORT', '8000')}")
    
    # Start both processes
    health_process = Process(target=start_health_app)
    main_process = Process(target=start_main_api)
    
    health_process.start()
    main_process.start()
    
    try:
        # Wait for both processes
        health_process.join()
        main_process.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        health_process.terminate()
        main_process.terminate()
        health_process.join()
        main_process.join()