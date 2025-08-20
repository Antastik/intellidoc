#!/usr/bin/env python3
"""
IntelliDoc API Launcher

Starts the FastAPI server for document processing.
Handles dependency issues gracefully and provides clear error messages.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import uvicorn
    from src.api.main import app
    
    print("🚀 Starting IntelliDoc API...")
    print("📖 API Documentation will be available at: http://localhost:8000/api/v1/docs")
    print("🏥 Health check available at: http://localhost:8000/health")
    print("📊 System status available at: http://localhost:8000/api/v1/status")
    print()
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Disable reload to avoid import issues
        access_log=True
    )
    
except ImportError as e:
    print(f"❌ Failed to start API due to missing dependencies: {e}")
    print()
    print("🔧 Possible solutions:")
    print("1. Install missing dependencies:")
    print("   pip install fastapi uvicorn")
    print()
    print("2. If you're seeing NumPy compatibility issues:")
    print("   pip install 'numpy<2.0'")
    print()
    print("3. Install all project dependencies:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Failed to start API: {e}")
    sys.exit(1)
