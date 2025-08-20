"""
Basic API tests for IntelliDoc FastAPI service.

Tests the fundamental API endpoints and functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root API information endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "IntelliDoc API"
    assert data["version"] == "1.0.0"
    assert "endpoints" in data
    assert "features" in data


def test_health_endpoint():
    """Test the simple health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_system_status_endpoint():
    """Test the system status endpoint"""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "IntelliDoc API"
    assert data["version"] == "1.0.0"
    assert "uptime_seconds" in data
    assert "components" in data
    assert "queue_status" in data


def test_system_health_endpoint():
    """Test the detailed health check endpoint"""
    response = client.get("/api/v1/status/health")
    # This might return 200 or 503 depending on component availability
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_documents_formats_endpoint():
    """Test the supported formats endpoint"""
    response = client.get("/api/v1/documents/formats")
    assert response.status_code == 200
    data = response.json()
    assert "supported_formats" in data
    assert "processing_capabilities" in data
    assert "max_file_size_mb" in data


def test_jobs_stats_endpoint():
    """Test the job statistics endpoint"""
    response = client.get("/api/v1/jobs/stats/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_jobs" in data
    assert "by_status" in data
    assert "queue_health" in data


def test_jobs_list_endpoint():
    """Test the jobs list endpoint"""
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "total" in data
    assert "page" in data
    assert "per_page" in data


def test_openapi_docs():
    """Test that OpenAPI documentation is available"""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["title"] == "IntelliDoc API"
    assert data["info"]["version"] == "1.0.0"
    assert "paths" in data


if __name__ == "__main__":
    # Run basic tests
    print("Testing IntelliDoc API endpoints...")
    
    try:
        test_root_endpoint()
        print("✓ Root endpoint working")
        
        test_health_endpoint()
        print("✓ Health endpoint working")
        
        test_system_status_endpoint()
        print("✓ System status endpoint working")
        
        test_documents_formats_endpoint()
        print("✓ Documents formats endpoint working")
        
        test_jobs_stats_endpoint()
        print("✓ Jobs stats endpoint working")
        
        test_jobs_list_endpoint()
        print("✓ Jobs list endpoint working")
        
        test_openapi_docs()
        print("✓ OpenAPI documentation available")
        
        print("\n🎉 All basic API tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
