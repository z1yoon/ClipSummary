import os
import pytest
from fastapi.testclient import TestClient
import sys

# Add parent directory to path to import from the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your FastAPI app
from main import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)