import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import json

# Add parent directory to path to import from the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your FastAPI app
from main import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def authenticated_client():
    """Create a test client with authentication mocked."""
    # Mock the authentication dependency
    def mock_get_current_user():
        return {
            "id": 1,
            "username": "testuser",
            "email": "test@example.com"
        }
    
    # Override the dependency
    from api.auth import get_current_user
    app.dependency_overrides[get_current_user] = mock_get_current_user
    
    client = TestClient(app)
    
    yield client
    
    # Clean up the override after test
    app.dependency_overrides.clear()

@pytest.fixture(autouse=True)
def mock_redis():
    """Mock Redis connections for all tests."""
    with patch('utils.cache.get_redis_client') as mock_get_redis:
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock_client.set.return_value = True
        mock_client.exists.return_value = False
        mock_get_redis.return_value = mock_client
        yield mock_client

@pytest.fixture(autouse=True)
def mock_file_operations():
    """Mock file operations for all tests."""
    with patch('os.path.exists') as mock_exists, \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open') as mock_open:
        mock_exists.return_value = True
        mock_makedirs.return_value = None
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        yield {
            'exists': mock_exists,
            'makedirs': mock_makedirs,
            'open': mock_open,
            'file': mock_file
        }

@pytest.fixture(autouse=True)
def mock_database():
    """Mock database operations for all tests."""
    with patch('sqlite3.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield {
            'connection': mock_conn,
            'cursor': mock_cursor
        }