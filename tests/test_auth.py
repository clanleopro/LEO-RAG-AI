# tests/test_auth.py
import pytest
from fastapi.testclient import TestClient

def test_no_auth_header(client: TestClient):
    response = client.post("/api/query", json={"question": "test"})
    assert response.status_code == 401

def test_invalid_auth_header(client: TestClient):
    response = client.post("/api/query", json={"question": "test"}, headers={"Authorization": "Bearer bad-key"})
    assert response.status_code == 401

def test_valid_query_auth(client: TestClient, query_headers: dict):
    response = client.post("/api/query", json={"question": "test"}, headers=query_headers)
    assert response.status_code == 200

def test_insufficient_scope(client: TestClient, query_headers: dict):
    # query role shouldn't be able to list files (needs documents:read)
    response = client.get("/api/pdfs", headers=query_headers)
    assert response.status_code == 403

def test_admin_has_all_scopes(client: TestClient, admin_headers: dict):
    response = client.get("/api/pdfs", headers=admin_headers)
    assert response.status_code == 200
    
    response = client.post("/api/query", json={"question": "test"}, headers=admin_headers)
    assert response.status_code == 200

def test_api_key_header(client: TestClient):
    response = client.post("/api/query", json={"question": "test"}, headers={"X-API-Key": "test-query-key"})
    assert response.status_code == 200
