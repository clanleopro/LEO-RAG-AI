# tests/test_query.py
import pytest
from fastapi.testclient import TestClient

def test_query_validation(client: TestClient, query_headers: dict):
    # Empty query
    response = client.post("/api/query", headers=query_headers, json={"question": ""})
    assert response.status_code == 422
    
    # top_k too large
    response = client.post("/api/query", headers=query_headers, json={"question": "test", "top_k": 999})
    assert response.status_code == 422
    
    # Path traversal in filter_doc
    response = client.post("/api/query", headers=query_headers, json={"question": "test", "filter_doc": "../etc/passwd"})
    assert response.status_code == 422

def test_query_empty_store(client: TestClient, query_headers: dict):
    response = client.post("/api/query", headers=query_headers, json={"question": "What is the capacity?"})
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["low_confidence"] is True
    assert "do not provide enough reliable information" in data["answer"]
    assert len(data["citations"]) == 0
