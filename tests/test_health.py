# tests/test_health.py
import pytest
from fastapi.testclient import TestClient

def test_liveness(client: TestClient):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_readiness(client: TestClient):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
