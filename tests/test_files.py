# tests/test_files.py
import pytest
from fastapi.testclient import TestClient

def test_list_pdfs(client: TestClient, read_headers: dict):
    response = client.get("/api/pdfs", headers=read_headers)
    assert response.status_code == 200
    assert "documents" in response.json()

def test_path_traversal_download(client: TestClient, read_headers: dict):
    response = client.get("/api/pdfs/..%2f..%2f.env", headers=read_headers)
    assert response.status_code == 400
    assert "Invalid document identifier" in response.json()["error"]

def test_path_traversal_delete(client: TestClient, admin_headers: dict):
    response = client.delete("/api/pdfs/..%2f..%2f.env", headers=admin_headers)
    assert response.status_code == 400
    assert "Invalid document identifier" in response.json()["error"]

def test_download_not_found(client: TestClient, read_headers: dict):
    response = client.get("/api/pdfs/nonexistent.pdf", headers=read_headers)
    assert response.status_code == 404
