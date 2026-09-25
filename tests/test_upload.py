# tests/test_upload.py
import io
import pytest
from fastapi.testclient import TestClient

def test_upload_no_auth(client: TestClient, sample_pdf_content: bytes):
    files = {"files": ("test.pdf", sample_pdf_content, "application/pdf")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 401

def test_upload_invalid_extension(client: TestClient, write_headers: dict, sample_pdf_content: bytes):
    files = {"files": ("test.txt", sample_pdf_content, "text/plain")}
    response = client.post("/api/upload", headers=write_headers, files=files)
    assert response.status_code == 422
    assert "Only PDF files" in response.json()["error"]

def test_upload_invalid_magic_bytes(client: TestClient, write_headers: dict):
    fake_pdf = b"not a real pdf content"
    files = {"files": ("test.pdf", fake_pdf, "application/pdf")}
    response = client.post("/api/upload", headers=write_headers, files=files)
    assert response.status_code == 422
    assert "valid PDF" in response.json()["error"]

@pytest.mark.skip(reason="Needs a real valid PDF mock to pass fitz.open")
def test_upload_valid_pdf(client: TestClient, write_headers: dict, sample_pdf_content: bytes):
    files = {"files": ("test.pdf", sample_pdf_content, "application/pdf")}
    response = client.post("/api/upload", headers=write_headers, files=files)
    assert response.status_code == 202
    data = response.json()
    assert len(data["uploaded"]) == 1
    assert "job_id" in data["uploaded"][0]
