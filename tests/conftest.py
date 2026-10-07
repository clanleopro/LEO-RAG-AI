# tests/conftest.py
"""
Pytest configuration and fixtures.
Mocks out real LLM and embedding calls to ensure tests are fast, deterministic,
and do not require external credentials.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Must set before importing app modules
os.environ["ENVIRONMENT"] = "test"
os.environ["AUTH_ENABLED"] = "true"
os.environ["API_KEYS"] = "test-admin-key:admin,test-query-key:query,test-write-key:documents:write,test-read-key:documents:read"
os.environ["OPENAI_API_KEY"] = "sk-test-mock-key"
os.environ["VECTOR_DB"] = "chroma"
os.environ["CHROMA_COLLECTION"] = "test_collection"
os.environ["CLAMAV_SOCKET"] = ""  # Disable malware scanning for tests


@pytest.fixture(autouse=True)
def mock_directories() -> Generator[None, None, None]:
    """Isolate data directories to a temporary location for each test run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_pdfs = tmp_path / "source_pdfs"
        vector_dir = tmp_path / "vectorstore"
        source_pdfs.mkdir(parents=True)
        vector_dir.mkdir(parents=True)

        with patch("app.services.config.SOURCE_PDFS", source_pdfs), \
             patch("app.services.config.VECTOR_DIR", vector_dir):
            yield


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock the sentence-transformers embedding model."""
    import numpy as np
    
    mock_model = MagicMock()
    # Return deterministic random vectors based on input length to avoid identical embeddings
    def mock_encode(texts, **kwargs):
        res = []
        for t in texts:
            np.random.seed(len(t))
            vec = np.random.rand(384).astype(np.float32)
            # Normalize
            vec = vec / np.linalg.norm(vec)
            res.append(vec)
        return res
        
    mock_model.encode.side_effect = mock_encode
    mock_model.get_sentence_embedding_dimension.return_value = 384
    
    with patch("app.services.embeddings._load_model", return_value=mock_model):
        yield mock_model


@pytest.fixture(autouse=True)
def mock_llm():
    """Mock OpenAI API calls."""
    with patch("app.services.llm.generate", return_value="Mocked LLM answer.") as mock_gen, \
         patch("app.services.llm.stream_chat") as mock_stream:
        mock_stream.return_value = (c for c in ["Mocked", " LLM", " answer."])
        yield mock_gen


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def admin_headers() -> dict:
    return {"Authorization": "Bearer test-admin-key"}

@pytest.fixture
def query_headers() -> dict:
    return {"Authorization": "Bearer test-query-key"}

@pytest.fixture
def write_headers() -> dict:
    return {"Authorization": "Bearer test-write-key"}

@pytest.fixture
def read_headers() -> dict:
    return {"Authorization": "Bearer test-read-key"}

@pytest.fixture
def sample_pdf_content() -> bytes:
    """A minimal valid PDF file (1 page, empty)."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources <<>> >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n214\n%%EOF\n"
    )
