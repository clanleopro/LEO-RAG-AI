# tests/test_ingest.py
import pytest
from unittest.mock import patch
from app.services import ingest_service
from pathlib import Path

def test_ingest_missing_file():
    with pytest.raises(ingest_service.IngestionError, match="File not found"):
        ingest_service.ingest_pdf("nonexistent.pdf")
