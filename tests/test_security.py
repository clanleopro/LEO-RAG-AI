# tests/test_security.py
from app.services.llm import _sanitize_error

def test_sanitize_error():
    err = "API error: sk-12345abcdef67890ghijklmnopq is invalid"
    safe = _sanitize_error(err)
    assert "sk-12345abcdef67890ghijklmnopq" not in safe
    assert "<REDACTED>" in safe
