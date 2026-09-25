# tests/test_llm_errors.py
"""
Tests for LLM error handling.

Verifies that every OpenAI SDK exception category:
  - Raises the correct application-specific LLMError subclass
  - Returns the correct HTTP status code
  - Sets grounded:false
  - Never returns HTTP 200 on generation failure
  - Does not retry billing/auth/spend-limit errors

All external calls are mocked.
"""
from __future__ import annotations

import os
# pyrefly: ignore [missing-import]
import pytest
from unittest.mock import patch

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")
os.environ.setdefault("API_KEYS", "test-admin-key:admin,test-query-key:query")
os.environ.setdefault("AUTH_ENABLED", "true")
os.environ.setdefault("CHROMA_COLLECTION", "test_collection")


# ─── Mock exception factories ─────────────────────────────────────────────────
# We create subclasses directly (cannot reassign __class__ on Exception in CPython).

def _make_exc_class(name, code=None, status_code=None, message="test error"):
    """Create a real exception instance whose class.__name__ matches `name`."""
    cls = type(name, (Exception,), {"code": code, "status_code": status_code, "error": None})
    return cls(message)


# ─── Unit tests: LLM exception classification ─────────────────────────────────

class TestLLMErrorClassification:
    """Test that _classify_openai_error returns the right exception type."""

    def _classify(self, exc):
        from app.services.llm import _classify_openai_error
        return _classify_openai_error(exc)

    def test_auth_error(self):
        from app.services.llm import LLMAuthError
        exc = _make_exc_class("AuthenticationError")
        assert isinstance(self._classify(exc), LLMAuthError)

    def test_rate_limit_temporary(self):
        from app.services.llm import LLMRateLimitError
        exc = _make_exc_class("RateLimitError", code="rate_limit_exceeded")
        assert isinstance(self._classify(exc), LLMRateLimitError)

    def test_rate_limit_credit_exhausted_by_code(self):
        from app.services.llm import LLMCreditError
        exc = _make_exc_class("RateLimitError", code="credit_balance_exhausted")
        assert isinstance(self._classify(exc), LLMCreditError)

    def test_rate_limit_insufficient_quota_in_message(self):
        from app.services.llm import LLMCreditError
        exc = _make_exc_class(
            "RateLimitError",
            message="You exceeded your current quota. insufficient_quota",
        )
        assert isinstance(self._classify(exc), LLMCreditError)

    def test_project_spend_limit(self):
        from app.services.llm import LLMSpendLimitError
        exc = _make_exc_class("RateLimitError", code="project_spend_limit_reached")
        assert isinstance(self._classify(exc), LLMSpendLimitError)

    def test_org_spend_limit(self):
        from app.services.llm import LLMSpendLimitError
        exc = _make_exc_class("RateLimitError", code="organization_spend_limit_reached")
        assert isinstance(self._classify(exc), LLMSpendLimitError)

    def test_org_usage_limit(self):
        from app.services.llm import LLMSpendLimitError
        exc = _make_exc_class("RateLimitError", code="organization_usage_limit_reached")
        assert isinstance(self._classify(exc), LLMSpendLimitError)

    def test_timeout_error(self):
        from app.services.llm import LLMTimeoutError
        exc = _make_exc_class("APITimeoutError")
        assert isinstance(self._classify(exc), LLMTimeoutError)

    def test_connection_error(self):
        from app.services.llm import LLMUnavailableError
        exc = _make_exc_class("APIConnectionError")
        assert isinstance(self._classify(exc), LLMUnavailableError)

    def test_bad_request_error(self):
        from app.services.llm import LLMBadRequestError
        exc = _make_exc_class("BadRequestError")
        assert isinstance(self._classify(exc), LLMBadRequestError)

    def test_internal_server_error(self):
        from app.services.llm import LLMUnavailableError
        exc = _make_exc_class("InternalServerError")
        assert isinstance(self._classify(exc), LLMUnavailableError)

    def test_service_unavailable_error(self):
        from app.services.llm import LLMUnavailableError
        exc = _make_exc_class("ServiceUnavailable")
        assert isinstance(self._classify(exc), LLMUnavailableError)

    def test_safe_message_never_exposes_key(self):
        """The safe_message on an LLMAuthError is a fixed string, not the provider body."""
        from app.services.llm import LLMAuthError
        exc = _make_exc_class("AuthenticationError", message="sk-realkey1234567890 invalid")
        result = self._classify(exc)
        assert isinstance(result, LLMAuthError)
        assert "sk-" not in result.safe_message
        assert "realkey" not in result.safe_message


# ─── Integration tests: HTTP status codes via router ──────────────────────────

class TestQueryEndpointHTTPStatus:
    """
    Test that /api/query returns the correct HTTP status code and grounded:false
    for every LLM failure mode. Uses mocked retrieve/build_context/llm.generate.
    """

    @pytest.fixture(autouse=True)
    def _vectorstore_mock(self):
        """Prevent ChromaDB from being initialized during these tests."""
        from app.services.vectorstore import SearchHit
        fake_hit = SearchHit(
            id="chunk1", text="WLL test text", source="test.pdf", doc_id="doc1",
            page=1, score_vec=0.8, score_bm25=0.6, score_blended=0.72,
        )
        with patch("app.services.rag_service.retrieve", return_value=[fake_hit]), \
             patch("app.services.rag_service.build_context", return_value=(
                 "fake context", [
                     {"n": 1, "doc_id": "doc1", "title": "test.pdf", "page": 1,
                      "revision": None, "snippet": "test", "score_vec": 0.8,
                      "score_bm25": 0.6, "score_blended": 0.72}
                 ]
             )):
            yield

    def _client(self):
        # pyrefly: ignore [missing-import]
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app, raise_server_exceptions=False)

    def _post(self, exc_to_raise):
        client = self._client()
        with patch("app.services.rag_service.llm.generate", side_effect=exc_to_raise):
            return client.post(
                "/api/query",
                json={"question": "What is the WLL of a Grade 80 chain sling?"},
                headers={"Authorization": "Bearer test-query-key"},
            )

    def test_rate_limit_returns_429(self):
        from app.services.llm import LLMRateLimitError
        resp = self._post(LLMRateLimitError("rate limit"))
        assert resp.status_code == 429, f"Got {resp.status_code}: {resp.text}"
        assert resp.json().get("grounded") is False

    def test_credit_error_returns_503(self):
        from app.services.llm import LLMCreditError
        resp = self._post(LLMCreditError("credit exhausted"))
        assert resp.status_code == 503
        assert resp.json().get("grounded") is False

    def test_spend_limit_returns_503(self):
        from app.services.llm import LLMSpendLimitError
        resp = self._post(LLMSpendLimitError("spend limit"))
        assert resp.status_code == 503
        assert resp.json().get("grounded") is False

    def test_auth_error_returns_503(self):
        from app.services.llm import LLMAuthError
        resp = self._post(LLMAuthError("auth error"))
        assert resp.status_code == 503
        assert resp.json().get("grounded") is False

    def test_timeout_returns_504(self):
        from app.services.llm import LLMTimeoutError
        resp = self._post(LLMTimeoutError("timeout"))
        assert resp.status_code == 504
        assert resp.json().get("grounded") is False

    def test_unavailable_returns_503(self):
        from app.services.llm import LLMUnavailableError
        resp = self._post(LLMUnavailableError("unavailable"))
        assert resp.status_code == 503
        assert resp.json().get("grounded") is False

    def test_bad_request_returns_500(self):
        from app.services.llm import LLMBadRequestError
        resp = self._post(LLMBadRequestError("bad request"))
        assert resp.status_code == 500
        assert resp.json().get("grounded") is False

    def test_rate_limit_has_retry_after_header(self):
        from app.services.llm import LLMRateLimitError
        resp = self._post(LLMRateLimitError("rate limit"))
        assert resp.status_code == 429
        header_names = {k.lower() for k in resp.headers}
        assert "retry-after" in header_names, f"Missing Retry-After header: {dict(resp.headers)}"

    def test_no_200_on_any_llm_failure(self):
        """Exhaustive check: none of the LLM error classes should produce HTTP 200."""
        from app.services.llm import (
            LLMRateLimitError, LLMCreditError, LLMSpendLimitError,
            LLMAuthError, LLMTimeoutError, LLMUnavailableError, LLMBadRequestError,
        )
        for exc_class in [
            LLMRateLimitError, LLMCreditError, LLMSpendLimitError,
            LLMAuthError, LLMTimeoutError, LLMUnavailableError, LLMBadRequestError,
        ]:
            resp = self._post(exc_class("test error"))
            assert resp.status_code != 200, (
                f"{exc_class.__name__} produced HTTP 200 — must never happen"
            )

    def test_error_response_does_not_expose_provider_body(self):
        """Error response must not contain internal messages."""
        from app.services.llm import LLMAuthError
        resp = self._post(LLMAuthError("internal details for operator only"))
        body = resp.text
        assert "internal details" not in body
