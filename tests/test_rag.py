# tests/test_rag.py
import pytest
from app.services import rag_service
from app.services.vectorstore import SearchHit

def test_high_risk_detection():
    assert rag_service._is_high_risk("approve this lift plan") is True
    assert rag_service._is_high_risk("what is the capacity of CM-750?") is True
    assert rag_service._is_high_risk("is it safe to lift 10 tons?") is True
    assert rag_service._is_high_risk("what does WLL mean?") is True
    assert rag_service._is_high_risk("how do I inspect a damaged sling?") is True
    
    # Non-high risk
    assert rag_service._is_high_risk("what is a shackle?") is False
    assert rag_service._is_high_risk("explain the history of cranes") is False

def test_context_budgeting():
    hits = [
        SearchHit(id="1", text="A" * 100, source="doc1", doc_id="d1", page=1, score_vec=0.9, score_bm25=0.9, score_blended=0.9),
        SearchHit(id="2", text="B" * 500, source="doc1", doc_id="d1", page=2, score_vec=0.8, score_bm25=0.8, score_blended=0.8),
        SearchHit(id="3", text="C" * 100, source="doc1", doc_id="d1", page=3, score_vec=0.7, score_bm25=0.7, score_blended=0.7),
    ]
    
    # Budget of 400 chars. Should include 1 (length ~ 130 with prefix) and 3. 2 should be skipped entirely.
    context, citations = rag_service.build_context(hits, max_context_chars=400)
    
    assert "A" * 100 in context
    assert "C" * 100 in context
    assert "B" * 500 not in context
    
    assert len(citations) == 2
    assert citations[0]["n"] == 1
    assert citations[1]["n"] == 3


def test_document_answer_and_no_answer_fallback(monkeypatch):
    hits = [SearchHit(id="1", text="A shackle connects lifting components.",
                      source="manual.pdf", doc_id="manual", page=4,
                      score_vec=.8, score_bm25=.7, score_blended=.8)]
    monkeypatch.setattr(rag_service, "retrieve", lambda *args, **kwargs: hits)
    monkeypatch.setattr(rag_service.llm, "generate", lambda **kwargs: "A shackle connects lifting components [1].")
    monkeypatch.setattr(rag_service.llm, "generate_general", lambda **kwargs: "General answer.")
    result = rag_service.answer("What is a shackle?")
    assert result["grounded"] is True
    assert result["meta"]["answer_source"] == "documents"
    assert result["citations"][0]["page"] == 4

    monkeypatch.setattr(rag_service.llm, "generate", lambda **kwargs: "PDF_NO_ANSWER")
    result = rag_service.answer("What is a shackle?")
    assert result["grounded"] is False
    assert result["citations"] == []
    assert result["meta"]["answer_source"] == "general_knowledge"
    assert "General answer." in result["answer"]


def test_out_of_scope_returns_no_answer_or_citations(monkeypatch):
    monkeypatch.setattr(rag_service.llm, "is_industry_related", lambda query: False)
    monkeypatch.setattr(rag_service, "retrieve", lambda *args, **kwargs: pytest.fail("retrieval should not run"))
    result = rag_service.answer("What is the capital of Japan?")
    assert result["meta"]["answer_source"] == "out_of_scope"
    assert result["citations"] == []
    assert result["grounded"] is False
    assert "rigging" in result["answer"]
