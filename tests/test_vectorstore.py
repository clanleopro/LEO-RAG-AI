# tests/test_vectorstore.py
from app.services import vectorstore

def test_bm25_tokenizer():
    text = "The SWL of the ASME B30.9 wire-rope sling is 10kg."
    tokens = vectorstore._tokenize(text)
    # Stop words removed, standards codes kept, units kept
    assert "asme" in tokens
    assert "b30.9" in tokens
    assert "swL" not in tokens # wait, lowercased -> "swl"
    assert "swl" in tokens
    assert "wire-rope" in tokens
    assert "10kg" in tokens
    assert "the" not in tokens

def test_deterministic_id():
    id1 = vectorstore._deterministic_id("doc.pdf", 1, "text")
    id2 = vectorstore._deterministic_id("doc.pdf", 1, "text")
    id3 = vectorstore._deterministic_id("doc.pdf", 2, "text")
    
    assert id1 == id2
    assert id1 != id3
