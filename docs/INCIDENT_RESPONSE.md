# Incident Response

This document outlines the procedure for handling reports of unsafe, inaccurate, or hallucinated outputs from LEO Rigging AI.

## Triage

1. **Identify the query**: Obtain the exact user question and the response provided.
2. **Review the citations**:
   - Did the system cite a specific document?
   - Is the cited document correct and up-to-date for the user's jurisdiction?
3. **Check the context**: Use the `/api/search` endpoint to see what chunks were retrieved for the query.
4. **Identify the failure mode**:
   - **Retrieval failure**: The correct document was not retrieved or scored too low.
   - **Chunking failure**: The text was split in a way that removed critical context (e.g., a "Do not" was separated from the action).
   - **LLM Hallucination**: The model ignored the prompt and provided an ungrounded answer.
   - **Prompt Injection**: The model followed instructions embedded in the document.

## Remediation

### 1. Document Error (Bad Data)
If the ingested document is outdated or incorrect:
- Delete the document using `DELETE /api/pdfs/{doc_id}`.
- Re-ingest the corrected version.

### 2. Retrieval Failure
If the correct chunk was missed:
- Adjust `HYBRID_WEIGHT_DENSE` vs `HYBRID_WEIGHT_BM25`.
- Check if the term needs to be added to the domain-aware BM25 tokenizer in `vectorstore.py`.
- Consider enabling the reranker (`USE_RERANKER=true`).

### 3. LLM Hallucination
If the model generated an ungrounded claim:
- Verify that `MIN_RELEVANCE_SCORE` is not set too low (default 0.35).
- If the topic is high-risk, consider adding it to the `_HIGH_RISK_PATTERNS` regex list in `rag_service.py` to trigger the mandatory safety notice and clarifying questions.
