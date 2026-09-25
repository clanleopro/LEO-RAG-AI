# Ingestion Pipeline

The ingestion process in LEO Rigging AI ensures high-quality retrieval.

## Pipeline Steps

1.  **Validation**: PDFs are checked for magic bytes, encryption, and page count limits.
2.  **Extraction**: Text is extracted via PyMuPDF. If the extracted text is too short (scanned PDF), the system falls back to Tesseract OCR.
3.  **Chunking**: The `smart_chunk` algorithm splits text on paragraph and sentence boundaries, avoiding mid-sentence cuts. It uses `tiktoken` to respect token limits (`CHUNK_TOKENS`).
4.  **Embedding**: Chunks are embedded using `sentence-transformers` (e.g., `multilingual-e5-small`). Appropriate query/passage prefixes are applied.
5.  **Storage**: Chunks and embeddings are upserted to ChromaDB. Metadata (checksum, page, document ID, model) is stored alongside.
6.  **Lexical Index**: The in-memory BM25 index is rebuilt to include the new chunks.

## Re-ingestion

If a document with the same name is ingested again, the system automatically deletes the old vector chunks before processing the new file, preventing stale data duplication.
