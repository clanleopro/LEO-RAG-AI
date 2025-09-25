import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.services.ingest_service import ingest_all_pdfs

if __name__ == "__main__":
    results = ingest_all_pdfs()
    # results is a List[Tuple[str,int]] -> make it JSON-friendly
    payload = [{"filename": name, "chunks_upserted": cnt} for (name, cnt) in results]
    print(json.dumps({"ingested": payload, "file_count": len(payload), "total_chunks": sum(c for _, c in results)}, indent=2))
