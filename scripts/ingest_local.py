import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.services.ingest_service import ingest_pdfs
if __name__ == "__main__":
    res = ingest_pdfs()
    print(res.model_dump())
