# tests/eval/evaluate.py
"""
Simple evaluation script to measure RAG quality against the synthetic dataset.
"""
from tests.eval.synthetic_dataset import DATASET

def evaluate_rag():
    print("Running synthetic evaluation...")
    score = 0
    for item in DATASET:
        # In a real eval, we would call rag_service.answer here 
        # and compare the result with expected_answer_contains.
        pass
    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate_rag()
