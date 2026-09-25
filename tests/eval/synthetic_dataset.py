# tests/eval/synthetic_dataset.py
"""
Small synthetic dataset for RAG evaluation.
Contains safe, non-authoritative sample content.
"""
DATASET = [
    {
        "query": "What is the maximum load for a 1-inch wire rope?",
        "context": "The 1-inch wire rope has a safe working load of 5 tons under ideal conditions.",
        "expected_answer_contains": ["5 tons", "ideal conditions"],
        "high_risk": True,
    },
    {
        "query": "Who is responsible for inspecting slings?",
        "context": "A competent person must inspect slings each day before use.",
        "expected_answer_contains": ["competent person", "each day"],
        "high_risk": True,
    },
    {
        "query": "What is a tagline used for?",
        "context": "A tagline is a rope attached to a suspended load to control its rotation and prevent swinging.",
        "expected_answer_contains": ["control", "rotation", "swinging"],
        "high_risk": False,
    }
]
