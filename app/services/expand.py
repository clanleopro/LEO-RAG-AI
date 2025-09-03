# app/services/expand.py
from __future__ import annotations
from typing import List, Dict
import json
from pathlib import Path

# Path to your JSON file
ALIASES_PATH = Path(__file__).resolve().parents[1] / "data" / "rigging_aliases.json"

# Load aliases from JSON
try:
    with open(ALIASES_PATH, "r", encoding="utf-8") as f:
        ALIASES: Dict[str, List[str]] = json.load(f)
except Exception as e:
    print(f"[expand] Failed to load aliases from {ALIASES_PATH}: {e}")
    ALIASES = {}

def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set(); out = []
    for s in items:
        k = s.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out

def expanded_queries(q: str, max_variants: int = 6) -> List[str]:
    """
    Given a user query, return variants including acronym expansions
    to improve retrieval recall.
    """
    out: List[str] = [q]
    ql = q.lower()

    # If the query already contains a known acronym or alias, add its variants
    for key, variants in ALIASES.items():
        if key.lower() in ql or any(v.lower() in ql for v in variants):
            out.extend([key, *variants])

    out = _uniq_keep_order(out)
    return out[:max_variants]

# Backward-compat alias
def expand_queries(q: str) -> List[str]:
    return expanded_queries(q)
