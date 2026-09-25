# tests/test_vectorstore_math.py
"""
Pure-math tests for score normalization and cosine similarity conversion.
These tests do NOT import chromadb or touch any real vector store.
They test the mathematical correctness of _norm directly.
"""
import numpy as np
import pytest


# ─── Inline the function under test (pure math, no chromadb) ──────────────────
# This avoids importing vectorstore which would initialize a real ChromaDB client.

def _norm(vals):
    """Linear min-max normalization matching app/services/vectorstore.py."""
    if not vals:
        return []
    arr = np.array(vals, dtype=np.float32)
    # If all values are effectively zero
    if (np.abs(arr) < 1e-6).all():
        return [0.0] * len(arr)
    # If all values are exactly the same (std ≈ 0)
    if float(arr.std()) < 1e-6:
        return [1.0] * len(arr)
    # Linear min-max to [0, 1]
    z_min, z_max = float(arr.min()), float(arr.max())
    return [float((v - z_min) / (z_max - z_min + 1e-6)) for v in arr]


# ─── Normalization tests ───────────────────────────────────────────────────────

class TestNorm:
    def test_empty(self):
        assert _norm([]) == []

    def test_all_zeros(self):
        # Zero BM25 match case: should NOT get misleadingly high normalized scores
        result = _norm([0.0, 0.0, 0.0])
        assert result == [0.0, 0.0, 0.0], "All-zero scores must stay zero after normalization"

    def test_constant_nonzero(self):
        # All candidates with same score — all get 1.0 (equal treatment)
        result = _norm([5.0, 5.0, 5.0])
        assert result == [1.0, 1.0, 1.0]

    def test_linear_range(self):
        result = _norm([0.0, 5.0, 10.0])
        assert np.isclose(result[0], 0.0, atol=1e-4)
        assert np.isclose(result[1], 0.5, atol=1e-4)
        assert np.isclose(result[2], 1.0, atol=1e-4)

    def test_negative_range(self):
        result = _norm([-10.0, 0.0, 10.0])
        assert np.isclose(result[0], 0.0, atol=1e-4)
        assert np.isclose(result[1], 0.5, atol=1e-4)
        assert np.isclose(result[2], 1.0, atol=1e-4)

    def test_single_value(self):
        # Single nonzero value — treated as constant → 1.0
        result = _norm([7.3])
        assert result == [1.0]

    def test_output_clamped(self):
        # All output values must be in [0, 1]
        vals = [0.1, 0.5, 0.3, 0.9, 0.0, 1.0]
        result = _norm(vals)
        for v in result:
            assert 0.0 <= v <= 1.0, f"Normalized value {v} out of range"

    def test_order_preserved(self):
        vals = [0.1, 0.9, 0.5]
        result = _norm(vals)
        # Order of magnitudes should be preserved
        assert result[1] > result[2] > result[0]


# ─── Cosine similarity math tests ─────────────────────────────────────────────

class TestCosineSimilarityConversion:
    """
    Tests the math of `sim = 1.0 - distance` with Chroma cosine space.
    In cosine distance space: distance = 1 - cosine_similarity.
    Therefore: cosine_similarity = 1 - distance.
    """

    def _cosine_dist(self, a, b):
        """Compute cosine distance = 1 - cosine_similarity."""
        a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        return float(1.0 - sim)

    def test_identical_embeddings(self):
        """Identical embeddings → distance ≈ 0 → sim ≈ 1.0."""
        v = [0.6, 0.8, 0.0]
        dist = self._cosine_dist(v, v)
        sim = max(0.0, min(1.0, 1.0 - dist))
        assert np.isclose(sim, 1.0, atol=1e-5), f"Expected sim=1.0, got {sim}"

    def test_orthogonal_embeddings(self):
        """Orthogonal vectors → distance ≈ 1 → sim ≈ 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        dist = self._cosine_dist(a, b)
        sim = max(0.0, min(1.0, 1.0 - dist))
        assert np.isclose(sim, 0.0, atol=1e-5), f"Expected sim=0.0, got {sim}"

    def test_opposite_embeddings(self):
        """Opposite vectors → distance ≈ 2 → clipped to sim = 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        dist = self._cosine_dist(a, b)
        sim = max(0.0, min(1.0, 1.0 - dist))
        # Clipped to 0.0 (not negative)
        assert np.isclose(sim, 0.0, atol=1e-5), f"Expected sim=0.0, got {sim}"

    def test_sim_in_unit_range(self):
        """After clipping, sim must always be in [0, 1]."""
        for _ in range(20):
            a = np.random.randn(128)
            b = np.random.randn(128)
            dist = self._cosine_dist(a, b)
            sim = max(0.0, min(1.0, 1.0 - dist))
            assert 0.0 <= sim <= 1.0, f"sim={sim} out of [0,1]"

    def test_zero_bm25_score_does_not_inflate_blended(self):
        """
        A candidate with zero BM25 score and mediocre dense score
        should not get inflated blended relevance after normalization.
        """
        # dense_scores: [0.9, 0.5, 0.3]   bm25_scores: [0.0, 0.0, 0.0]
        dense = _norm([0.9, 0.5, 0.3])
        bm25  = _norm([0.0, 0.0, 0.0])     # all zero → all 0.0

        w_d, w_b = 0.55, 0.45
        blended = [w_d * d + w_b * b for d, b in zip(dense, bm25)]

        # The zero-BM25 scores must not push any candidate above its normalized dense score
        for d, bl in zip(dense, blended):
            assert bl <= d + 1e-6, (
                f"Blended score {bl} exceeds dense-only score {d} — "
                "BM25 zeros must not inflate relevance"
            )

    def test_dense_only_result_ranked_correctly(self):
        """Dense-only results: ordering must match the input similarity order."""
        dense_scores = [0.9, 0.3, 0.6]
        bm25_scores = [0.0, 0.0, 0.0]

        dn = _norm(dense_scores)
        bn = _norm(bm25_scores)

        w_d, w_b = 0.55, 0.45
        blended = [w_d * d + w_b * b for d, b in zip(dn, bn)]

        # Rank by blended
        ranked_idx = sorted(range(len(blended)), key=lambda i: blended[i], reverse=True)
        # Original rank by dense: 0 (0.9), 2 (0.6), 1 (0.3)
        expected_rank = [0, 2, 1]
        assert ranked_idx == expected_rank, (
            f"Ranking mismatch: got {ranked_idx}, expected {expected_rank}"
        )
