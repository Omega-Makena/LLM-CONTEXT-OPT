"""Embedding backend with graceful degradation.

Chosen backend: sentence-transformers (all-MiniLM-L6-v2), local, GPU-capable,
no API key. If it is not installed yet, we fall back to a deterministic
hashing embedder so the whole pipeline still runs end-to-end. The fallback is
*not* semantic — it exists purely so `demo.py` works before you pip install.

Swap the backend by constructing `Embedder(model_name=...)`; the rest of the
engine only depends on `Embedder.encode`.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder:
    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384,
        force_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self.backend = "hash-fallback"
        if force_fallback:
            return  # eval ablation: deliberately use the hash embedder
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
            self.backend = "sentence-transformers"
        except Exception:
            # Left on the hash fallback; demo still runs.
            self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) L2-normalized float32 matrix."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is not None:
            vecs = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(vecs, dtype=np.float32)
        return self._hash_encode(texts)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # --- deterministic fallback -------------------------------------------
    def _hash_encode(self, texts: list[str]) -> np.ndarray:
        """Bag-of-hashed-tokens vectors. Deterministic, cheap, non-semantic.

        Good enough that exact/near term overlap produces high cosine, which
        keeps the retrieval demo legible until real embeddings are installed.
        """
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _l2_normalize(out)
