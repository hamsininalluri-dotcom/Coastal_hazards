"""Text embeddings. Prefers a multilingual sentence-transformer; falls back to a stateless
hashed char-n-gram embedder when the model/library is unavailable (offline, no torch)."""
import os

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

SBERT_NAME = os.getenv("HAZARD_SBERT", "paraphrase-multilingual-MiniLM-L12-v2")


class HashEmbedder:
    name = "hash-char-ngram"

    def __init__(self, dim=512):
        self.v = HashingVectorizer(n_features=dim, analyzer="char_wb", ngram_range=(2, 4), alternate_sign=False, norm="l2")

    def encode(self, texts):
        return self.v.transform([t or "" for t in texts]).toarray().astype(np.float32)


class SbertEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer  # heavy import kept lazy
        self.m = SentenceTransformer(SBERT_NAME)
        self.name = f"sbert:{SBERT_NAME}"

    def encode(self, texts):
        return np.asarray(self.m.encode([t or "" for t in texts], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


_CACHE = {}


def get_embedder(mode=None):
    """mode: auto | sbert | hash (env HAZARD_EMBEDDER)."""
    mode = mode or os.getenv("HAZARD_EMBEDDER", "auto")
    if mode in _CACHE:
        return _CACHE[mode]
    emb = None
    if mode in ("auto", "sbert"):
        try:
            emb = SbertEmbedder()
        except Exception as e:  # ImportError, no network, etc.
            if mode == "sbert":
                raise
            print(f"[embeddings] sentence-transformers unavailable ({type(e).__name__}); using hash embedder")
    emb = emb or HashEmbedder()
    _CACHE[mode] = emb
    return emb
