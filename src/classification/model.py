"""Hazard classifier: multi-class type (+not_hazard => relevance), severity, calibrated confidence.

kinds: baseline = word TF-IDF + Logistic Regression (uncalibrated)
       selected = char+word TF-IDF + sentence embeddings + Logistic Regression, sigmoid-calibrated
"""
import time

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.common import ALL_TYPES, HAZARD_TYPES, MODELS_DIR
from src.semantic.embeddings import get_embedder

SEV_ORDER = ["low", "moderate", "high"]


class Features:
    def __init__(self, kind, embedder=None):
        self.kind, self.embedder = kind, embedder
        if kind == "baseline":
            self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        else:
            self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
            self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3)

    def _emb(self, texts):
        return sp.csr_matrix(self.embedder.encode(texts)) * 1.5

    def fit_transform(self, texts):
        if self.kind == "baseline":
            return self.word.fit_transform(texts)
        return sp.hstack([self.word.fit_transform(texts), self.char.fit_transform(texts), self._emb(texts)]).tocsr()

    def transform(self, texts):
        if self.kind == "baseline":
            return self.word.transform(texts)
        return sp.hstack([self.word.transform(texts), self.char.transform(texts), self._emb(texts)]).tocsr()


class HazardClassifier:
    def __init__(self, kind="selected", relevance_threshold=0.5):
        assert kind in ("baseline", "selected")
        self.kind, self.thr = kind, relevance_threshold
        self.feats = Features(kind, get_embedder() if kind == "selected" else None)
        self.embedder_name = self.feats.embedder.name if self.feats.embedder else None  # model must be reloaded with the SAME embedder
        self.model_used = kind

    def fit(self, texts, types, severities):
        X = self.feats.fit_transform(list(texts))
        base = LogisticRegression(max_iter=2000, C=10, class_weight="balanced")
        self.type_clf = (CalibratedClassifierCV(base, method="sigmoid", cv=3) if self.kind == "selected" else base).fit(X, types)
        types = np.asarray(types)
        sev = np.asarray(severities)
        m = (types != "not_hazard") & np.isin(sev, SEV_ORDER)
        self.sev_clf = LogisticRegression(max_iter=2000, C=5, class_weight="balanced").fit(X[m], sev[m])
        return self

    def predict_proba(self, texts):
        X = self.feats.transform(list(texts))
        P = self.type_clf.predict_proba(X)
        return X, P, list(self.type_clf.classes_)

    def predict(self, texts):
        X, P, cls = self.predict_proba(texts)
        S, scls = self.sev_clf.predict_proba(X), list(self.sev_clf.classes_)
        out = []
        for p, s in zip(P, S):
            probs = {c: float(v) for c, v in zip(cls, p)}
            p_h = 1.0 - probs["not_hazard"]
            top = max(HAZARD_TYPES, key=lambda k: probs.get(k, 0))
            rel = p_h >= self.thr
            out.append(dict(type=top if rel else "not_hazard", type_probs=probs, p_hazard=p_h, relevant=bool(rel),
                            severity=scls[int(np.argmax(s))] if rel else "none",
                            confidence=probs[top] if rel else 1 - p_h, model_used=self.model_used))
        return out

    def save(self, path=None):
        path = path or MODELS_DIR / f"{self.kind}.joblib"
        emb, self.feats.embedder = self.feats.embedder, None  # embedder is re-created on load
        joblib.dump(self, path)
        self.feats.embedder = emb
        return path

    @staticmethod
    def load(kind="selected", path=None):
        m = joblib.load(path or MODELS_DIR / f"{kind}.joblib")
        if m.embedder_name:
            m.feats.embedder = get_embedder("hash" if m.embedder_name.startswith("hash") else "sbert")  # raises if sbert is unavailable -> pipeline falls back
        return m


def timed(fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t
