"""Retrieval layer: embed song descriptors and return top-N candidates by cosine similarity."""

import hashlib
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_PATH = "data/song_vectors.pkl"

_model = None


def _get_model():
    """Lazy-load the sentence-transformers model so importing rag.py is cheap."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def song_to_doc(song: Dict) -> str:
    """Convert one song dict to the text string we embed."""
    descriptor = song.get("descriptor", "").strip()
    return f"{song['title']} by {song['artist']}. {song['genre']} {song['mood']}. {descriptor}"


def _catalog_signature(songs: List[Dict]) -> str:
    """Hash the docs so we know when the cache is stale."""
    joined = "\n".join(song_to_doc(s) for s in songs)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_index(
    songs: List[Dict], cache_path: str = DEFAULT_CACHE_PATH
) -> Tuple[np.ndarray, List[Dict]]:
    """Return (embedding matrix, songs). Reads cache if signature matches; rebuilds otherwise."""
    signature = _catalog_signature(songs)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("signature") == signature and cached.get("model") == EMBED_MODEL_NAME:
                return cached["vectors"], songs
        except (pickle.PickleError, EOFError, KeyError, AttributeError):
            pass  # corrupt cache -> rebuild silently

    docs = [song_to_doc(s) for s in songs]
    vectors = _get_model().encode(docs, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {"signature": signature, "model": EMBED_MODEL_NAME, "vectors": vectors}, f
        )
    return vectors, songs


def retrieve(
    query: str, vectors: np.ndarray, songs: List[Dict], top_n: int = 15
) -> List[Tuple[Dict, float]]:
    """Embed `query` and return top_n (song, cosine_score) tuples sorted desc."""
    if not query or not query.strip():
        return []
    query_vec = _get_model().encode([query], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype=np.float32)[0]
    scores = vectors @ query_vec  # cosine since both are L2-normalized
    n = min(top_n, len(songs))
    top_idx = np.argsort(-scores)[:n]
    return [(songs[i], float(scores[i])) for i in top_idx]
