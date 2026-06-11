"""Retrieval sanity tests: queries should pull semantically-related songs to the top

These tests load the real catalog and the real sentence-transformers model, so
they're slower than the scoring unit tests but verify end-to-end retrieval
"""

import pytest

from src.rag import build_index, retrieve
from src.recommender import load_songs


@pytest.fixture(scope="module")
def index():
    songs = load_songs("data/songs.csv")
    vectors, songs = build_index(songs)
    return vectors, songs


def _top_genres(results, n=5):
    return [s["genre"] for s, _ in results[:n]]


def test_chill_study_query_pulls_lofi_or_ambient_first(index):
    vectors, songs = index
    results = retrieve("chill lofi for late-night studying", vectors, songs, top_n=15)
    assert len(results) == 15
    top5 = _top_genres(results, 5)
    #the top results should be dominated by chill/focus genres, not punk/edm
    assert any(g in {"lofi", "ambient"} for g in top5[:2])
    assert "punk" not in top5
    assert "edm" not in top5


def test_workout_query_pulls_intense_genres(index):
    vectors, songs = index
    results = retrieve("high energy pump-up workout music", vectors, songs, top_n=15)
    top_moods = [s["mood"] for s, _ in results[:5]]
    assert "intense" in top_moods
    #ambient/relaxed shouldn't dominate the head of the list
    assert top_moods[0] != "relaxed"


def test_empty_query_returns_no_candidates(index):
    vectors, songs = index
    assert retrieve("", vectors, songs) == []
    assert retrieve("   ", vectors, songs) == []


def test_cosine_scores_are_sorted_descending(index):
    vectors, songs = index
    results = retrieve("smooth jazz cafe morning", vectors, songs, top_n=10)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
