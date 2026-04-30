"""CLI entrypoint.

Default mode (RAG): pass a free-text query.
    python -m src.main "chill lofi for studying"

Demo mode: replay the four Phase-4 evaluation profiles (no query, no retrieval).
    python -m src.main --demo
"""

import argparse
import sys
from typing import Dict, List, Tuple

from src.recommender import load_songs, recommend_songs, score_song

TOP_N_RETRIEVE = 15
TOP_K_RANK = 5


PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "likes_acoustic": False,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "likes_acoustic": True,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.9,
        "likes_acoustic": False,
    },
    "Adversarial: Conflicted Listener": {
        "genre": "pop",
        "mood": "sad",
        "energy": 0.9,
        "likes_acoustic": True,
    },
}


def print_recs(name: str, user_prefs: dict, songs: list) -> None:
    print(f"\n=== {name} ===")
    print(f"Profile: {user_prefs}")
    recommendations = recommend_songs(user_prefs, songs, k=TOP_K_RANK)
    print(f"\n{'#':<3}{'Title':<28}{'Artist':<20}{'Score':>6}")
    print("-" * 70)
    for i, (song, score, explanation) in enumerate(recommendations, 1):
        print(f"{i:<3}{song['title']:<28}{song['artist']:<20}{score:>6.2f}")
        print(f"   Because: {explanation}\n")


def run_demo() -> None:
    """Replay the original four-profile evaluation (preserved for Phase-4 evals)."""
    songs = load_songs("data/songs.csv")
    for name, prefs in PROFILES.items():
        print_recs(name, prefs, songs)


def _rerank(
    candidates: List[Tuple[Dict, float]], user_prefs: Dict, k: int
) -> List[Tuple[Dict, float, float, List[str]]]:
    """Re-rank retrieval candidates with score_song. Returns (song, retrieval, score, reasons)."""
    rescored = []
    for song, retrieval_score in candidates:
        score, reasons = score_song(user_prefs, song)
        rescored.append((song, retrieval_score, score, reasons))
    rescored.sort(key=lambda x: (-x[2], x[0]["id"]))
    return rescored[:k]


def run_query(query: str) -> int:
    # Imported lazily so `--demo` works without numpy / sentence-transformers installed.
    from src.query_parser import parse_query
    from src.rag import build_index, retrieve

    if not query or not query.strip():
        print("Empty query. Try: python -m src.main \"chill lofi for studying\"")
        return 1

    songs = load_songs("data/songs.csv")
    user_prefs = parse_query(query)

    print(f"\nQuery: {query!r}")
    print(f"Parsed prefs: {user_prefs}")

    vectors, songs = build_index(songs)
    candidates = retrieve(query, vectors, songs, top_n=TOP_N_RETRIEVE)

    print(f"\nTop-{len(candidates)} retrieval candidates (cosine):")
    print(f"{'#':<3}{'Title':<28}{'Artist':<20}{'Cos':>6}")
    print("-" * 70)
    for i, (song, cos) in enumerate(candidates, 1):
        print(f"{i:<3}{song['title']:<28}{song['artist']:<20}{cos:>6.3f}")

    if not candidates:
        # Fallback: rank the whole catalog with score_song so we still return something.
        print("\nNo retrieval candidates; falling back to full-catalog scoring.")
        candidates = [(s, 0.0) for s in songs]

    final = _rerank(candidates, user_prefs, k=TOP_K_RANK)

    print(f"\nFinal top-{TOP_K_RANK} (re-ranked by score_song):")
    print(f"{'#':<3}{'Title':<28}{'Artist':<20}{'Cos':>6}{'Score':>8}")
    print("-" * 78)
    for i, (song, cos, score, reasons) in enumerate(final, 1):
        print(f"{i:<3}{song['title']:<28}{song['artist']:<20}{cos:>6.3f}{score:>8.2f}")
        print(f"   Because: {'; '.join(reasons)}\n")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="music-recommender",
        description="Free-text music recommender (RAG over a small song catalog).",
    )
    parser.add_argument("query", nargs="*", help="Free-text query, e.g. 'chill lofi for studying'.")
    parser.add_argument("--demo", action="store_true", help="Replay the four Phase-4 profiles.")
    args = parser.parse_args(argv)

    if args.demo:
        run_demo()
        return 0
    return run_query(" ".join(args.query))


if __name__ == "__main__":
    sys.exit(main())
