"""Streamlit UI wrapper around the existing CLI pipeline

Run with:
    streamlit run app.py
"""

import streamlit as st

from src.query_parser import parse_query
from src.rag import build_index, retrieve
from src.recommender import load_songs, score_song

TOP_N_RETRIEVE = 15
TOP_K_RANK = 5

st.set_page_config(page_title="VibeFinder", page_icon="🎵", layout="centered")


@st.cache_resource(show_spinner="Loading catalog and embedding model…")
def _load_pipeline():
    songs = load_songs("data/songs.csv")
    vectors, songs = build_index(songs)
    return vectors, songs


def _rerank(candidates, user_prefs, k):
    rescored = []
    for song, cos in candidates:
        score, reasons = score_song(user_prefs, song)
        rescored.append((song, cos, score, reasons))
    rescored.sort(key=lambda x: (-x[2], x[0]["id"]))
    return rescored[:k]


st.title("🎵 VibeFinder")
st.caption("Describe the vibe in plain English. The recommender retrieves semantically similar songs and re-ranks them with an explainable rule-based scorer.")

vectors, songs = _load_pipeline()

query = st.text_input(
    "What are you in the mood for?",
    placeholder="e.g. chill lofi for studying",
    value="chill lofi for studying",
)

if not query.strip():
    st.info("Enter a query above to see recommendations.")
    st.stop()

user_prefs = parse_query(query)
candidates = retrieve(query, vectors, songs, top_n=TOP_N_RETRIEVE)

if not candidates:
    candidates = [(s, 0.0) for s in songs]

final = _rerank(candidates, user_prefs, k=TOP_K_RANK)

st.subheader("Parsed preferences")
st.json(user_prefs)

st.subheader(f"Top {TOP_K_RANK} recommendations")
for i, (song, cos, score, reasons) in enumerate(final, 1):
    with st.container(border=True):
        cols = st.columns([6, 2, 2])
        cols[0].markdown(f"**{i}. {song['title']}** — {song['artist']}")
        cols[1].metric("Cosine", f"{cos:.3f}")
        cols[2].metric("Score", f"{score:.2f}")
        st.markdown(f"_Because:_ {'; '.join(reasons)}")

with st.expander(f"Show top-{TOP_N_RETRIEVE} retrieval candidates (cosine only)"):
    st.table(
        [
            {
                "#": i,
                "Title": song["title"],
                "Artist": song["artist"],
                "Genre": song["genre"],
                "Mood": song["mood"],
                "Cosine": round(cos, 3),
            }
            for i, (song, cos) in enumerate(candidates, 1)
        ]
    )
