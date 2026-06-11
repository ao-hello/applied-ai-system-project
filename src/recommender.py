"""Core recommender logic: data loading, scoring, and ranking"""

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

GENRE_WEIGHT = 2.0
MOOD_WEIGHT = 1.0
ENERGY_WEIGHT = 1.0
ACOUSTIC_BONUS = 0.5
ACOUSTIC_THRESHOLD = 0.7


@dataclass
class Song:
    """A single track and its numeric/categorical attributes"""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """A listener's taste preferences used to score songs"""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


def _score(
    song_genre: str,
    song_mood: str,
    song_energy: float,
    song_acousticness: float,
    pref_genre: str,
    pref_mood: str,
    target_energy: float,
    likes_acoustic: bool,
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    if song_genre == pref_genre:
        score += GENRE_WEIGHT
        reasons.append(f"genre match ({song_genre}) +{GENRE_WEIGHT}")

    if song_mood == pref_mood:
        score += MOOD_WEIGHT
        reasons.append(f"mood match ({song_mood}) +{MOOD_WEIGHT}")

    energy_sim = 1.0 - abs(song_energy - target_energy)
    energy_points = ENERGY_WEIGHT * energy_sim
    score += energy_points
    reasons.append(
        f"energy sim (|{song_energy:.2f}-{target_energy:.2f}|) +{energy_points:.2f}"
    )

    if likes_acoustic and song_acousticness > ACOUSTIC_THRESHOLD:
        score += ACOUSTIC_BONUS
        reasons.append(f"acoustic bonus ({song_acousticness:.2f}) +{ACOUSTIC_BONUS}")

    return score, reasons


class Recommender:
    """OOP recommender operating over `Song` + `UserProfile` dataclasses"""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Return (score, reasons) for one song against a user profile."""
        return _score(
            song.genre, song.mood, song.energy, song.acousticness,
            user.favorite_genre, user.favorite_mood,
            user.target_energy, user.likes_acoustic,
        )

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return top-k songs ranked by score desc, ties broken by id."""
        scored = [(self.score(user, s)[0], s.id, s) for s in self.songs]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [s for _, _, s in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why `song` fits `user`."""
        score, reasons = self.score(user, song)
        return f"Score {score:.2f}: " + "; ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from CSV into list of dicts, casting numeric columns"""
    float_cols = {"energy", "valence", "danceability", "acousticness"}
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("id"):
                continue
            row["id"] = int(row["id"])
            row["tempo_bpm"] = int(row["tempo_bpm"])
            for key in float_cols:
                row[key] = float(row[key])
            songs.append(row)
    print(f"Loaded songs: {len(songs)}")
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a song dict against user_prefs dict; return (score, reasons)"""
    return _score(
        song["genre"], song["mood"], song["energy"], song["acousticness"],
        user_prefs.get("genre", ""), user_prefs.get("mood", ""),
        float(user_prefs.get("energy", 0.5)),
        bool(user_prefs.get("likes_acoustic", False)),
    )


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """Rank songs by score_song, return top-k as (song, score, explanation)"""
    scored = [
        (song, *score_song(user_prefs, song)) for song in songs
    ]
    scored_fmt = [(s, sc, "; ".join(rs)) for s, sc, rs in scored]
    scored_fmt.sort(key=lambda x: (-x[1], x[0]["id"]))
    return scored_fmt[:k]
