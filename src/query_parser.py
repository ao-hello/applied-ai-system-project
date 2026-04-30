"""Keyword/regex parser: free-text query -> structured prefs dict consumed by score_song."""

import re
from typing import Dict

# Order matters: multi-word genres must match before their single-word components.
GENRE_PATTERNS = [
    ("indie pop", r"\bindie\s*pop\b"),
    ("hip hop", r"\bhip[\s-]?hop\b|\btrap\b|\brap\b"),
    ("synthwave", r"\bsynth\s*wave\b|\bsynthwave\b|\bretrowave\b"),
    ("r&b", r"\br\s*&\s*b\b|\brnb\b|\brhythm and blues\b"),
    ("lofi", r"\blo[\s-]?fi\b"),
    ("edm", r"\bedm\b|\belectronic dance\b|\bhouse music\b|\bfestival\b"),
    ("ambient", r"\bambient\b|\bsoundscape\b"),
    ("jazz", r"\bjazz\b"),
    ("folk", r"\bfolk\b|\bacoustic folk\b"),
    ("punk", r"\bpunk\b|\bhardcore\b"),
    ("rock", r"\brock\b|\bmetal\b|\bguitar\b"),
    ("pop", r"\bpop\b"),
]

MOOD_PATTERNS = [
    ("focused", r"\bfocus(ed|ing)?\b|\bstud(y|ying)\b|\bconcentrat\w*\b|\bdeep work\b"),
    ("chill", r"\bchill\b|\bmellow\b|\blaid[\s-]?back\b|\bunwind\b|\bcozy\b"),
    ("relaxed", r"\brelax\w*\b|\bcalm\b|\bsleep\b|\bmeditat\w*\b|\byoga\b|\bquiet\b"),
    ("happy", r"\bhappy\b|\bupbeat\b|\bjoyful\b|\bsunny\b|\bfun\b|\bcheer\w*\b"),
    ("intense", r"\bintense\b|\bhype\b|\benerget\w*\b|\bhard\b|\baggressive\b|\bworkout\b|\bgym\b|\bpump\w*\b"),
    ("moody", r"\bmood\w*\b|\bmelanchol\w*\b|\bsad\b|\bemotional\b|\bintrospect\w*\b|\bnight drive\b"),
]

# Cues that push target_energy up or down. Score-based so multiple cues compound.
HIGH_ENERGY_CUES = re.compile(
    r"\b(high energy|intense|hype|workout|gym|pump\w*|cardio|run|running|"
    r"party|festival|aggressive|hard|fast|loud|dance|dancing)\b"
)
LOW_ENERGY_CUES = re.compile(
    r"\b(chill|mellow|calm|low energy|soft|quiet|sleep|sleepy|meditat\w*|"
    r"relax\w*|study|focus\w*|reading|background|ambient|slow)\b"
)

ACOUSTIC_CUES = re.compile(
    r"\b(acoustic|unplugged|fingerpick\w*|folk|singer[\s-]?songwriter|"
    r"campfire|coffee\s?shop)\b"
)

DEFAULT_ENERGY = 0.5
HIGH_ENERGY_TARGET = 0.85
LOW_ENERGY_TARGET = 0.3


def _first_match(text: str, patterns) -> str:
    for label, pat in patterns:
        if re.search(pat, text):
            return label
    return ""


def _infer_energy(text: str) -> float:
    high = len(HIGH_ENERGY_CUES.findall(text))
    low = len(LOW_ENERGY_CUES.findall(text))
    if high > low:
        return HIGH_ENERGY_TARGET
    if low > high:
        return LOW_ENERGY_TARGET
    return DEFAULT_ENERGY


def parse_query(query: str) -> Dict:
    """Extract structured prefs from a free-text query.

    Returns a dict shaped for `score_song`: genre, mood, energy, likes_acoustic.
    Missing categorical fields are empty strings (no false matches in scorer).
    """
    text = (query or "").lower()
    return {
        "genre": _first_match(text, GENRE_PATTERNS),
        "mood": _first_match(text, MOOD_PATTERNS),
        "energy": _infer_energy(text),
        "likes_acoustic": bool(ACOUSTIC_CUES.search(text)),
    }
