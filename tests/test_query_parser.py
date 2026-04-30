"""Unit tests for the keyword/regex query parser."""

import pytest

from src.query_parser import (
    DEFAULT_ENERGY,
    HIGH_ENERGY_TARGET,
    LOW_ENERGY_TARGET,
    parse_query,
)


def test_chill_lofi_for_studying_extracts_lofi_focused_low_energy():
    prefs = parse_query("chill lofi for studying")
    assert prefs["genre"] == "lofi"
    assert prefs["mood"] == "focused"
    assert prefs["energy"] == LOW_ENERGY_TARGET


def test_high_energy_pop_workout():
    prefs = parse_query("high energy pop for the gym")
    assert prefs["genre"] == "pop"
    assert prefs["mood"] == "intense"
    assert prefs["energy"] == HIGH_ENERGY_TARGET
    assert prefs["likes_acoustic"] is False


def test_acoustic_folk_for_a_slow_morning():
    prefs = parse_query("acoustic folk for a slow morning")
    assert prefs["genre"] == "folk"
    assert prefs["likes_acoustic"] is True
    assert prefs["energy"] == LOW_ENERGY_TARGET


def test_indie_pop_beats_pop_when_both_words_present():
    prefs = parse_query("indie pop summer playlist")
    assert prefs["genre"] == "indie pop"


def test_hip_hop_synonyms():
    assert parse_query("hard trap for the gym")["genre"] == "hip hop"
    assert parse_query("hip-hop party")["genre"] == "hip hop"


def test_empty_query_returns_neutral_defaults():
    prefs = parse_query("")
    assert prefs["genre"] == ""
    assert prefs["mood"] == ""
    assert prefs["energy"] == DEFAULT_ENERGY
    assert prefs["likes_acoustic"] is False


def test_mixed_high_and_low_cues_falls_back_to_default():
    # one high cue, one low cue, neither dominates
    prefs = parse_query("intense but mellow vibes")
    assert prefs["energy"] == DEFAULT_ENERGY


@pytest.mark.parametrize(
    "query,expected_mood",
    [
        ("songs to help me focus", "focused"),
        ("calm music for sleep", "relaxed"),
        ("upbeat happy summer hits", "happy"),
        ("aggressive workout playlist", "intense"),
        ("late night moody synthwave", "moody"),
    ],
)
def test_mood_extraction(query, expected_mood):
    assert parse_query(query)["mood"] == expected_mood
