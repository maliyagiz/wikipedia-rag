"""Tiny rule-based query classifier: person | place | both.

The spec explicitly says rule-based / keyword-based is fine. We:
1) Try exact (case-insensitive) name matches against the ingested entities.
2) Fall back to keyword cues ("where", "located", "tower" -> place, etc.).
3) Default to `both` when uncertain so the LLM can still answer.
"""
from __future__ import annotations

from config import PEOPLE, PLACES

PLACE_HINTS = {
    "where", "located", "country", "city", "tower", "wall", "mountain",
    "river", "monument", "palace", "temple", "bridge", "statue", "pyramid",
    "tall", "height", "built", "visit",
}
PERSON_HINTS = {
    "who", "born", "discover", "invented", "wrote", "painted", "famous for",
    "scientist", "artist", "actor", "singer", "player", "wife", "husband",
}


def _norm(s: str) -> str:
    return s.lower().strip()


def classify(query: str) -> str:
    q = _norm(query)

    person_matches = sum(1 for p in PEOPLE if _norm(p) in q)
    place_matches = sum(1 for p in PLACES if _norm(p) in q)

    if person_matches and place_matches:
        return "both"
    if person_matches:
        return "person"
    if place_matches:
        return "place"

    person_hits = sum(1 for w in PERSON_HINTS if w in q)
    place_hits = sum(1 for w in PLACE_HINTS if w in q)

    if person_hits and place_hits:
        return "both"
    if person_hits > place_hits:
        return "person"
    if place_hits > person_hits:
        return "place"
    return "both"
