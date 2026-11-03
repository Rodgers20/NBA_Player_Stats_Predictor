"""One position vocabulary, shared by every source.

Three feeds describe a player's position and none of them agree:

* **football-data.org** — `Offence`, `Midfield`, `Defence`, `Goalkeeper`
* **API-Football** — `Attacker`, `Midfielder`, `Defender`, `Goalkeeper`
* **StatsBomb events** — `Right Wing`, `Left Center Back`, and twenty more

The front end used to guess by substring, matching `"back"`, `"midfield"`,
`"wing"` and `"forward"`. None of those words appear in the first two feeds, so
`Offence`, `Defence`, `Attacker` and `Defender` all fell through to the default
and rendered as **MF** — a striker and a centre-back both drawn in midfield.

Two rules follow from that:

1. Resolution is a **table**, not substring matching.
2. An unrecognised value resolves to **None**, never to a default position. A
   blank badge is honest; a confident wrong one is not.

Coarse sources produce coarse badges (`FW`, `MF`, `DF`) rather than being
promoted to a specific role. Rendering "ST" for a player the feed only calls
"Offence" would be inventing precision the source never claimed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """A resolved position.

    `x` and `y` are pitch coordinates in a 0-1 space, attacking upward, so the
    badge can be drawn without the caller knowing anything about the vocabulary.
    """
    code: str
    label: str
    #: "specific" when the source named a role, "coarse" when it named a band.
    tier: str
    x: float
    y: float


def _p(code: str, label: str, tier: str, x: float, y: float) -> Position:
    return Position(code=code, label=label, tier=tier, x=x, y=y)


# Coarse bands. Drawn in the centre of their third, because the source did not
# say which side of the pitch the player occupies.
_COARSE: dict[str, Position] = {
    "goalkeeper": _p("GK", "Goalkeeper", "coarse", 0.50, 0.92),
    "defence": _p("DF", "Defender", "coarse", 0.50, 0.76),
    "defender": _p("DF", "Defender", "coarse", 0.50, 0.76),
    "midfield": _p("MF", "Midfielder", "coarse", 0.50, 0.50),
    "midfielder": _p("MF", "Midfielder", "coarse", 0.50, 0.50),
    "offence": _p("FW", "Forward", "coarse", 0.50, 0.18),
    "offense": _p("FW", "Forward", "coarse", 0.50, 0.18),
    "attacker": _p("FW", "Forward", "coarse", 0.50, 0.18),
    "forward": _p("FW", "Forward", "coarse", 0.50, 0.18),
}

# StatsBomb's granular vocabulary, as it appears in the event data.
_SPECIFIC: dict[str, Position] = {
    "goalkeeper": _p("GK", "Goalkeeper", "specific", 0.50, 0.93),

    "right back": _p("RB", "Right Back", "specific", 0.82, 0.76),
    "left back": _p("LB", "Left Back", "specific", 0.18, 0.76),
    "right center back": _p("RCB", "Right Centre Back", "specific", 0.62, 0.80),
    "left center back": _p("LCB", "Left Centre Back", "specific", 0.38, 0.80),
    "center back": _p("CB", "Centre Back", "specific", 0.50, 0.80),
    "right wing back": _p("RWB", "Right Wing Back", "specific", 0.86, 0.66),
    "left wing back": _p("LWB", "Left Wing Back", "specific", 0.14, 0.66),

    "right defensive midfield": _p("RDM", "Right Defensive Midfield", "specific", 0.64, 0.62),
    "left defensive midfield": _p("LDM", "Left Defensive Midfield", "specific", 0.36, 0.62),
    "center defensive midfield": _p("CDM", "Defensive Midfield", "specific", 0.50, 0.62),
    "right center midfield": _p("RCM", "Right Centre Midfield", "specific", 0.64, 0.48),
    "left center midfield": _p("LCM", "Left Centre Midfield", "specific", 0.36, 0.48),
    "center midfield": _p("CM", "Centre Midfield", "specific", 0.50, 0.48),
    "right midfield": _p("RM", "Right Midfield", "specific", 0.84, 0.48),
    "left midfield": _p("LM", "Left Midfield", "specific", 0.16, 0.48),
    "right attacking midfield": _p("RAM", "Right Attacking Midfield", "specific", 0.64, 0.32),
    "left attacking midfield": _p("LAM", "Left Attacking Midfield", "specific", 0.36, 0.32),
    "center attacking midfield": _p("CAM", "Attacking Midfield", "specific", 0.50, 0.32),

    "right wing": _p("RW", "Right Wing", "specific", 0.86, 0.22),
    "left wing": _p("LW", "Left Wing", "specific", 0.14, 0.22),
    "right center forward": _p("RCF", "Right Centre Forward", "specific", 0.62, 0.14),
    "left center forward": _p("LCF", "Left Centre Forward", "specific", 0.38, 0.14),
    "center forward": _p("ST", "Striker", "specific", 0.50, 0.12),
    "secondary striker": _p("SS", "Second Striker", "specific", 0.50, 0.24),
    "striker": _p("ST", "Striker", "specific", 0.50, 0.12),
}

_TABLE: dict[str, Position] = {**_COARSE, **_SPECIFIC}


def resolve(position: str | None) -> Position | None:
    """Resolve any source's position string, or None if it is unrecognised.

    Returning None is deliberate. The previous substring ladder defaulted to
    midfield, which silently mislabelled every player whose position came from a
    current-season feed.
    """
    if not position:
        return None
    return _TABLE.get(" ".join(position.strip().lower().split()))


def known_values() -> list[str]:
    """Every string this module resolves, for coverage checks."""
    return sorted(_TABLE)
