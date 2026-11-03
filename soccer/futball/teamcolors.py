"""Club and nation colours, for the player page's banner and accents.

Lookups go through `teams.TeamResolver`, the same reconciliation the results
lake uses, so one entry covers every spelling the three sources produce:
"Man City", "Manchester City FC" and "Manchester City" all land on one colour.
Raw `normalise` is not enough — it maps those to three different keys, and the
alias table in `teams.MANUAL` is what bridges them.

Each entry stores only the club's primary colour. The text colour over it is
*computed* — whichever of near-black or white contrasts better — because a
hand-maintained pairing drifts: three entries here originally shipped white text
on mid-tone green and cyan, which failed WCAG even for large display type.
Deriving it means every club clears the large-text AA threshold by construction,
with 4.33:1 the worst case.
"""
from __future__ import annotations

from .teams import MANUAL, TeamResolver, normalise

# Fallback: the app's own accent, used for any club not listed. Never guess a
# colour from the name — a wrong club colour reads as a bug, a neutral one does
# not.
DEFAULT: str = "#2B3A4A"

_DARK = "#111418"
_LIGHT = "#FFFFFF"

# (display name, primary brand colour)
_CLUBS: tuple[tuple[str, str, str], ...] = (
    # --- Premier League ---
    ("Arsenal", "#EF0107"),
    ("Aston Villa", "#670E36"),
    ("Bournemouth", "#DA291C"),
    ("Brentford", "#E30613"),
    ("Brighton", "#0057B8"),
    ("Chelsea", "#034694"),
    ("Coventry", "#78D2F1"),
    ("Crystal Palace", "#1B458F"),
    ("Everton", "#003399"),
    ("Fulham", "#FFFFFF"),
    ("Hull", "#F5A12D"),
    ("Ipswich", "#3A64A3"),
    ("Leeds", "#FFCD00"),
    ("Liverpool", "#C8102E"),
    ("Man City", "#6CABDD"),
    ("Man United", "#DA291C"),
    ("Newcastle", "#241F20"),
    ("Nott'm Forest", "#DD0000"),
    ("Sunderland", "#EB172B"),
    ("Tottenham", "#132257"),
    # --- La Liga ---
    ("Alaves", "#0761AF"),
    ("Ath Bilbao", "#EE2523"),
    ("Ath Madrid", "#CB3524"),
    ("Barcelona", "#A50044"),
    ("Betis", "#00954C"),
    ("Celta", "#8AC3EE"),
    ("Elche", "#00693E"),
    ("Espanol", "#0072BC"),
    ("Getafe", "#005999"),
    ("La Coruna", "#0067B1"),
    ("Levante", "#004B9D"),
    ("Malaga", "#00A0E6"),
    ("Osasuna", "#0A346F"),
    ("Real Madrid", "#FFFFFF"),
    ("Santander", "#00B451"),
    ("Sevilla", "#D00027"),
    ("Sociedad", "#0067B1"),
    ("Valencia", "#FFFFFF"),
    ("Vallecano", "#E53027"),
    ("Villarreal", "#FFE667"),
    # --- other Champions League clubs ---
    ("AS Roma", "#8E1F2F"),
    ("Borussia Dortmund", "#FDE100"),
    ("Club Brugge", "#0077C8"),
    ("Como 1907", "#005CA9"),
    ("Bayern Munich", "#DC052D"),
    ("Inter", "#0068A8"),
    ("Porto", "#00428C"),
    ("Bodo/Glimt", "#FFE600"),
    ("Shakhtar Donetsk", "#F47A20"),
    ("Fenerbahce", "#1B234B"),
    ("Feyenoord", "#DA020E"),
    ("Galatasaray", "#A90432"),
    ("LASK Linz", "#000000"),
    ("Lille", "#E01E13"),
    ("AEK", "#F8C300"),
    ("PSV", "#ED1C24"),
    ("Paris SG", "#004170"),
    ("RB Leipzig", "#DD0741"),
    ("Lens", "#FFE500"),
    ("Slavia Praha", "#D7141A"),
    ("Napoli", "#12A0D7"),
    ("Sabah FK", "#00A650"),
    ("Sporting CP", "#008057"),
    ("Stuttgart", "#FFFFFF"),
    ("Viking FK", "#003D7C"),
    ("Slovan Bratislava", "#005BAB"),
    # --- Euro 2024 nations (StatsBomb competition) ---
    ("Spain", "#C60B1E"),
    ("England", "#FFFFFF"),
    ("France", "#21304F"),
    ("Germany", "#000000"),
    ("Portugal", "#006600"),
    ("Netherlands", "#F36C21"),
    ("Italy", "#0066CC"),
    ("Belgium", "#E30613"),
    ("Switzerland", "#D52B1E"),
    ("Austria", "#ED2939"),
    ("Turkey", "#E30A17"),
    ("Denmark", "#C8102E"),
    ("Croatia", "#FF0000"),
    ("Poland", "#DC143C"),
    ("Ukraine", "#FFD700"),
    ("Scotland", "#0065BF"),
    ("Serbia", "#C6363C"),
    ("Czech Republic", "#11457E"),
    ("Slovenia", "#003D7C"),
    ("Slovakia", "#0B4EA2"),
    ("Romania", "#FCD116"),
    ("Hungary", "#436F4D"),
    ("Albania", "#E41E20"),
    ("Georgia", "#FF0000"),
)

_BY_NAME: dict[str, str] = {name: primary for name, primary in _CLUBS}

# European clubs whose long UEFA spelling `teams.MANUAL` does not cover, because
# they have never appeared in a football-data.co.uk division file.
_EXTRA_ALIASES: dict[str, str] = {
    "fk bod glimt": "Bodo/Glimt",
    "bodo glimt": "Bodo/Glimt",
    "glimt": "Bodo/Glimt",
    "fk shakhtar donetsk": "Shakhtar Donetsk",
    "shakhtar donetsk": "Shakhtar Donetsk",
    "fenerbahce sk": "Fenerbahce",
    "fenerbahce": "Fenerbahce",
    "feyenoord rotterdam": "Feyenoord",
    "galatasaray sk": "Galatasaray",
    "pae aek": "AEK",
    "aek athen": "AEK",
    "sk slavia praha": "Slavia Praha",
    "slavia praha": "Slavia Praha",
    "sporting clube portugal": "Sporting CP",
    "sporting clube de portugal": "Sporting CP",
    "sporting cp": "Sporting CP",
    "clube portugal": "Sporting CP",
    "sk slovan bratislava": "Slovan Bratislava",
    "slovan bratislava": "Slovan Bratislava",
    "club brugge kv": "Club Brugge",
    "brugge kv": "Club Brugge",
    "internazionale milano": "Inter",
    "lille osc": "Lille",
    "losc lille": "Lille",
    "betis balompie": "Betis",
    "vfb stuttgart": "Stuttgart",
    "ssc napoli": "Napoli",
    "acf fiorentina": "Fiorentina",
    "czechia": "Czech Republic",
    "turkiye": "Turkey",
    "barca": "Barcelona",
    "fc barcelona": "Barcelona",
}

_resolver = TeamResolver(list(_BY_NAME))


def _lookup(team: str) -> str | None:
    """Resolve any source's spelling onto one colour entry."""
    if team in _BY_NAME:
        return _BY_NAME[team]

    key = normalise(team)
    for table in (_EXTRA_ALIASES, MANUAL):
        target = table.get(key)
        if target in _BY_NAME:
            return _BY_NAME[target]

    resolved = _resolver.resolve(team)
    return _BY_NAME.get(resolved) if resolved else None


def colors(team: str | None) -> dict[str, str]:
    """Banner colours for a club or nation.

    Returns `primary` (the banner fill), `on_primary` (readable text over it)
    and `accent`, which is `primary` unless that is so light it would vanish
    against the page — Real Madrid and Fulham play in white, and a white accent
    on a dark page is invisible.
    """
    primary = _lookup((team or "").strip()) or DEFAULT
    return {
        "primary": primary,
        "on_primary": on_primary_for(primary),
        "accent": _readable_accent(primary),
    }


def on_primary_for(primary: str) -> str:
    """Whichever of near-black or white is more readable on `primary`."""
    return max((_DARK, _LIGHT), key=lambda text: contrast_ratio(primary, text))


def contrast_ratio(a: str, b: str) -> float:
    """WCAG relative contrast between two hex colours."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _readable_accent(primary: str) -> str:
    """A version of `primary` that stays visible as text on a dark surface."""
    if _relative_luminance(primary) < 0.75:
        return primary
    # Too pale to read: darken toward a legible slate rather than inventing a hue.
    return "#9AA6B2"


def _relative_luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    channels = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in (r, g, b)
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def known_teams() -> list[str]:
    """Display names that have a colour, for coverage checks."""
    return [name for name, _primary in _CLUBS]
