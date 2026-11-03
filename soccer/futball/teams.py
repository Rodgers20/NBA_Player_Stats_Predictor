"""Team-name reconciliation across sources.

football-data.co.uk writes "Man City"; openfootball writes "Manchester City FC";
StatsBomb writes "Manchester City". A league table built from one source and a
fixture list built from another will silently disagree unless the names are
reconciled, producing phantom clubs and broken joins.

The canonical spelling is **football-data.co.uk's**, because the results lake and
the rolling fixture file both use it, and those two drive standings and fixtures.

Reconciliation is deliberately conservative: normalise aggressively, then match.
A name that does not resolve is REPORTED AND DROPPED rather than guessed — a
wrong merge corrupts a league table in a way that is very hard to notice.
"""
from __future__ import annotations

import re
import unicodedata

# Club-name noise that carries no identity.
_AFFIXES = (
    "fc", "cf", "afc", "sc", "ac", "as", "ss", "ssc", "us", "usl", "sv", "tsv",
    "vfl", "vfb", "tsg", "bsc", "fsv", "sp", "cd", "rc", "rcd", "ud", "sd",
    "club", "calcio", "de", "futbol", "football", "association", "sport",
    "sporting", "societa", "spa", "srl", "eV", "ev", "borussia", "real",
)
_STOP_AT_START = ("1", "1.", "fc", "ac", "as", "ss", "sv", "vfl", "vfb", "tsg")


def _strip_accents(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(c)
    )


def normalise(name: str) -> str:
    """Aggressively reduce a club name to a comparable key."""
    if not name:
        return ""
    s = _strip_accents(name).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t]
    # drop affix tokens, but never reduce a name to nothing
    kept = [t for t in tokens if t not in _AFFIXES]
    if not kept:
        kept = tokens
    return " ".join(kept)


def _key_variants(name: str) -> set[str]:
    """Safe comparison keys for a club name.

    Single trailing tokens are deliberately NOT used. "Hull City" reduces to the
    token "city", which then matched "Manchester City" and produced a fixture of
    Man City versus Man City. Identity-bearing tokens ("city", "united",
    "rovers", "athletic") are shared by dozens of clubs, so only the full
    normalised name and its leading tokens are safe.
    """
    base = normalise(name)
    variants = {base}
    parts = base.split()
    if len(parts) > 1:
        variants.add(" ".join(parts[:2]))
        if parts[0] not in _GENERIC:
            variants.add(parts[0])
    return {v for v in variants if len(v) >= 4}


# Tokens that name a club type rather than a club.
_GENERIC = {
    "city", "united", "town", "rovers", "wanderers", "albion", "athletic",
    "county", "hotspur", "villa", "forest", "palace", "olympique", "stade",
    "real", "atletico", "sporting", "racing", "inter", "dynamo", "union",
}


# Cases normalisation cannot reach, because the two sources use genuinely
# different words rather than different decorations.
MANUAL: dict[str, str] = {
    "manchester city": "Man City",
    "manchester united": "Man United",
    "newcastle united": "Newcastle",
    "tottenham hotspur": "Tottenham",
    "wolverhampton wanderers": "Wolves",
    "brighton and hove albion": "Brighton",
    "west ham united": "West Ham",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "norwich city": "Norwich",
    "nottingham forest": "Nott'm Forest",
    "sheffield united": "Sheffield United",
    "west bromwich albion": "West Brom",
    "queens park rangers": "QPR",
    "athletic bilbao": "Ath Bilbao",
    "atletico madrid": "Ath Madrid",
    "atletico de madrid": "Ath Madrid",
    "real sociedad": "Sociedad",
    "real betis": "Betis",
    "celta vigo": "Celta",
    "rayo vallecano": "Vallecano",
    "bayern munchen": "Bayern Munich",
    "bayern munich": "Bayern Munich",
    "borussia dortmund": "Dortmund",
    "dortmund": "Dortmund",
    "borussia monchengladbach": "M'gladbach",
    "monchengladbach": "M'gladbach",
    "bayer leverkusen": "Leverkusen",
    "leverkusen": "Leverkusen",
    "eintracht frankfurt": "Ein Frankfurt",
    "werder bremen": "Werder Bremen",
    "hertha berlin": "Hertha",
    "internazionale milano": "Inter",
    "internazionale": "Inter",
    "milan": "Milan",
    "juventus": "Juventus",
    "napoli": "Napoli",
    "roma": "Roma",
    "lazio": "Lazio",
    "hellas verona": "Verona",
    "paris saint germain": "Paris SG",
    "paris saint-germain": "Paris SG",
    "olympique marseille": "Marseille",
    "olympique de marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "saint etienne": "St Etienne",
    # resolved by inspecting what the resolver could not place, then checking
    # each against the canonical list rather than guessing
    "athletic": "Ath Bilbao",
    "rayo vallecano madrid": "Vallecano",
    "1 heidenheim 1846": "Heidenheim",
    "1 mainz 05": "Mainz",
    "hamburger": "Hamburg",
    "stade brestois 29": "Brest",
    "stade rennais 1901": "Rennes",
    "1 koln": "FC Koln",
    "1 union berlin": "Union Berlin",
    "borussia moenchengladbach": "M'gladbach",
    "coventry city": "Coventry",
    "ca osasuna": "Osasuna",
    "deportivo alaves": "Alaves",
    "espanyol barcelona": "Espanol",
    "bayer 04 leverkusen": "Leverkusen",
    "1899 hoffenheim": "Hoffenheim",
    "acf fiorentina": "Fiorentina",
    "aj auxerre": "Auxerre",
    "ogc nice": "Nice",
    "racing lens": "Lens",
    # football-data.org spells the 2026/27 promoted Spanish clubs in full;
    # football-data.co.uk uses the short forms already in the lake. Confirmed
    # against the canonical SP1 name list before adding.
    "deportivo la coruna": "La Coruna",
    "racing santander": "Santander",
    # MLS: ESPN's spellings vs football-data.co.uk's USA.csv (checked against its 31 club names).
    "d c united": "DC United",
    "la galaxy": "Los Angeles Galaxy",
    "lafc": "Los Angeles FC",
    "red bull new york": "New York Red Bulls",
    "new york red bulls": "New York Red Bulls",
    # ESPN's short or English spellings for three European clubs (checked against the lake names).
    "deportivo": "La Coruna",
    "espanyol": "Espanol",
    "cologne": "FC Koln",
}


class TeamResolver:
    """Resolves a foreign spelling onto the canonical football-data name."""

    def __init__(self, canonical: list[str]) -> None:
        self.canonical = sorted({c for c in canonical if c})
        self._canonical_set = set(self.canonical)
        self._exact: dict[str, str] = {}
        self._loose: dict[str, str] = {}
        for name in self.canonical:
            self._exact[normalise(name)] = name
            for v in _key_variants(name):
                # first writer wins; ambiguous variants are removed below
                self._loose.setdefault(v, name)
        # drop variants that map to more than one canonical club
        counts: dict[str, set[str]] = {}
        for name in self.canonical:
            for v in _key_variants(name):
                counts.setdefault(v, set()).add(name)
        for v, owners in counts.items():
            if len(owners) > 1:
                self._loose.pop(v, None)

    def resolve(self, name: str) -> str | None:
        """Canonical name, or None. Never guesses."""
        if not name:
            return None
        if name in self._canonical_set:
            return name

        key = normalise(name)
        if key in self._exact:
            return self._exact[key]

        if key in MANUAL:
            target = MANUAL[key]
            return target if target in self._canonical_set else None

        # Prefix containment: one normalised name must start with the other, and
        # the shorter must carry enough of the longer to be identifying. This
        # accepts "nottingham forest" ~ "nott m forest" only via MANUAL, and
        # rejects "hull city" ~ "manchester city" outright.
        candidates = [
            canon for canon_key, canon in self._exact.items()
            if (key.startswith(canon_key) or canon_key.startswith(key))
            and min(len(key), len(canon_key)) >= 4
        ]
        if len(candidates) == 1:
            return candidates[0]

        for v in sorted(_key_variants(name), key=len, reverse=True):
            if v in self._loose:
                return self._loose[v]
        return None
