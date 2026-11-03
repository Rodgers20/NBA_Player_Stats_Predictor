"""Club directory: crests, display names, and who plays for whom.

Built from football-data.org's `/competitions/{code}/teams`, which returns every
club in a competition with an official crest URL and its full squad. Two
problems this solves, both visible on the player pages:

1. **Badges.** The UI previously drew a club's initials whenever a crest could
   not be resolved, producing "BA" for Barça and "C" for Celta. Every club in
   the three competitions now has a real crest URL.
2. **Club, not country.** StatsBomb's Euro 2024 data knows only national teams,
   so a player page coloured itself France blue for Mbappé. The squad lists give
   a club for ~1,000 players, so the page can show Real Madrid white instead.
   A player with no known club falls back to the neutral default — never to the
   national colour, which is what made Mbappé blue in the first place.

Everything is read from the raw cache; refreshing is `cli.cmd_refresh`'s job, so
importing this never touches the network.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from . import config
from .sources import football_data_org as fdo
from .teams import MANUAL, TeamResolver, normalise

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def directory() -> dict[str, dict]:
    """Canonical club name -> {name, short_name, crest, team_id, competitions}.

    Keyed by the club's *display* name as football-data.org spells it; use
    `find` to resolve any other source's spelling onto an entry.
    """
    out: dict[str, dict] = {}
    for code in fdo.COMPETITIONS:
        payload = fdo.read_teams(code)
        if not payload:
            continue
        for row in fdo.parse_teams(payload):
            entry = out.setdefault(row["name"], {
                "name": row["name"],
                "short_name": row["short_name"],
                "tla": row["tla"],
                "crest": row["crest"],
                "team_id": row["team_id"],
                "founded": row.get("founded"),
                "venue": row.get("venue"),
                "website": row.get("website"),
                "club_colors": row.get("club_colors"),
                "country": row.get("country"),
                "country_flag": row.get("country_flag"),
                "coach": row.get("coach"),
                "squad": row.get("squad") or [],
                "competitions": [],
            })
            if code not in entry["competitions"]:
                entry["competitions"].append(code)
    return out


# Spellings the shared resolver cannot bridge, because the two sources use
# genuinely different words rather than different decorations.
_CLUB_ALIASES: dict[str, str] = {
    "nott m forest": "Nottingham Forest FC",
    "nottingham forest": "Nottingham Forest FC",
    "la coruna": "RC Deportivo La Coruña",
    "deportivo": "RC Deportivo La Coruña",
    "santander": "Real Racing Club de Santander",
    "vallecano": "Rayo Vallecano de Madrid",
    "espanol": "RCD Espanyol de Barcelona",
    "sociedad": "Real Sociedad de Fútbol",
    "ath bilbao": "Athletic Club",
    "ath madrid": "Club Atlético de Madrid",
    "betis": "Real Betis Balompié",
    "alaves": "Deportivo Alavés",
    "malaga": "Málaga CF",
    "elche": "Elche CF",
    "levante": "Levante UD",
    "getafe": "Getafe CF",
    "osasuna": "CA Osasuna",
}


@lru_cache(maxsize=1)
def _resolver() -> TeamResolver:
    return TeamResolver(list(directory()))


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    """Every normalised spelling that should reach a directory entry.

    football-data.org uses three names for one club — "FC Barcelona",
    "Barça" and the crest's own naming — while the results lake uses
    football-data.co.uk's "Barcelona". All of them are indexed here so a crest
    lookup never depends on which source produced the string.
    """
    index: dict[str, str] = {}
    for name, entry in directory().items():
        for spelling in (name, entry.get("short_name")):
            if spelling:
                index.setdefault(normalise(spelling), name)
    # Clubs whose football-data.co.uk spelling the alias table does not bridge
    # to football-data.org's own naming. These targets are written to match a
    # directory key exactly; the fuzzy resolver is NOT used as a fallback here
    # because its prefix matching maps "RCD Espanyol de Barcelona" onto
    # "FC Barcelona", which would put Espanyol's crest on Barcelona.
    for foreign, canonical in _CLUB_ALIASES.items():
        if canonical in directory():
            index[foreign] = canonical
        else:
            log.warning("club alias target not in directory: %r", canonical)

    # football-data.co.uk spellings, via the project's existing alias table.
    # Same rule: exact directory names only.
    for foreign, canonical in MANUAL.items():
        if canonical in directory():
            index.setdefault(foreign, canonical)
    return index


def find(team: str | None) -> dict | None:
    """Directory entry for any spelling of a club, or None."""
    if not team:
        return None
    name = team.strip()
    entries = directory()
    if name in entries:
        return entries[name]
    hit = _alias_index().get(normalise(name))
    if hit:
        return entries[hit]
    resolved = _resolver().resolve(name)
    return entries.get(resolved) if resolved else None


def crest(team: str | None) -> str | None:
    """Official crest URL for a club, or None if it is not in the directory."""
    entry = find(team)
    return entry.get("crest") if entry else None


def display_name(team: str | None) -> str | None:
    """The short, readable form — "Barcelona" rather than "FC Barcelona"."""
    entry = find(team)
    if not entry:
        return team
    return entry.get("short_name") or entry.get("name")


@lru_cache(maxsize=1)
def _squad_index() -> dict[str, str]:
    """Normalised player name -> club display name.

    Where two clubs list the same normalised name, the first wins and a warning
    is logged rather than silently picking one — a wrong club would put the
    wrong colour and badge on that player's page.
    """
    index: dict[str, str] = {}
    clashes: set[str] = set()
    for code in fdo.COMPETITIONS:
        payload = fdo.read_teams(code)
        if not payload:
            continue
        for row in fdo.parse_teams(payload):
            for player in row["squad"]:
                key = normalise(player["name"])
                if not key:
                    continue
                if key in index and index[key] != row["name"]:
                    clashes.add(player["name"])
                    continue
                index.setdefault(key, row["name"])
    if clashes:
        log.info("squad index: %d ambiguous player name(s), kept first club: %s",
                 len(clashes), ", ".join(sorted(clashes)[:5]))
    return index


def club_of(player_name: str | None) -> dict | None:
    """The club a player belongs to, from the published squad lists.

    This is how a Euro 2024 page finds Mbappé's *club* colours when the event
    data only knows he played for France.

    StatsBomb records full legal names ("Kylian Mbappé Lottin") while squad
    lists carry the common form ("Kylian Mbappé"), so an exact miss falls back
    to matching a squad name that is a leading run of the given one. The match
    must be at least two tokens long, or "Rodri" and "Rodrigo" style collisions
    would attach players to the wrong club.
    """
    if not player_name:
        return None
    key = normalise(player_name)
    if not key:
        return None

    index = _squad_index()
    club = index.get(key)
    if club is None:
        tokens = key.split()
        for size in range(len(tokens) - 1, 1, -1):
            club = index.get(" ".join(tokens[:size]))
            if club:
                break
    return directory().get(club) if club else None


def directory_with_aliases() -> list[dict]:
    """Directory entries, each carrying every spelling that reaches it.

    The front end matches club names coming from three different feeds, and
    normalising on its own is not enough — "Nott'm Forest" and "Nottingham
    Forest FC" share no tokens. Shipping the alias list means the browser does
    not need its own copy of the reconciliation rules.
    """
    aliases: dict[str, set[str]] = {name: set() for name in directory()}
    for spelling, canonical in _alias_index().items():
        if canonical in aliases:
            aliases[canonical].add(spelling)
    return [
        {**entry, "aliases": sorted(aliases.get(name, ()))}
        for name, entry in directory().items()
    ]


def competition_emblem(code: str) -> str | None:
    """Competition badge URL, for the sidebar and page headers."""
    return config.COMPETITION_EMBLEMS.get(code)


def refresh_caches() -> None:
    """Drop the memoised indexes after the raw layer is rewritten."""
    for fn in (directory, _resolver, _alias_index, _squad_index, _squad_detail_index):
        fn.cache_clear()


@lru_cache(maxsize=1)
def _squad_detail_index() -> dict[str, dict]:
    """Normalised player name -> squad details from API-Football.

    Shirt number lives here and nowhere else: football-data.org's squad rows
    carry a `shirtNumber` field that is null for every player checked, while
    API-Football populates it.
    """
    from .sources import api_football as af

    index: dict[str, dict] = {}
    clashes: set[str] = set()
    for row in af.cached_squads():
        name = row.get("name")
        key = normalise(name or "")
        if not key:
            continue
        detail = {"number": row.get("number"), "age": row.get("age")}
        if key in index and index[key] != detail:
            clashes.add(key)
            continue
        index.setdefault(key, detail)
    for key in clashes:
        index.pop(key, None)
    return index


def squad_detail(player_name: str | None) -> dict:
    """Shirt number and age for a player, or empty when unknown.

    Falls back to a leading run of the name, because StatsBomb records full
    legal names ("Alexander Isak Sweden") where squads use the common form.
    """
    key = normalise(player_name or "")
    if not key:
        return {}
    index = _squad_detail_index()
    hit = index.get(key)
    if hit:
        return hit
    tokens = key.split()
    # Squad names are abbreviated ("A. Isak"), so try initial-plus-surname too.
    if len(tokens) >= 2:
        initial = f"{tokens[0][0]} {' '.join(tokens[1:])}"
        hit = index.get(initial)
        if hit:
            return hit
        for size in range(len(tokens), 1, -1):
            hit = index.get(f"{tokens[0][0]} {' '.join(tokens[1:size])}")
            if hit:
                return hit
        # And the reverse abbreviation: football-data.org carries the full legal
        # name where squads use the common one, so "Alvaro Valles Rosa" has to
        # reach "Alvaro Valles". Longest prefix first — the shortest is a bare
        # first name and would collide with half the division.
        for size in range(len(tokens) - 1, 1, -1):
            hit = index.get(" ".join(tokens[:size]))
            if hit:
                return hit
    return {}


def first_choice_keeper(team: str | None) -> dict | None:
    """The club's number 1, or None when the squad does not settle it.

    **This is the shirt, not the team sheet.** Line-ups are not published on any
    free tier, so who actually started cannot be known; the number 1 shirt is
    the closest published proxy and it is right for most clubs — Courtois,
    Raya, Donnarumma, Joan García all wear it. It is wrong wherever a
    first-choice keeper keeps another number: Oblak has worn 13 at Atlético for
    a decade, so Atlético resolves to whoever holds the 1.

    Callers must present the result as the club's number 1 and never as the
    keeper who played, and must cope with None.
    """
    entry = find(team) if team else None
    if not entry:
        return None
    keepers = [p for p in (entry.get("squad") or []) if p.get("position") == "Goalkeeper"]
    if not keepers:
        return None

    numbered = [(p, squad_detail(p.get("name")).get("number")) for p in keepers]
    wearing_one = [p for p, number in numbered if number == 1]
    if len(wearing_one) == 1:
        chosen, number = wearing_one[0], 1
    elif len(keepers) == 1:
        # One keeper in the squad: no ambiguity to resolve, shirt or not.
        chosen, number = numbered[0]
    else:
        return None

    return {
        "name": chosen.get("name"),
        "player_id": chosen.get("player_id"),
        "shirt": number,
        "nationality": chosen.get("nationality"),
        "club": entry.get("name"),
        "crest": entry.get("crest"),
    }
