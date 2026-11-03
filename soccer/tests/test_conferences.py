from futball import conferences as c


def body():
    def entry(name):
        return {"team": {"displayName": name}}
    return {"children": [
        {"name": "Eastern Conference", "standings": {"entries": [entry("Inter Miami CF"), entry("New York City FC")]}},
        {"name": "Western Conference", "standings": {"entries": [entry("LAFC"), entry("Austin FC")]}},
    ]}


def test_parse_maps_each_team_to_its_conference_through_the_resolver():
    got = c.parse(body(), resolve=lambda name, code: {"Inter Miami CF": "Inter Miami", "New York City FC": "New York City"}.get(name, name))
    assert got == {"Inter Miami": "Eastern Conference", "New York City": "Eastern Conference",
                   "LAFC": "Western Conference", "Austin FC": "Western Conference"}


def test_parse_ignores_a_payload_without_conferences():
    assert c.parse({}, resolve=lambda n, code: n) == {}
    assert c.parse({"children": [{"name": "X"}]}, resolve=lambda n, code: n) == {}


TABLE = [  # already ordered by points, as league.standings produces it
    {"team": "A", "points": 60}, {"team": "B", "points": 58}, {"team": "C", "points": 55}, {"team": "D", "points": 50},
]
CONF = {"A": "Western Conference", "B": "Eastern Conference", "C": "Western Conference", "D": "Eastern Conference"}


def test_positions_restart_in_each_conference_and_rows_are_grouped_east_first():
    rows = c.rank_in_conferences(TABLE, CONF, band=lambda pos: ("MLS Cup playoffs", "primary") if pos <= 1 else None)
    assert [(r["conference"], r["position"], r["team"]) for r in rows] == [
        ("Eastern Conference", 1, "B"), ("Eastern Conference", 2, "D"),
        ("Western Conference", 1, "A"), ("Western Conference", 2, "C")]


def test_playoff_band_uses_the_conference_rank_not_the_overall_rank():
    rows = c.rank_in_conferences(TABLE, CONF, band=lambda pos: ("MLS Cup playoffs", "primary") if pos <= 1 else None)
    by_team = {r["team"]: r for r in rows}
    assert by_team["B"]["qualification"] == "MLS Cup playoffs" and by_team["A"]["qualification"] == "MLS Cup playoffs"
    assert by_team["C"]["qualification"] is None and by_team["D"]["qualification"] is None


def test_a_team_without_a_known_conference_is_kept_unranked_at_the_end():
    rows = c.rank_in_conferences(TABLE + [{"team": "Z", "points": 1}], CONF, band=lambda pos: None)
    assert rows[-1]["team"] == "Z" and rows[-1]["conference"] is None and rows[-1]["position"] == 5
