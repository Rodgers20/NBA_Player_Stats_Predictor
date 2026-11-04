from nhlpred.lines import _abbr, ev_per_unit, no_vig_home, parse_pickcenter

SUMMARY = {"pickcenter": [{"provider": {"name": "DraftKings"}, "overUnder": 6.5, "spread": 1.5, "overOdds": -142.0, "underOdds": 120.0,
                           "awayTeamOdds": {"moneyLine": -148, "spreadOdds": 160.0}, "homeTeamOdds": {"moneyLine": 124, "spreadOdds": -192.0}}]}


def test_parse_pickcenter_reads_home_and_away_prices():
    row = parse_pickcenter(SUMMARY, "t")
    assert (row["book"], row["home_ml"], row["away_ml"], row["total"], row["puck_home"]) == ("DraftKings", 124, -148, 6.5, 1.5)


def test_parse_pickcenter_without_moneyline_is_none():
    assert parse_pickcenter({"pickcenter": [{"overUnder": 6.5}]}, "t") is None
    assert parse_pickcenter({}, "t") is None


def test_no_vig_probability_sums_cleanly():
    p = no_vig_home(parse_pickcenter(SUMMARY, "t"))
    assert 0.40 < p < 0.45
    assert no_vig_home(None) is None


def test_ev_per_unit():
    assert abs(ev_per_unit(0.5, 100)) < 1e-9          # coin flip at even money
    assert ev_per_unit(0.6, 100) > 0 > ev_per_unit(0.4, 100)
    assert ev_per_unit(None, 100) is None


def test_espn_abbreviations_map_to_nhl_api():
    assert [_abbr(a) for a in ("UTAH", "SJ", "TB", "NJ", "LA", "TOR")] == ["UTA", "SJS", "TBL", "NJD", "LAK", "TOR"]
