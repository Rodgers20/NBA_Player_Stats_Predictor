from pathlib import Path

import pytest

from cbbpred import draft, players
from cbbpred.draft import DraftFeedError

ROW = ('<div class="mock-row" data-pos="SF"><div class="mock-row-pick-number">{rank}</div>'
       '<a href="/players/{slug}"><div class="mock-row-name">{name}</div>'
       '<div class="mock-row-school-position">SF | {school} </div></a>'
       '<div class="section height-weight"><div>6&#39;7&quot;</div><div>230 <span>lbs</span></div></div>'
       '<div class="section year-age desktop"><div>Freshman</div><div>19.7 yrs</div></div></div>')
INTL = ROW.replace('<div class="mock-row-pick-number">{rank}</div>',
                   '<div class="mock-row-pick-number">{rank}</div><img class="flag-border" alt="Slovenia" src="x"/>')


def board_html(n: int = 40) -> str:
    rows = [ROW.format(rank=i, slug=f"p{i}", name=f"Player {i}", school="Kansas") for i in range(1, n + 1)]
    rows[3] = INTL.format(rank=4, slug="p4", name="Stefan Joksimović", school="Baskonia")
    return "".join(rows) + "".join(rows)          # the real page lists everyone twice


def pick_row(n: int, owner: str, origin: str | None = None) -> str:
    trade = f'<div class="trade"><a class="disabled" href="/{origin}">X</a></div>' if origin else ""
    return (f'<tr><td class="pick-number"><span>{n}</span></td><td><div class="team-link"><a href="/{owner}">'
            f'</a></div>{trade}</td></tr>')


def order_html(teams: list[str]) -> str:
    rounds = []
    for r in range(2):
        rows = [pick_row(i + 1, teams[i], None) for i in range(30)]
        if r == 0:
            rows[1] = pick_row(2, "thunder", teams[1])          # team 2's first-round pick belongs to OKC
        rounds.append('<div class="round-title">Round</div><table>' + "".join(rows) + "</table>")
    return "".join(rounds)


def odds_html(balls_pct: list[float]) -> str:
    rows = []
    for i, c in enumerate(balls_pct):
        cell = '<td class="percent ineligible-cell"><span>x</span></td>' if c == 0 else f'<td class="percent shaded">{c}</td>'
        rows.append(f'<tr><td class="team"><a href="/t{i}"></a></td>{cell}<td class="percent"></td></tr>')
    return "<table>" + "".join(rows) + "</table>"


TEAMS = [f"t{i}" for i in range(30)]


def test_board_dedupes_and_flags_international():
    b = draft.parse_board(board_html())
    assert [r["rank"] for r in b] == list(range(1, 41))
    assert b[3]["intl"] == "Slovenia" and b[0]["intl"] is None
    assert b[0]["height_in"] == 79 and b[0]["weight"] == 230 and b[0]["age"] == 19.7


def test_board_that_stops_parsing_raises():
    with pytest.raises(DraftFeedError):
        draft.parse_board("<html>redesigned</html>")


def test_order_maps_origin_to_owner():
    o = draft.parse_order(order_html(TEAMS))
    assert o["t1"]["r1"] == "thunder" and o["t1"]["r2"] == "t1" and o["t0"]["r1"] == "t0"
    with pytest.raises(DraftFeedError):
        draft.parse_order(order_html(TEAMS[:29] + ["t0"]).replace('<div class="round-title">Round</div>', "", 1))


def test_odds_become_balls_and_barred_team_takes_neighbours_count():
    pct = [5.9, 5.9, 5.9, 0, 8.8] + [8.8] * 5 + [5.9] * 4 + [2.9, 2.9]
    o = draft.parse_odds(odds_html(pct))
    assert o["balls"][:5] == [2, 2, 2, 3, 3] and o["balls"][-1] == 1
    assert o["restricted"] == {"t3": 1}


def test_link_espn_needs_matching_school_when_names_collide():
    pool = [{"id": "1", "name": "Jordan Smith Jr", "team": "Arkansas"}, {"id": "2", "name": "Jordan Smith", "team": "Duke"}]
    board = [{"name": "Jordan Smith Jr.", "school": "Arkansas", "intl": None},
             {"name": "Stefan Joksimović", "school": "Baskonia", "intl": "Slovenia"}]
    linked = draft.link_espn(board, pool)
    assert [b["espn_id"] for b in linked] == ["1", None]


def test_merge_keeps_newest_identity_and_each_seasons_line():
    old = {"id": "9", "name": "A", "team": "Old U", "line": {"pts": 10.0}}
    new = {"id": "9", "name": "A", "team": "New U", "line": {"pts": 14.0}}
    (p,) = players.merge({2026: [old], 2027: [new]})
    assert p["team"] == "New U" and p["stats"] == {"2026": {"pts": 10.0}, "2027": {"pts": 14.0}}
