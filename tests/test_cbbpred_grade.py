from cbbpred.evaluate import confidence, cover_side, total_side, vs_market
from cbbpred.grade import grade_row, track_record
import polars as pl


def call(**kw):
    base = {"game_id": "1", "home_score": 80, "away_score": 70, "spread_line": 7.0, "total_line": 150.0,
            "p_home_win": 0.7, "p_home_cover": 0.6, "p_over": 0.4, "proj_margin": 6.0, "proj_total": 145.0,
            "market_prob_home": 0.75}
    return {**base, **kw}


def test_cover_side_handles_push_and_none():
    assert cover_side(10, 7) == "home" and cover_side(3, 7) == "away"
    assert cover_side(7, 7) == "push" and cover_side(7, None) is None
    assert total_side(151, 150) == "over" and total_side(150, 150) == "push"


def test_grade_row_correct_picks():
    g = grade_row(call(home_score=75, away_score=65))                  # total 140 < 150
    assert g["winner_correct"] and g["market_winner_correct"]
    assert g["ats_result"] == "home" and g["ats_correct"] is True
    assert g["total_result"] == "under" and g["total_correct"] is True
    assert g["margin_error"] == 4 and g["total_error"] == 5


def test_push_is_not_a_win_or_a_loss():
    g = grade_row(call(home_score=77, away_score=70))                   # margin 7 == spread 7
    assert g["ats_result"] == "push" and g["ats_correct"] is None
    g = grade_row(call(home_score=80, away_score=70, total_line=150.0))  # total 150 == line
    assert g["total_result"] == "push" and g["total_correct"] is None


def test_missing_line_leaves_betting_results_empty():
    g = grade_row(call(spread_line=None, total_line=None, p_home_cover=None, p_over=None, market_prob_home=None))
    assert g["ats_correct"] is None and g["total_correct"] is None and g["market_winner_correct"] is None
    assert g["winner_correct"] is True


def test_market_winner_falls_back_to_spread_without_moneyline():
    assert grade_row(call(market_prob_home=None, spread_line=-3.0))["market_winner_correct"] is False


def test_track_record_excludes_ungraded_rows():
    df = pl.DataFrame([grade_row(call()), grade_row(call(p_home_win=0.4, p_home_cover=None, spread_line=None))])
    t = track_record(df)
    assert t["winner"]["n"] == 2 and t["winner"]["wins"] == 1
    assert t["ats"]["n"] == 1 and t["games"] == 2


def test_vs_market_counts_ats_and_skips_pushes():
    bt = pl.DataFrame({"season": [2026] * 3, "margin": [10.0, 7.0, -3.0], "total": [150.0] * 3,
                       "pred_margin": [9.0, 9.0, -9.0], "pred_total": [140.0] * 3,
                       "spread_line": [7.0, 7.0, 4.0], "total_line": [145.0] * 3})
    out = vs_market(bt)
    assert out["ats"]["n"] == 2 and out["ats"]["wins"] == 2     # the 7-point game is a push
    assert out["ou"]["n"] == 3 and out["ou"]["wins"] == 0       # model said under; all went over


def test_vs_market_is_none_without_lines():
    assert vs_market(pl.DataFrame({"season": [2026], "margin": [1.0]})) is None


def test_confidence_labels():
    assert confidence(0.9) == "high" and confidence(0.1) == "high"
    assert confidence(0.7) == "medium" and confidence(0.56) == "low" and confidence(0.5) == "coin_flip"
