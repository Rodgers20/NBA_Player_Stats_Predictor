"""Walk-forward backtest of the player prop probabilities on this season's ESPN box scores.

For every match day after a burn-in, every player the model would have priced (eligible from
matches strictly before that day) is scored against what happened — a player who did not play
counts as a miss, so the test is not flattered by pricing only the players who turned up.
Team expectations come from the simple opponent-concession proxy (no match model exists for
past dates); production swaps in the match model's expected goals / shots on target.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

from . import forms as fm
from . import model, price

EPS = 1e-4
EDGES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 1.0)
MIN_VERDICT_N = 200


def outcome(market: str, line: float | None, stats: dict | None) -> int:
    """1 if the prop hit. `stats` is the player's line in that match, or None if he did not play."""
    if stats is None:
        return 0
    if market == "anytime_goal":
        return int(stats["goals"] >= 1)
    if market == "assist":
        return int(stats["assists"] >= 1)
    return int(stats["sot"] > (line or 0.0))


def log_loss(pairs: Iterable[tuple[float, int]]) -> float:
    pairs = list(pairs)
    total = sum(-(y * math.log(min(1 - EPS, max(EPS, p))) + (1 - y) * math.log(min(1 - EPS, max(EPS, 1 - p)))) for p, y in pairs)
    return total / len(pairs) if pairs else float("nan")


def baseline_log_loss(pairs: Sequence[tuple[float, int]]) -> float:
    """Log loss of always predicting the observed hit rate: the hardest constant baseline (in-sample)."""
    rate = sum(y for _, y in pairs) / len(pairs)
    return log_loss([(rate, y) for _, y in pairs])


def calibration(pairs: Sequence[tuple[float, int]], edges: Sequence[float] = EDGES) -> list[dict]:
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        inside = [(p, y) for p, y in pairs if lo <= p < hi or (hi == edges[-1] and p == hi)]
        rows.append({"bin": f"{round(lo * 100)}-{round(hi * 100)}%", "n": len(inside),
                     "mean_pred": sum(p for p, _ in inside) / len(inside) if inside else None,
                     "hit_rate": sum(y for _, y in inside) / len(inside) if inside else None})
    return rows


def verdict(model_ll: float, base_ll: float, n: int) -> str:
    if n < MIN_VERDICT_N:
        return f"too few predictions ({n}) for a verdict"
    gain = (base_ll - model_ll) / base_ll * 100
    if model_ll < base_ll:
        return f"beats the base-rate baseline by {gain:.1f}% log loss"
    return f"does not beat the base-rate baseline ({gain:+.1f}% log loss)"


def _league_avgs(aggs: dict[str, fm.TeamAggregate]) -> dict[str, float]:
    games = sum(a.games for a in aggs.values()) or 1
    return {"goals": sum(a.goals_for for a in aggs.values()) / games, "sot": sum(a.sot_for for a in aggs.values()) / games}


_STATE: dict[tuple[int, str], tuple] = {}


def _day_state(rows: list[dict], day: str) -> tuple:
    """Forms, team aggregates, priors and league averages as of `day`; cached per (rows object, day) for the tuning grid."""
    key = (id(rows), day)
    if key not in _STATE:
        forms, aggs = fm.build_forms(rows, before=day), fm.team_aggregates(rows, before=day)
        _STATE[key] = (forms, aggs, price.position_priors(forms), _league_avgs(aggs))
    return _STATE[key]


def predictions(rows: list[dict], eval_from: str, dispersion: dict[str, float] | None = None,
                markets: tuple[str, ...] = price.MARKETS, strength: dict[str, float] | None = None) -> list[dict]:
    """Every priced (player, match, market) from `eval_from` on, with its realised outcome."""
    events: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))  # date -> event -> rows
    for r in rows:
        events[r["date"]][r["event_id"]].append(r)
    out: list[dict] = []
    for day in sorted(d for d in events if d >= eval_from):
        forms, aggs, priors, league = _day_state(rows, day)
        for event, event_rows in events[day].items():
            home = next((r["team"] for r in event_rows if r["is_home"]), None)
            away = next((r["team"] for r in event_rows if not r["is_home"]), None)
            if not home or not away or home not in aggs or away not in aggs:
                continue
            xg = {"home": price.proxy_expected(aggs[home], aggs[away], league["goals"], "goals"),
                  "away": price.proxy_expected(aggs[away], aggs[home], league["goals"], "goals")}
            actual = {r["espn_id"]: r for r in event_rows}
            priced = price.price_fixture(forms, aggs, priors, league, home, away, xg, None, markets, dispersion, strength)
            for p in priced:
                out.append({"date": day, "event_id": event, "competition_code": event_rows[0]["competition_code"],
                            "market": p["market"], "line": p["line"], "prob": p["prob"], "games": p["games"],
                            "outcome": outcome(p["market"], p["line"], actual.get(p["espn_id"]))})
    return out


EVAL_FROM = "2026-09-20"   # burn-in: every European side has played ~5 rounds by here
TUNE_UNTIL = "2026-10-10"  # dispersion is tuned on [EVAL_FROM, TUNE_UNTIL) and reported on the rest
PHI_GRID = (0.0, 0.15, 0.3, 0.5)
STRENGTH_GRID = {"goals": (6.0, 10.0, 15.0, 22.0), "assists": (8.0, 12.0, 18.0, 28.0), "sot": (2.0, 3.0, 5.0, 8.0)}
MARKET_OF = {"goals": "anytime_goal", "assists": "assist", "sot": "shots_on_target"}


def _tuning_loss(rows: list[dict], key: str, dispersion: dict[str, float], strength: dict[str, float]) -> float:
    window = [p for p in predictions(rows, EVAL_FROM, dispersion, (MARKET_OF[key],), strength) if p["date"] < TUNE_UNTIL]
    return log_loss([(p["prob"], p["outcome"]) for p in window])


def tune(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Choose each market's prior strength (and the SOT dispersion) by log loss on the tuning window only."""
    tried: list[dict] = []
    strength: dict[str, float] = {}
    for key, grid in STRENGTH_GRID.items():
        scored = [(_tuning_loss(rows, key, {}, {key: s}), s) for s in grid]
        tried += [{"param": f"strength.{key}", "value": s, "log_loss": ll} for ll, s in scored]
        strength[key] = min(scored)[1]
    phis = [(_tuning_loss(rows, "sot", {"sot": phi}, strength), phi) for phi in PHI_GRID]
    tried += [{"param": "dispersion.sot", "value": phi, "log_loss": ll} for ll, phi in phis]
    return {"strength": strength, "dispersion": {"sot": min(phis)[1]}}, tried


def top_gap(calibration_rows: list[dict], floor: float = 0.20, min_n: int = 50) -> tuple[float, int] | None:
    """Mean (predicted - actual) in points over the bins whose prediction is >= `floor`, pooled; None if too few."""
    rows = [c for c in calibration_rows if c["n"] and c["mean_pred"] is not None and c["mean_pred"] >= floor]
    n = sum(c["n"] for c in rows)
    if n < min_n:
        return None
    return sum((c["mean_pred"] - c["hit_rate"]) * c["n"] for c in rows) / n * 100, n


def _verdict_lines(summary: dict) -> list[str]:
    lines = ["## Verdict", ""]
    for (market, line), s in summary.items():
        name = market if line is None else f"{market} over {line}"
        gap = top_gap(s["calibration"])
        tail = (f" Above 20% it over-predicts by {gap[0]:.1f} points on average (n={gap[1]}), so treat the biggest numbers as "
                f"optimistic." if gap and gap[0] > 2.0 else
                f" Above 20% it is within {abs(gap[0]):.1f} points of reality (n={gap[1]})." if gap else
                " Too few high-probability predictions to judge the top end.")
        lines.append(f"- **{name}**: {s['verdict']}; on average it predicts {_fmt(s['mean_pred'])} against a real {_fmt(s['hit_rate'])}.{tail}")
    lines += ["", "None of this is a betting edge claim: the baseline is a constant rate, not a bookmaker's price. "
              "Rankings within a match are more trustworthy than absolute percentages.", ""]
    return lines


def _fmt(x: float | None, pct: bool = True) -> str:
    return "–" if x is None else (f"{x * 100:.1f}%" if pct else f"{x:.4f}")


def render_markdown(summary: dict, tuned: dict, tried: list[dict], meta: dict) -> str:
    lines = [
        "# Player props model: method and backtest", "",
        "Auto-generated by `python -m futball.props.backtest`. Inputs: ESPN box scores for the 2026/27 season in progress "
        f"({meta['matches']} finished matches, {meta['players']} players; latest {meta['last']}).", "",
        "## Method", "",
        "- **Rate**: goals / assists / shots on target per full-game equivalent (a start = 1, a substitute appearance = 0.3, because "
        "free box scores carry no minutes), shrunk toward the position average with prior strengths "
        "6 games (goals), 8 (assists), 3 (SOT).",
        "- **Exposure**: P(start) = 60% last five team matches + 40% season share, x0.3 if absent for 3+ team matches; "
        "non-starters add a substitute chance x0.3. Goalkeepers and players under 2 appearances / P(start) < 25% get no row.",
        "- **Team factor**: the player's rate is scaled by (this match's expected team goals or SOT) / (the team's shrunk season average). "
        "Production uses the match model's expected goals and SOT; the backtest uses an opponent-concession proxy because no "
        "match model exists for past dates.",
        "- **Probability**: Poisson for goals and assists; negative binomial for SOT with the dispersion tuned below.",
        "- **Not modelled**: penalty-taker status, set-piece roles, injuries/suspensions beyond the absence heuristic, minutes, tackles "
        "(no free source has them).", "",
        "## Backtest", "",
        f"Walk-forward: each match day from {EVAL_FROM} is priced using only earlier matches. Candidates are every previously seen "
        "player of the two teams who passes the gate; a player who did not play counts as a miss. "
        f"SOT dispersion tuned on {EVAL_FROM} to {TUNE_UNTIL}; **all numbers below are on matches from {TUNE_UNTIL} on**.", "",
        "Tuning search (log loss on the tuning window, one parameter at a time):", "",
        "| Parameter | Value | Tuning log loss |", "|---|---|---|",
        *[f"| {t['param']} | {t['value']} | {t['log_loss']:.4f} |" for t in tried], "",
        f"Chosen: prior strength {tuned['strength']}, SOT dispersion {tuned['dispersion']['sot']}. Saved to `futball/props/params.json`.", "",
        "| Market | Line | n | Hit rate | Mean predicted | Log loss | Baseline log loss | Brier | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (market, line), s in summary.items():
        lines.append(f"| {market} | {line if line is not None else '–'} | {s['n']} | {_fmt(s['hit_rate'])} | {_fmt(s['mean_pred'])} | "
                     f"{_fmt(s['log_loss'], False)} | {_fmt(s['baseline_log_loss'], False)} | {_fmt(s['brier'], False)} | {s['verdict']} |")
    lines += ["", "Baseline = always predict the observed hit rate (in-sample, so it is a deliberately hard bar).", ""]
    lines += _verdict_lines(summary)
    lines += ["## Calibration (predicted vs actual)", ""]
    for (market, line), s in summary.items():
        lines += [f"**{market}{'' if line is None else ' over ' + str(line)}**", "", "| Predicted bin | n | Mean predicted | Actual hit rate |", "|---|---|---|---|"]
        lines += [f"| {c['bin']} | {c['n']} | {_fmt(c['mean_pred'])} | {_fmt(c['hit_rate'])} |" for c in s["calibration"]]
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    from pathlib import Path

    from .. import boxscores

    stats = boxscores.load()
    rows = stats.to_dicts()
    tuned, tried = tune(rows)
    preds = [p for p in predictions(rows, TUNE_UNTIL, tuned["dispersion"], strength=tuned["strength"]) if p["date"] >= TUNE_UNTIL]
    summary = summarise(preds)
    meta = {"matches": stats["event_id"].n_unique(), "players": stats["espn_id"].n_unique(), "last": stats["date"].max()}
    out = Path(__file__).resolve().parents[2] / "docs" / "props-model.md"
    out.write_text(render_markdown(summary, tuned, tried, meta), encoding="utf-8")
    model.save_params(tuned)
    for (market, line), s in summary.items():
        print(f"{market:16s} {line!s:5s} n={s['n']:6d} hit={s['hit_rate']:.3f} pred={s['mean_pred']:.3f} "
              f"ll={s['log_loss']:.4f} base={s['baseline_log_loss']:.4f}  {s['verdict']}")
    print("wrote", out)
    return 0


def summarise(preds: list[dict]) -> dict:
    """Per (market, line): n, hit rate, model vs baseline log loss, Brier, calibration table, verdict."""
    groups: dict[tuple, list[tuple[float, int]]] = defaultdict(list)
    for p in preds:
        groups[(p["market"], p["line"])].append((p["prob"], p["outcome"]))
    out = {}
    for key, pairs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        m_ll, b_ll = log_loss(pairs), baseline_log_loss(pairs)
        out[key] = {"n": len(pairs), "hit_rate": sum(y for _, y in pairs) / len(pairs), "mean_pred": sum(p for p, _ in pairs) / len(pairs),
                    "log_loss": m_ll, "baseline_log_loss": b_ll, "brier": sum((p - y) ** 2 for p, y in pairs) / len(pairs),
                    "calibration": calibration(pairs), "verdict": verdict(m_ll, b_ll, len(pairs))}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
