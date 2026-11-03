"""Why the model picked what it picked, in the numbers it actually used.

These are **not** commentary. Every sentence quotes a feature that went into the
forecast — a rolling average, an Elo gap, a venue record — so a reader can check
the reasoning against the same evidence the model saw, and spot when a
confident-looking number rests on four matches of form.

Nothing here changes a probability. If the explanation and the forecast ever
disagree, the forecast is what was served and the explanation is the bug.
"""
from __future__ import annotations

from .. import config


def _name(team: str) -> str:
    return team


def _fmt(value: float | None, places: int = 1) -> str | None:
    return None if value is None else f"{value:.{places}f}"


def _mean(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _competition(code: str) -> str:
    return config.LEAGUES.get(code, {}).get("name", code)


def result(prediction: dict, row: dict) -> str | None:
    """Why this side is favoured."""
    home, away = prediction["home_team"], prediction["away_team"]
    pick = prediction["result_pick"]
    win = prediction["win"]
    elo_h, elo_a = row.get("home_elo"), row.get("away_elo")
    ppg_h, ppg_a = row.get("home_ppg_r5"), row.get("away_ppg_r5")

    parts: list[str] = []
    if pick == "draw":
        opening = (
            f"No side is favoured — the draw is the single likeliest result at "
            f"{win['draw'] * 100:.0f}%, ahead of {home} at {win['home'] * 100:.0f}% "
            f"and {away} at {win['away'] * 100:.0f}%."
        )
    else:
        side, other = (home, away) if pick == "home" else (away, home)
        opening = (
            f"{side} are favoured at {win[pick] * 100:.0f}%, against "
            f"{win['away' if pick == 'home' else 'home'] * 100:.0f}% for {other} "
            f"and {win['draw'] * 100:.0f}% for the draw."
        )
    parts.append(opening)

    if elo_h is not None and elo_a is not None:
        gap = elo_h - elo_a
        if abs(gap) < 25:
            parts.append(f"Elo separates them by only {abs(gap):.0f} points, so the edge is small.")
        else:
            stronger = home if gap > 0 else away
            parts.append(f"{stronger} carry {abs(gap):.0f} more Elo points into the match.")

    if ppg_h is not None and ppg_a is not None:
        parts.append(
            f"Over the last five: {home} {ppg_h:.1f} points per game, {away} {ppg_a:.1f}."
        )

    if pick != "draw":
        venue = row.get("home_venue_ppg") if pick == "home" else row.get("away_venue_ppg")
        if venue is not None:
            where = "at home" if pick == "home" else "on the road"
            parts.append(f"They average {venue:.1f} points per game {where} this season.")

    return " ".join(parts)


def goals(prediction: dict, row: dict) -> str | None:
    """Why the total lands where it does."""
    home, away = prediction["home_team"], prediction["away_team"]
    total = prediction["expected_goals"]["total"]
    over = prediction["total_goals"]["over_2_5"]
    side = "over" if over >= 0.5 else "under"
    confidence = max(over, 1 - over)

    attack = _mean(row.get("home_gf_r5"), row.get("away_gf_r5"))
    concede = _mean(row.get("home_ga_r5"), row.get("away_ga_r5"))

    parts = [
        f"The model expects {total:.2f} goals, which puts it {side} 2.5 at "
        f"{confidence * 100:.0f}%."
    ]
    if attack is not None and concede is not None:
        parts.append(
            f"Across their last five, {home} and {away} have scored {attack:.1f} "
            f"and conceded {concede:.1f} a game between them."
        )
    hg, ag = row.get("home_gf_r5"), row.get("away_gf_r5")
    if hg is not None and ag is not None and abs(hg - ag) >= 0.7:
        sharper, blunter = (home, away) if hg > ag else (away, home)
        parts.append(f"{sharper} are the sharper of the two in front of goal right now, not {blunter}.")
    return " ".join(parts)


def counts(prediction: dict, row: dict, market: str) -> str | None:
    """Why the corner or shots-on-target lines sit where they do."""
    block = prediction.get(market)
    if not block or not block.get("available"):
        return None

    home, away = prediction["home_team"], prediction["away_team"]
    noun = "corners" if market == "corners" else "shots on target"
    stat = "corners" if market == "corners" else "sot"
    h5, a5 = row.get(f"home_{stat}_r5"), row.get(f"away_{stat}_r5")
    h10, a10 = row.get(f"home_{stat}_r10"), row.get(f"away_{stat}_r10")

    line = block["total"]["line"]
    over = block["total"]["over"]
    side = "over" if over >= 0.5 else "under"

    parts = [
        f"{block['home']['expected']:.1f} {noun} for {home} and "
        f"{block['away']['expected']:.1f} for {away} — {block['total']['expected']:.1f} "
        f"between them, which is {side} {line} at {max(over, 1 - over) * 100:.0f}%."
    ]

    averages = []
    if h5 is not None:
        averages.append(f"{home} average {h5:.1f} over their last five"
                        + (f" and {h10:.1f} over ten" if h10 is not None else ""))
    if a5 is not None:
        averages.append(f"{away} {a5:.1f} over five"
                        + (f" and {a10:.1f} over ten" if a10 is not None else ""))
    if averages:
        # Only the leading character: `str.capitalize` would lower-case the
        # club names further along the sentence.
        joined = "; ".join(averages)
        parts.append(joined[:1].upper() + joined[1:] + ".")

    both = block.get("both_teams")
    if both and both.get("threshold") is not None:
        parts.append(
            f"Both reaching {both['threshold']}+ comes out at "
            f"{both['probability'] * 100:.0f}%."
        )
    return " ".join(parts)


def calibration(prediction: dict) -> str | None:
    """Say plainly when a rate was shifted by the model's own past misses."""
    applied = [
        c for c in (prediction.get("calibration") or {}).values()
        if c and abs(c["factor"] - 1.0) >= 0.005
    ]
    if not applied:
        return None
    said = []
    for c in sorted(applied, key=lambda c: c["market"]):
        noun = {"goals": "goals", "corners": "corners", "sot": "shots on target"}[c["market"]]
        direction = "over" if c["factor"] < 1 else "under"
        said.append(
            f"{noun} were predicted {c['predicted_mean']:.2f} against {c['actual_mean']:.2f} "
            f"actual across {c['n']} settled forecasts, so the rate is nudged "
            f"{'down' if c['factor'] < 1 else 'up'} {abs(1 - c['factor']) * 100:.1f}% "
            f"to stop the model leaning {direction}"
        )
    return (
        "Adjusted from this competition's own record: "
        + "; ".join(said)
        + "."
    )


def limits(prediction: dict, row: dict) -> list[str]:
    """What the reader should not read into this."""
    out: list[str] = []
    context = prediction.get("context") or {}
    fewest = min(
        context.get("home_season_matches") or 0,
        context.get("away_season_matches") or 0,
    )
    if fewest < 5:
        out.append(
            f"Only {fewest} league match{'' if fewest == 1 else 'es'} of this season "
            "behind the weaker-evidenced side, so form leans on last season."
        )
    for market, noun in (("corners", "Corner"), ("sot", "Shots-on-target")):
        block = prediction.get(market)
        if block and not block.get("available"):
            out.append(f"{noun} lines are unavailable: {block.get('note')}")
    if not (prediction.get("fixture") or {}).get("odds", {}).get("home"):
        out.append("No closing odds for this fixture, so the forecast is form-only.")
    return out


def build(prediction: dict, row: dict) -> list[dict]:
    """The explanation panel: one entry per market the model committed to."""
    entries = [
        {"market": "result", "label": "Match result", "text": result(prediction, row)},
        {"market": "goals", "label": "Goals", "text": goals(prediction, row)},
        {"market": "corners", "label": "Corners", "text": counts(prediction, row, "corners")},
        {"market": "sot", "label": "Shots on target", "text": counts(prediction, row, "sot")},
        {"market": "calibration", "label": "Self-correction", "text": calibration(prediction)},
    ]
    return [e for e in entries if e["text"]]
