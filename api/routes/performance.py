"""Public model evaluation and timestamp-verified saved game forecasts."""
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo
from fastapi import APIRouter
from api.data import League
from utils.league_config import get_config

router = APIRouter(tags=['performance'])
REPORT_DIR = Path(__file__).resolve().parents[2] / 'docs' / 'game-models'


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _stamp(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return stamp if stamp.tzinfo else None
    except (ValueError, TypeError):
        return None


def _tally(games, key):
    picked = [g[key] for g in games if g.get(key) is not None]
    return dict(n=len(picked), correct=sum(picked), rate=sum(picked) / len(picked) if picked else None)


def _summary(games):
    margins = [g['margin_error'] for g in games if g['margin_error'] is not None]
    totals = [g['total_error'] for g in games if g['total_error'] is not None]
    return dict(games=len(games), winner=_tally(games, 'winner_correct'),
                spread=_tally(games, 'spread_correct'), total_ou=_tally(games, 'total_correct'),
                margin_mae=sum(margins) / len(margins) if margins else None,
                total_mae=sum(totals) / len(totals) if totals else None)


def _against(value, line, pick, sides):
    """Did the pick beat the saved sportsbook line? None for no pick, no line or a push."""
    if pick not in sides or line is None:
        return None
    return None if value == line else (value > line) == (pick == sides[0])


def _live_history(league):
    raw = _read(get_config(league).data_dir / 'prediction_history.json')
    games, excluded = [], 0
    for day, record in raw.items() if isinstance(raw, dict) else []:
        if not isinstance(record, dict):
            continue  # WNBA legacy prop records are not game forecasts.
        generated = _stamp(record.get('generated_at'))
        for game in record.get('games') or []:
            tip = _stamp(game.get('game_time'))
            actual = game.get('actuals') or {}
            hs, aws = _number(actual.get('home_score')), _number(actual.get('away_score'))
            if hs is None or aws is None:
                continue
            if not generated or not tip or generated >= tip:
                excluded += 1
                continue
            prediction = game.get('predictions') or {}
            ph, pa = _number(prediction.get('model_home_score')), _number(prediction.get('model_away_score'))
            pt = _number(prediction.get('model_total'))
            winner = prediction.get('winner_pick')
            home, away = game.get('home'), game.get('away')
            season = int(day[:4]) - int(league == 'nba' and int(day[5:7]) < 7)
            games.append(dict(date=day, season=season, game_id=game.get('game_id'),
                home_team=home, away_team=away, home_score=hs, away_score=aws,
                predicted_home_score=ph, predicted_away_score=pa, predicted_winner=winner,
                winner_correct=(winner == (home if hs > aws else away)) if winner in (home, away) and hs != aws else None,
                margin_error=abs((ph-pa)-(hs-aws)) if ph is not None and pa is not None else None,
                total_error=abs(pt-hs-aws) if pt is not None else None,
                generated_at=generated.isoformat(), tipoff=tip.isoformat()))
    now = datetime.now(ZoneInfo('America/New_York'))
    current = now.year - int(league == 'nba' and now.month < 7)
    seasons = []
    for year in sorted({current} | {g['season'] for g in games}, reverse=True):
        selected = sorted([g for g in games if g['season'] == year], key=lambda g: (g['date'],str(g['game_id'])), reverse=True)
        months = sorted({g['date'][:7] for g in selected}, reverse=True)
        seasons.append(dict(season=year, record=_summary(selected), games=selected,
            by_month=[dict(month=m, **_summary([g for g in selected if g['date'].startswith(m)])) for m in months]))
    return dict(current_season=current, seasons=seasons, excluded_unverified=excluded,
        message='Saved forecasts count only when their timestamp is before the recorded tipoff. Legacy forecasts may use earlier model versions; they are separate from the fixed holdout evaluation.')


@router.get('/performance')
def get_performance(league: League = 'nba'):
    report = _read(REPORT_DIR / f'{league}-evaluation.json')
    evaluation = report if isinstance(report, dict) and report.get('holdout') else None
    ledger = _read(get_config(league).data_dir / 'game_forecast_ledger.json')
    version = evaluation.get('version') if evaluation else None
    # "v1+market-prior" is the live, market-assisted variant of the evaluated model "v1": same family, same record.
    family = [entry for entry in ledger.get('forecasts', []) if str(entry.get('model_version') or '').split('+')[0] == version] if isinstance(ledger, dict) and version else []
    entries = [entry for entry in family if not entry.get('void')]  # preseason/cancelled games can never be graded
    graded = []
    for entry in entries:
        actual = entry.get('actual') or {}
        hs, aws = _number(actual.get('home_score')), _number(actual.get('away_score'))
        generated, tip = _stamp(entry.get('generated_at')), _stamp(entry.get('tipoff'))
        if hs is None or aws is None or not generated or not tip or generated >= tip:
            continue
        prediction, market = entry.get('prediction') or {}, entry.get('market') or {}
        ph, pa = _number(prediction.get('predicted_home_score')), _number(prediction.get('predicted_away_score'))
        pt = _number(prediction.get('total'))
        winner = prediction.get('predicted_winner')
        spread, line_total = _number(market.get('spread')), _number(market.get('total'))  # spread is the home team's line
        graded.append(dict(date=tip.astimezone(ZoneInfo('America/New_York')).date().isoformat(), game_id=entry.get('game_id'),
            home_team=entry['home_team'], away_team=entry['away_team'], home_score=hs, away_score=aws,
            predicted_home_score=ph, predicted_away_score=pa, predicted_total=pt, predicted_winner=winner,
            winner_correct=winner == (entry['home_team'] if hs > aws else entry['away_team'])
            if winner in (entry.get('home_team'), entry.get('away_team')) and hs != aws else None,
            margin_error=abs((ph-pa)-(hs-aws)) if ph is not None and pa is not None else None,
            total_error=abs(pt-hs-aws) if pt is not None else None,
            market_spread=spread, spread_pick=prediction.get('spread_pick'),
            spread_correct=_against(hs - aws, -spread if spread is not None else None, prediction.get('spread_pick'), ('HOME', 'AWAY')),
            market_total=line_total, total_pick=prediction.get('total_pick'),
            total_correct=_against(hs + aws, line_total, prediction.get('total_pick'), ('OVER', 'UNDER')),
            model_version=entry.get('model_version')))
    graded.sort(key=lambda g: (g['date'], str(g['game_id'])), reverse=True)
    prospective = dict(model_version=version, saved=len(entries), pending=sum(not entry.get('actual') for entry in entries),
                       void=len(family) - len(entries), record=_summary(graded), games=graded,
                       message='First-published forecasts for this model version, saved before tipoff with the sportsbook lines of that moment. Results are separate from legacy saved calls and fixed holdout metrics.')
    return dict(league=league, prospective=prospective, checked_at=datetime.now(ZoneInfo('America/New_York')).isoformat(),
                evaluation=evaluation, live=_live_history(league),
                market_record=dict(available=False, message='No historical sportsbook quote archive: spread, over/under profitability and closing-line value have not been evaluated.'))
