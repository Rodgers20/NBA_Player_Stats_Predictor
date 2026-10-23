"""Immutable first-published pregame forecasts and free quote snapshots.

Separate from legacy self-calibration history: no retroactive forecast creation,
no overwrite after tipoff, and no mixing model versions in performance summaries.
"""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import os
import tempfile


def _utc(value):
    parsed=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Tipoff needs an explicit timezone')
    return parsed.astimezone(timezone.utc)


def record_forecasts(league, games, predictions, path=None, now=None):
    if league not in ('nba','wnba'):raise ValueError('Invalid league')
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError('now needs timezone')
    path=Path(path) if path else Path(__file__).resolve().parents[1]/'data'/league/'game_forecast_ledger.json'
    ledger=json.loads(path.read_text()) if path.exists() else {'version':1,'league':league,'forecasts':[]}
    by_match={p['matchup']:p for p in predictions}
    seen={(str(x['game_id']),x['model_version']) for x in ledger['forecasts']}
    added=0
    for game in games:
        prediction=by_match.get(game.get('matchup'))
        if not prediction or not game.get('game_id') or game.get('status_state','pre')!='pre':continue
        try:tip=_utc(game.get('start_time') or game.get('game_time'))
        except (ValueError,TypeError):continue
        if tip<=now:continue
        metadata=prediction.get('model_metadata') or {}
        version=metadata.get('version') or 'historical-form-fallback'
        identity=(str(game['game_id']),version)
        if identity in seen:continue
        ledger['forecasts'].append(dict(game_id=identity[0],model_version=version,
            generated_at=now.isoformat(),tipoff=tip.isoformat(),home_team=game['home_team'],away_team=game['away_team'],
            prediction=prediction,market={k:game.get(k) for k in ('spread','total','home_ml','away_ml','odds_source','odds_updated_at')},
            actual=None))
        seen.add(identity);added+=1
    if added:
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
        try:
            with os.fdopen(fd,'w') as f:json.dump(ledger,f,indent=2,allow_nan=False)
            os.replace(temp,path)
        finally:
            if os.path.exists(temp):os.unlink(temp)
    return added


def grade_forecasts(league, team_games=None, path=None, now=None):
    """Attach completed official box-score outcomes to existing pregame records."""
    if league not in ('nba','wnba'):raise ValueError('Invalid league')
    now=now or datetime.now(timezone.utc)
    path=Path(path) if path else Path(__file__).resolve().parents[1]/'data'/league/'game_forecast_ledger.json'
    if not path.exists():return 0
    ledger=json.loads(path.read_text())
    if team_games is None:
        from utils.game_data import load_game_team_history
        team_games=load_game_team_history(league)
    # Forecasts carry the schedule provider's id (ESPN) and box scores the league's (NBA.com), so a game is
    # identified by its Eastern date and its two teams; the id only joins when both sides share a provider.
    pairs={str(g).zfill(10):pair for g,pair in team_games.groupby('game_id')}
    by_day={(pair.iloc[0].date.date(),frozenset(pair.team)):pair for pair in pairs.values() if len(pair)==2}
    covered=team_games.date.max().date() if len(team_games) else None
    graded=changed=0
    for entry in ledger['forecasts']:
        if entry.get('actual') or entry.get('void'):continue
        try:
            if not _utc(entry['generated_at']) < _utc(entry['tipoff']) < now:continue
        except (ValueError,KeyError):continue
        tipday=_utc(entry['tipoff']).astimezone(ZoneInfo('America/New_York')).date()
        pair=pairs.get(str(entry['game_id']).zfill(10))
        if pair is None or pair.iloc[0].date.date()!=tipday:  # guard id collisions with the advertised date
            pair=by_day.get((tipday,frozenset((entry['home_team'],entry['away_team']))))
        if pair is None:
            # Scores already run past this date without the game: an exhibition or a cancellation, never gradable.
            if covered and covered>tipday:
                entry['void']='no official regular-season or playoff result';changed+=1
            continue
        home=pair[pair.team==entry['home_team']];away=pair[pair.team==entry['away_team']]
        if len(home)!=1 or len(away)!=1:continue
        entry['actual']=dict(home_score=float(home.iloc[0].PTS),away_score=float(away.iloc[0].PTS),
                            graded_at=now.isoformat(),source='validated completed team box scores')
        graded+=1
    if graded or changed:
        fd,temp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
        try:
            with os.fdopen(fd,'w') as f:json.dump(ledger,f,indent=2,allow_nan=False)
            os.replace(temp,path)
        finally:
            if os.path.exists(temp):os.unlink(temp)
    return graded
