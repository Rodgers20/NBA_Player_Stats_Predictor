"""Build a complete public snapshot; failed exports leave the previous snapshot intact.

Personal journal, CLV and line-history data is deliberately never exported.
"""
import argparse
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import tempfile
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import urllib.parse
import urllib.request

def fetch(base, path, method='GET'):
    request = urllib.request.Request(base + path, method=method)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def save(root, name, data):
    path = root / (name + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))


def season_game_ids(root):
    """Completed regular-season games of the newest season that has results (what team pages show)."""
    for path in sorted(root.glob('teams-*.json'), reverse=True):
        teams = json.loads(path.read_text()).get('teams', [])
        ids = {g['id'] for team in teams for g in team.get('games', [])}
        if ids:
            return ids
    return set()


def export_boxscores(base, league, root, previous):
    """Finished games never change, so box scores from the previous snapshot are reused."""
    def one(game_id):
        old = previous / league / 'boxscore' / f'{game_id}.json'
        if old.exists():
            cached = json.loads(old.read_text())
            if cached.get('state') == 'post' and cached.get('available') and all(p.get('image') for t in cached['teams'] for p in t['players']):
                return game_id, cached
        try:
            return game_id, fetch(base, f'/api/games/{game_id}/boxscore?league={league}')
        except OSError:
            return game_id, None
    ids = sorted(season_game_ids(root))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, ids))
    saved = 0
    for game_id, data in results:
        if data and data.get('available'):
            path = root / 'boxscore' / f'{game_id}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
            saved += 1
    print(f'{league}: {saved}/{len(ids)} box scores exported')


def export_boxscores_only(base, out):
    """Adds box scores to an existing snapshot without regenerating anything else."""
    out = Path(out).resolve()
    fetch(base, '/api/health')
    for league in ('nba', 'wnba'):
        export_boxscores(base, league, out / league, out)
    manifest = out / 'manifest.json'
    if manifest.exists():
        files = [path for path in out.rglob('*') if path.is_file() and path != manifest]
        data = json.loads(manifest.read_text())
        data.update(snapshot_files=len(files) + 1, snapshot_bytes=sum(path.stat().st_size for path in files),
                    generated_at=datetime.now(timezone.utc).isoformat())
        manifest.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))


def export(base, out, refresh_odds=False):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fetch(base, '/api/health')
    with tempfile.TemporaryDirectory(prefix='nba-export-', dir=out.parent) as directory:
        stage = Path(directory) / 'data'
        stage.mkdir()
        for league in ('nba', 'wnba'):
            root = stage / league
            if refresh_odds:
                result = fetch(base, f'/api/props/refresh?league={league}&fetch_odds=true', method='POST')
                print(f'{league}: quote refresh evaluated {result.get("count", 0)} markets; {result.get("message", "")}')
            save(root, 'props', fetch(base, f'/api/props?league={league}&direction=all&include_research=true&limit=500'))
            save(root, 'defense', fetch(base, f'/api/defense?league={league}'))
            save(root, 'trends', fetch(base, f'/api/trends?league={league}&limit=400'))
            save(root, 'alt-lines', fetch(base, f'/api/props/alt-lines?league={league}'))
            games = fetch(base, f'/api/games?league={league}')
            save(root, 'games', games)
            slate_date = games.get('target_date') or datetime.now(ZoneInfo('America/New_York')).date().isoformat()
            save(root, 'scheduled-players', fetch(base, f'/api/players/scheduled?league={league}&game_date={slate_date}'))
            predictions = fetch(base, f'/api/games/predictions?league={league}')
            save(root, 'predictions', predictions)
            if predictions.get('target_date') == games.get('target_date'):
                from utils.game_forecast_ledger import record_forecasts
                count = record_forecasts(league, games.get('games', []), predictions.get('predictions', []))
                print(f'{league}: saved {count} first-published pregame forecasts')
            teams = fetch(base, f'/api/teams?league={league}')
            seasons = teams.get('seasons', [])[:3]
            teams['available_seasons'] = seasons
            save(root, 'teams', teams)
            for season in seasons:
                season_teams = teams if season == teams.get('season') else fetch(base, f'/api/teams?league={league}&season={season}')
                season_teams['available_seasons'] = seasons
                save(root, f'teams-{season}', season_teams)
            export_boxscores(base, league, root, out)
            save(root, 'highlights', fetch(base, f'/api/highlights?league={league}'))
            save(root, 'standings', fetch(base, f'/api/standings?league={league}'))
            from utils.game_forecast_ledger import grade_forecasts
            grade_forecasts(league)
            save(root, 'performance', fetch(base, f'/api/performance?league={league}'))
            today = datetime.now(ZoneInfo('America/New_York')).date()
            dates = sorted({(today + timedelta(days=offset)).isoformat() for offset in range(-2, 4)} | {slate_date})
            matchups = set()
            for day in dates:
                day_games = fetch(base, f'/api/games?league={league}&game_date={day}')
                save(root, f'schedule/{day}', day_games)
                for game in day_games.get('games', []):
                    pair = f'{game["away_team"]}-{game["home_team"]}'
                    if pair not in matchups:  # a pairing is date-independent, so one file serves every day it appears
                        matchups.add(pair)
                        save(root, f'matchup/{pair}', fetch(base, f'/api/games/matchup?league={league}&home={game["home_team"]}&away={game["away_team"]}'))
                    if game.get('game_id') and game.get('status_state') in ('in', 'post'):
                        save(root, f'boxscore/{game["game_id"]}', fetch(base, f'/api/games/{game["game_id"]}/boxscore?league={league}'))
                save(root, f'schedule/{day}-predictions', fetch(base, f'/api/games/predictions?league={league}&game_date={day}'))
            save(root, 'schedule-index', {'current_date': today.isoformat(), 'target_date': slate_date, 'dates': dates})
            if league == 'wnba':
                save(root, 'hitrates', fetch(base, '/api/wnba/hitrates'))
            players = fetch(base, f'/api/players?league={league}')
            save(root, 'players', players)
            for player in players['players']:
                key = player.encode('utf-8').hex()
                endpoint = '/api/player/' + urllib.parse.quote(player, safe='')
                save(root, f'player/{key}', {
                    'stats': fetch(base, endpoint + f'/stats?league={league}'),
                    'series': fetch(base, endpoint + f'/series?league={league}&games=200'),
                })
            print(f'{league}: exported {len(players["players"])} players')
        save(stage, 'props-record', fetch(base, '/api/props/record'))
        files = [path for path in stage.rglob('*') if path.is_file()]
        save(stage, 'manifest', {
            'schema_version': 3,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'leagues': ['nba', 'wnba'],
            'snapshot_files': len(files),
            'snapshot_bytes': sum(path.stat().st_size for path in files),
        })
        backup = Path(directory) / 'previous'
        if out.exists():
            out.rename(backup)
        try:
            stage.rename(out)
        except OSError:
            if backup.exists():
                backup.rename(out)
            raise
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='frontend/public/data')
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--refresh-odds', action='store_true', help='Buy quotes within the persistent API credit budget before exporting.')
    parser.add_argument('--boxscores-only', action='store_true', help='Only add game box scores to the existing snapshot.')
    args = parser.parse_args()
    if args.boxscores_only:
        export_boxscores_only(args.api.rstrip('/'), args.out)
        return
    print(f'Export complete: {export(args.api.rstrip("/"), args.out, refresh_odds=args.refresh_odds)}')


if __name__ == '__main__':
    main()
