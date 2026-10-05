"""Build a complete public snapshot; failed exports leave the previous snapshot intact.

Personal journal data is deliberately never exported.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

def fetch(base, path):
    with urllib.request.urlopen(base + path, timeout=120) as response:
        return json.load(response)


def save(root, name, data):
    path = root / (name + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))


def export(base, out):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fetch(base, '/api/health')
    with tempfile.TemporaryDirectory(prefix='nba-export-', dir=out.parent) as directory:
        stage = Path(directory) / 'data'
        stage.mkdir()
        for league in ('nba', 'wnba'):
            root = stage / league
            save(root, 'props', fetch(base, f'/api/props?league={league}&direction=all&include_research=true&limit=500'))
            save(root, 'alt-lines', fetch(base, f'/api/props/alt-lines?league={league}'))
            save(root, 'games', fetch(base, f'/api/games?league={league}'))
            save(root, 'predictions', fetch(base, f'/api/games/predictions?league={league}'))
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
            'schema_version': 2,
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
    args = parser.parse_args()
    print(f'Export complete: {export(args.api.rstrip("/"), args.out)}')


if __name__ == '__main__':
    main()
