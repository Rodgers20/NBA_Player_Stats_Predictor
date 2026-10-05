import json
import pytest
from scripts import export_api


def test_failed_export_keeps_previous_snapshot(tmp_path, monkeypatch):
    out = tmp_path / 'data'
    out.mkdir()
    (out / 'old.json').write_text('original')
    def fail(base, path):
        if path == '/api/health': return {'status': 'ok'}
        raise RuntimeError('unavailable')
    monkeypatch.setattr(export_api, 'fetch', fail)
    with pytest.raises(RuntimeError): export_api.export('http://test', out)
    assert (out / 'old.json').read_text() == 'original'


def test_export_writes_empty_lists_and_collision_safe_player_paths(tmp_path, monkeypatch):
    calls = []
    def fetch(base, path):
        calls.append(path)
        if path.startswith('/api/players?'): return {'players': ['A’ja Wilson']}
        if '/series?' in path: return {'games': []}
        if path.startswith('/api/props?'): return {'props': []}
        return {}
    monkeypatch.setattr(export_api, 'fetch', fetch)
    out = export_api.export('http://test', tmp_path / 'data')
    assert json.loads((out / 'nba/props.json').read_text()) == {'props': []}
    player_file = out / 'wnba/player' / ('A’ja Wilson'.encode().hex() + '.json')
    assert json.loads(player_file.read_text()) == {'stats': {}, 'series': {'games': []}}
    assert not any('/chart-data?' in path for path in calls)
    assert sum('/series?' in path for path in calls) == 2
    assert not any('journal' in path for path in calls)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['generated_at']
    assert manifest['schema_version'] == 2
    files = [path for path in out.rglob('*') if path.is_file() and path.name != 'manifest.json']
    assert manifest['snapshot_files'] == len(files)
    assert manifest['snapshot_bytes'] == sum(path.stat().st_size for path in files)


def test_compact_series_matches_chart_values_and_nulls(monkeypatch):
    import pandas as pd
    from api.routes import players
    frame = pd.DataFrame([
        {'_date': pd.Timestamp('2026-09-28'), 'MATCHUP': 'NYL vs. LVA', 'SEASON': '2026',
         'PTS': 20, 'REB': 8, 'AST': 4, 'FG3M': 2, 'STL': 1, 'BLK': 0},
        {'_date': pd.Timestamp('2026-09-30'), 'MATCHUP': 'NYL @ LVA', 'SEASON': '2026',
         'PTS': None, 'REB': 9, 'AST': 5, 'FG3M': 1, 'STL': 0, 'BLK': 2},
    ])
    monkeypatch.setattr(players, '_player', lambda name, league: frame)
    series = players.get_player_series('Test', games=200, league='wnba')['games']
    chart = players.get_player_chart_data('Test', stat='PTS+REB', games=200, league='wnba', line=None)['games']
    assert [row['date'] for row in series] == [row['date'] for row in chart]
    assert [row['is_home'] for row in series] == [True, False]
    assert [sum((row['pts'], row['reb'])) if row['pts'] is not None else None for row in series] == [row['value'] for row in chart]
