import json

from scripts.export_api import localize_headshots


def test_headshots_are_served_from_the_snapshot(tmp_path):
    stage, previous = tmp_path / 'stage', tmp_path / 'previous'
    (stage / 'nba').mkdir(parents=True)
    (previous / 'headshots').mkdir(parents=True)
    (previous / 'headshots' / '1.png').write_bytes(b'old')
    props = {'props': [{'headshot_url': f'https://cdn.nba.com/headshots/nba/latest/1040x760/{i}.png'} for i in (1, 2, 3)]}
    (stage / 'nba' / 'props.json').write_text(json.dumps(props))

    localize_headshots(stage, previous, download=lambda person_id: b'new' if person_id == '2' else None)

    urls = [row['headshot_url'] for row in json.loads((stage / 'nba' / 'props.json').read_text())['props']]
    assert urls == ['/data/headshots/1.png', '/data/headshots/2.png',
                    'https://cdn.nba.com/headshots/nba/latest/1040x760/3.png']
    assert (stage / 'headshots' / '1.png').read_bytes() == b'old'
    assert (stage / 'headshots' / '2.png').read_bytes() == b'new'


def test_an_empty_refresh_never_replaces_a_board_for_the_same_slate(tmp_path):
    from scripts.export_api import keep_board_if_emptied
    previous, stage = tmp_path / 'previous', tmp_path / 'stage'
    for root in (previous, stage):
        root.mkdir()
    board = {'count': 2, 'status': 'ready', 'target_date': '2026-10-23', 'props': [{}, {}]}
    (previous / 'props.json').write_text(json.dumps(board))
    (stage / 'props.json').write_text(json.dumps({'count': 0, 'status': 'empty', 'target_date': '2026-10-23', 'props': []}))
    assert keep_board_if_emptied(stage, previous) is True
    assert json.loads((stage / 'props.json').read_text())['count'] == 2
    # A new slate date is never papered over with yesterday's picks.
    (stage / 'props.json').write_text(json.dumps({'count': 0, 'status': 'empty', 'target_date': '2026-10-24', 'props': []}))
    assert keep_board_if_emptied(stage, previous) is False
    assert json.loads((stage / 'props.json').read_text())['count'] == 0
