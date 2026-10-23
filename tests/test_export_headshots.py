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
