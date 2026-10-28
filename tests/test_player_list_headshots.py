import pandas as pd

from api.routes import players


def test_player_list_carries_each_players_latest_headshot(monkeypatch):
    frame = pd.DataFrame({'PLAYER_NAME': ['Ann Lee', 'Ann Lee', 'Bo Ray', 'No Id'], 'Player_ID': [1, 2, 3, None]})
    monkeypatch.setattr(players, 'history', lambda league: frame)
    result = players.list_players(None, 'nba')
    assert result['players'] == ['Ann Lee', 'Bo Ray', 'No Id']
    assert result['headshots']['Ann Lee'].endswith('/2.png')  # the most recent id wins
    assert set(result['headshots']) == {'Ann Lee', 'Bo Ray'}
    assert set(players.list_players('bo', 'nba')['headshots']) == {'Bo Ray'}
