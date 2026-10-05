import pandas as pd
from scripts import refresh_wnba_history
from scripts.refresh_wnba_history import merge_history


def test_refresh_preserves_prior_seasons_and_accepts_corrected_box_score():
    old=pd.DataFrame([dict(PLAYER_NAME='A',GAME_DATE='May 01, 2025',PTS=10,SEASON='2025'),
                      dict(PLAYER_NAME='A',GAME_DATE='May 01, 2026',PTS=11,SEASON='2026')])
    incoming=pd.DataFrame([dict(PLAYER_NAME='A',GAME_DATE='May 01, 2026',PTS=12,SEASON='2026'),
                           dict(PLAYER_NAME='A',GAME_DATE='May 02, 2026',PTS=13,SEASON='2026')])
    merged=merge_history(old,incoming)
    assert len(merged)==3
    assert merged[merged.SEASON=='2025'].PTS.tolist()==[10]
    assert set(merged.PTS)=={10,12,13}


def test_save_history_normalizes_mixed_provider_ids_before_parquet(tmp_path, monkeypatch):
    old = pd.DataFrame([dict(PLAYER_NAME='A', GAME_DATE='May 01, 2026', PTS=10,
                             SEASON=2026, SEASON_ID=22026, Player_ID=1,
                             TEAM_ID=2, Game_ID=3)])
    old.to_csv(tmp_path / 'player_game_logs.csv', index=False)
    incoming = pd.DataFrame([dict(PLAYER_NAME='A', GAME_DATE='May 02, 2026', PTS=12,
                                  SEASON='2026', SEASON_ID='22026', Player_ID='1',
                                  TEAM_ID='2', Game_ID='4')])
    monkeypatch.setattr(refresh_wnba_history, 'engineer_features', lambda rows: rows.copy())

    result, backup = refresh_wnba_history.save_history(incoming, tmp_path)

    assert len(result) == 2
    assert pd.read_parquet(tmp_path / 'engineered_data.parquet')['SEASON_ID'].tolist() == ['22026', '22026']
    assert (backup / 'player_game_logs.csv').exists()
    assert not list(tmp_path.glob('.history-*'))
