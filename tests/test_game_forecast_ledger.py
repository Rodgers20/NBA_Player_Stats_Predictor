from datetime import datetime,timezone
import json
from utils.game_forecast_ledger import record_forecasts


def test_only_pregame_first_forecast_is_saved(tmp_path):
    path=tmp_path/'ledger.json';now=datetime(2026,10,10,12,tzinfo=timezone.utc)
    game=dict(game_id='1',matchup='BOS @ CLE',home_team='CLE',away_team='BOS',start_time='2026-10-10T23:00:00Z',status_state='pre',spread=3.5)
    pred=dict(matchup='BOS @ CLE',spread=-2,model_metadata={'version':'v1'})
    assert record_forecasts('nba',[game],[pred],path,now)==1
    assert record_forecasts('nba',[game],[dict(pred,spread=-8)],path,now)==0
    saved=json.loads(path.read_text())['forecasts'][0]
    assert saved['prediction']['spread']==-2
    assert saved['market']['spread']==3.5
    assert record_forecasts('nba',[dict(game,game_id='2',start_time='2026-10-10T11:00:00Z')],[pred],path,now)==0
    assert record_forecasts('nba',[dict(game,game_id='3',start_time='7:00 PM')],[pred],path,now)==0


def test_grading_never_creates_or_rewrites_forecasts(tmp_path):
    import pandas as pd
    from utils.game_forecast_ledger import grade_forecasts
    path=tmp_path/'ledger.json'
    game=dict(game_id='1',matchup='BOS @ CLE',home_team='CLE',away_team='BOS',start_time='2026-10-10T23:00:00Z',status_state='pre')
    pred=dict(matchup='BOS @ CLE',spread=-2,model_metadata={'version':'v1'})
    record_forecasts('nba',[game],[pred],path,datetime(2026,10,10,12,tzinfo=timezone.utc))
    teams=pd.DataFrame([dict(game_id='0000000001',team='CLE',PTS=102,date=pd.Timestamp('2026-10-10')),dict(game_id='0000000001',team='BOS',PTS=110,date=pd.Timestamp('2026-10-10'))])
    assert grade_forecasts('nba',teams,path,datetime(2026,10,11,12,tzinfo=timezone.utc))==1
    saved=json.loads(path.read_text())['forecasts'][0]
    assert saved['prediction']['spread']==-2
    assert saved['actual']['home_score']==102
    assert grade_forecasts('nba',teams,path,datetime(2026,10,11,12,tzinfo=timezone.utc))==0
