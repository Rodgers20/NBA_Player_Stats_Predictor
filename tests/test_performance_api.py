import json
from types import SimpleNamespace
from api.routes import performance


def test_only_timestamp_verified_game_forecasts_count(tmp_path, monkeypatch):
    good={'game_id':'1','home':'BOS','away':'NYK','game_time':'2026-04-10T23:00Z',
          'actuals':{'home_score':110,'away_score':100},
          'predictions':{'winner_pick':'BOS','model_home_score':108,'model_away_score':99,'model_total':207}}
    late={**good,'game_id':'2','game_time':'2026-04-10T20:00Z'}
    unknown={**good,'game_id':'3','game_time':'7:00 PM ET'}
    (tmp_path/'prediction_history.json').write_text(json.dumps({'2026-04-10':{'generated_at':'2026-04-10T22:00Z','games':[good,late,unknown]},'2026-08-10':[{'player_name':'not a game'}]}))
    monkeypatch.setattr(performance,'get_config',lambda league:SimpleNamespace(data_dir=tmp_path))
    result=performance._live_history('nba')
    season=next(s for s in result['seasons'] if s['season']==2025)
    assert season['record']['games']==1
    assert season['record']['winner']=={'n':1,'correct':1,'rate':1}
    assert season['record']['total_mae']==3
    assert season['record']['margin_mae']==1
    assert result['excluded_unverified']==2


def test_missing_evaluation_and_game_history_are_honest(tmp_path,monkeypatch):
    monkeypatch.setattr(performance,'REPORT_DIR',tmp_path)
    monkeypatch.setattr(performance,'get_config',lambda league:SimpleNamespace(data_dir=tmp_path))
    result=performance.get_performance('wnba')
    assert result['evaluation'] is None
    assert result['live']['seasons'][0]['record']['winner']['rate'] is None
    assert result['market_record']['available'] is False
    assert result['prospective']['saved']==0


def test_holdout_does_not_mix_with_live_or_other_model_ledger(tmp_path,monkeypatch):
    report={'version':'current','holdout':{'winner':{'accuracy':.7}}}
    (tmp_path/'nba-evaluation.json').write_text(json.dumps(report))
    (tmp_path/'game_forecast_ledger.json').write_text(json.dumps({'forecasts':[{'model_version':'old','actual':None},{'model_version':'current','actual':None}]}))
    monkeypatch.setattr(performance,'REPORT_DIR',tmp_path)
    monkeypatch.setattr(performance,'get_config',lambda league:SimpleNamespace(data_dir=tmp_path))
    result=performance.get_performance('nba')
    assert result['evaluation']==report
    assert result['prospective']['pending']==1
    assert result['live']['seasons'][0]['record']['games']==0


def test_current_model_ledger_grades_are_separate(tmp_path,monkeypatch):
    (tmp_path/'nba-evaluation.json').write_text(json.dumps({'version':'current','holdout':{'winner':{'accuracy':.7}}}))
    entry=dict(model_version='current',generated_at='2026-04-10T20:00Z',tipoff='2026-04-10T23:00Z',home_team='BOS',away_team='NYK',
        prediction=dict(predicted_winner='BOS',predicted_home_score=108,predicted_away_score=99,total=207),actual=dict(home_score=110,away_score=100))
    (tmp_path/'game_forecast_ledger.json').write_text(json.dumps({'forecasts':[entry]}))
    monkeypatch.setattr(performance,'REPORT_DIR',tmp_path)
    monkeypatch.setattr(performance,'get_config',lambda league:SimpleNamespace(data_dir=tmp_path))
    result=performance.get_performance('nba')
    assert result['prospective']['record']['winner']['rate']==1
    assert result['prospective']['record']['margin_mae']==1
    assert result['prospective']['record']['total_mae']==3
    assert result['prospective']['pending']==0
    assert result['live']['seasons'][0]['record']['games']==0
