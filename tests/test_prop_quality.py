from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import time
import pandas as pd
import numpy as np
import pytest
from utils.prop_quality import quote_problem, history_problem
from utils.manual_quote import evaluate


def quote():
    return dict(event_id='event', commence_time=datetime.now(ZoneInfo('America/New_York')).replace(hour=23,minute=59,second=59).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(), fetched_at=time.time())


@pytest.mark.parametrize('problem',['started','old','no_time','off_date'])
def test_recommendations_reject_bad_quotes(problem):
    q=quote()
    if problem=='started': q['commence_time']=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
    if problem=='old': q['updated_at']=(datetime.now(timezone.utc)-timedelta(hours=7)).isoformat()
    if problem=='no_time': del q['updated_at']
    if problem=='off_date': q['commence_time']=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat()
    assert quote_problem(q)


def test_recent_quote_and_stale_player_history():
    assert quote_problem(quote()) is None
    history=pd.DataFrame({'GAME_DATE':pd.date_range(end=pd.Timestamp.today()-pd.Timedelta(days=60),periods=20)})
    assert 'older than 14 days' in history_problem(history)


def test_manual_line_uses_actual_price_and_does_not_log_bets():
    from utils.personal_bets import list_bets
    class Model:
        calibration_residuals=np.linspace(-10,10,200)
        def predict_player_game(self,*args,**kwargs): return {'predicted_pts':20}
    history=pd.DataFrame({'PLAYER_NAME':['A']*20,'GAME_DATE':pd.date_range(end=pd.Timestamp.today()-pd.Timedelta(days=1),periods=20)})
    payload=dict(game_date=datetime.now(ZoneInfo('America/New_York')).date().isoformat(),book='Manual Book',side='Over',line=17.5,price=150,stat='PTS')
    result=evaluate(payload,history,Model(),is_home=True,available=True)
    assert 'User-entered quote' in result and 'at +150' in result
    assert 'No bet saved or placed' in result
    assert list_bets()==[]
    assert 'unavailable' in evaluate(payload,history,Model(),is_home=True,available=False)


def test_wnba_strict_board_excludes_synthetic_and_retired_players(monkeypatch):
    from utils.wnba_props import generate_wnba_props
    monkeypatch.setattr('utils.wnba_injuries.is_player_unavailable',lambda name:False)
    class Model:
        calibration_residuals=np.linspace(-5,5,100)
        def predict(self,features): return 20
    now=pd.Timestamp.today().normalize()
    rows=[]
    for player,age in [('Current',1),('Retired',90)]:
        for day in pd.date_range(end=now-pd.Timedelta(days=age),periods=20):
            rows.append(dict(PLAYER_NAME=player,_date=day,GAME_DATE=str(day.date()),TEAM_ABBREVIATION='LVA',
                             MATCHUP='LVA vs. SEA',PTS=20,REB=8,AST=4,MIN=30))
    markets={}
    for name in ('Current','Retired'):
        markets[name]={'PTS':dict(quote(),home_team='Las Vegas Aces',away_team='Seattle Storm',
                                 line=15.5,over_price=150,under_price=-110,bookmaker='Test')}
    result=generate_wnba_props(pd.DataFrame(rows),lambda stat:Model(),markets,
                              strict_quality=True,synthesize_missing=False,only_active_tonight=False)
    assert [p.player_name for p in result]==['Current']
    assert result[0].pick=='OVER'
    assert result[0].quote_source['event_id']=='event'
    assert generate_wnba_props(pd.DataFrame(rows),lambda stat:Model(),{},strict_quality=True,synthesize_missing=True)==[]


def test_cached_nba_board_rechecks_quote_expiry(monkeypatch):
    from utils import props_cache
    q=quote()
    prop={'quote_source':q}
    monkeypatch.setattr(props_cache,'_props_cache',{'main_page_data':[prop],'callback_data':[prop],'sidebar_data':[prop]})
    assert len(props_cache.get_cached_props()['main_page_data'])==1
    q['updated_at']=(datetime.now(timezone.utc)-timedelta(hours=7)).isoformat()
    cache=props_cache.get_cached_props()
    assert not cache['main_page_data'] and not cache['callback_data'] and not cache['sidebar_data']


def test_strict_wnba_board_caps_five_distinct_players_and_rejects_unpriced(monkeypatch):
    from utils.wnba_props import generate_wnba_props
    monkeypatch.setattr('utils.wnba_injuries.is_player_unavailable', lambda name: False)

    class Model:
        calibration_residuals = np.linspace(-5, 5, 100)
        def predict(self, features):
            return 20

    yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    rows, markets = [], {}
    for index in range(8):
        name = f'Player {index}'
        for day in pd.date_range(end=yesterday, periods=20):
            rows.append(dict(PLAYER_NAME=name, _date=day, GAME_DATE=str(day.date()),
                             TEAM_ABBREVIATION='LVA', MATCHUP='LVA vs. SEA',
                             PTS=20, REB=8, AST=4, MIN=30))
        markets[name] = {
            stat: dict(quote(), line=line, over_price=150 if index < 7 else None,
                       under_price=-110 if index < 7 else None, bookmaker='Test')
            for stat, line in [('PTS', 15.5), ('REB', 10.5)]
        }
    props = generate_wnba_props(pd.DataFrame(rows), lambda stat: Model(), markets,
                                strict_quality=True, only_active_tonight=False)
    assert len(props) == len({prop.player_name for prop in props}) == 5
    assert all(prop.ev > 0 and prop.has_live_odds for prop in props)
    assert all(prop.player_name != 'Player 7' for prop in props)
    for player in markets.values():
        for market in player.values():
            market['updated_at'] = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    assert generate_wnba_props(pd.DataFrame(rows), lambda stat: Model(), markets,
                               strict_quality=True, only_active_tonight=False) == []
