"""Exercise real Dash routes and form callbacks through its HTTP interface."""
import importlib
import json
import pytest
from utils import personal_bets


@pytest.fixture
def dashboard(monkeypatch):
    monkeypatch.setenv('NBA_DISABLE_BACKGROUND','1')
    def offline(*args, **kwargs):
        raise AssertionError('Legacy journal tests must not call the network')
    monkeypatch.setattr('requests.sessions.Session.request', offline)
    monkeypatch.setattr('pandas.DataFrame.to_parquet', offline)
    app=importlib.import_module('dashboard.app')
    app.app.server.test_client().get('/_dash-layout')
    return app


def dispatch(module, output_fragment, values, states=None, changed=None):
    key=next(k for k in module.app.callback_map if output_fragment in k)
    meta=module.app.callback_map[key]
    outs=meta['output']
    def output(o): return {'id':o.component_id,'property':o.component_property}
    payload={'output':key,'outputs':[output(o) for o in outs] if isinstance(outs,list) else output(outs),
             'inputs':[dict(i,value=values.get(i['id'])) for i in meta['inputs']],
             'state':[dict(i,value=(states or {}).get(i['id'])) for i in meta['state']],
             'changedPropIds':[changed] if changed else []}
    response=module.app.server.test_client().post('/_dash-update-component',json=payload)
    assert response.status_code==200, response.data.decode()[:1000]
    return response.json['response']


def test_journal_route_save_and_settle_through_http(dashboard):
    routed=dispatch(dashboard,'page-content.children',{'url':'/my-bets'},changed='url.pathname')
    assert 'My Bets' in json.dumps(routed)
    fields={'mode':'paper','league':'wnba','player':"A'ja Wilson",'game_date':'2026-09-17',
            'stat':'PTS','side':'Over','line':24.5,'price':-110,'book':'Test','stake':11,'notes':'HTTP test'}
    states={'bet-'+k:v for k,v in fields.items()}
    states['journal-token']='http-submit'
    result=dispatch(dashboard,'journal-save-status.children',{'journal-save':1,'journal-new':0},states,'journal-save.n_clicks')
    assert 'Saved paper entry' in result['journal-save-status']['children']
    identifier=personal_bets.list_bets()[0]['id']
    dispatch(dashboard,'journal-settle-status.children',{'journal-settle':1},
             {'journal-bet-id':identifier,'journal-result':'win'},'journal-settle.n_clicks')
    rendered=dispatch(dashboard,'journal-summary.children',{'journal-mode':'paper','journal-save-status':'saved','journal-settle-status':'win'})
    assert '$+10.00' in json.dumps(rendered)
    assert personal_bets.summary('real')['count']==0


def test_aja_actual_models_render_all_three_projections_without_odds(dashboard,monkeypatch):
    import utils.wnba_data_fetch as schedule
    import utils.wnba_injuries as injuries
    import utils.wnba_prediction_tracker as tracker
    monkeypatch.setattr(schedule,'get_todays_wnba_games',lambda:[])
    monkeypatch.setattr(injuries,'get_wnba_player_injury',lambda name:None)
    monkeypatch.setattr(tracker,'get_calibration_offsets',lambda:{})
    history=dashboard.WNBA_DF[dashboard.WNBA_DF.PLAYER_NAME=="A'ja Wilson"].sort_values('_date',ascending=False)
    rendered=str(dashboard._wnba_prediction_card(history))
    assert 'Unavailable:' not in rendered
    for stat in ('PTS','AST','REB'):
        assert f"children='{stat}'" in rendered
    assert 'History through' in rendered
    assert 'Projected stat line' in rendered


def test_page_load_does_not_call_odds_provider(dashboard,monkeypatch):
    import utils.odds_fetcher as nba
    import utils.wnba_odds_fetcher as wnba
    def unexpected(*a,**kw): raise AssertionError('No API calls on journal load')
    monkeypatch.setattr(nba,'_get',unexpected)
    monkeypatch.setattr(wnba,'_get',unexpected)
    response=dispatch(dashboard,'journal-odds-status.children',{'journal-refresh-odds':0},{'odds-league':'wnba'})
    assert 'No fresh quotes' in json.dumps(response)


@pytest.mark.parametrize('league', ['nba', 'wnba'])
def test_journal_game_refresh_uses_espn_without_paid_odds(dashboard, monkeypatch, league):
    from utils import espn_game_odds, odds_budget, odds_fetcher
    calls = []
    before = odds_budget.status()['daily']
    monkeypatch.setattr(odds_fetcher, 'get_game_odds',
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Paid game odds called')))
    monkeypatch.setattr(espn_game_odds, 'get_game_odds',
                        lambda selected, target, events, force_refresh=False:
                        calls.append((selected, target, events, force_refresh)) or {'GSW@LAL': {}})
    response = dispatch(dashboard, 'journal-game-status.children',
                        {'journal-refresh-games': 1}, {'odds-league': league},
                        'journal-refresh-games.n_clicks')
    assert calls == [(league, calls[0][1], None, True)]
    assert f'1 {league.upper()} matchups' in response['journal-game-status']['children']
    assert odds_budget.status()['daily'] == before
