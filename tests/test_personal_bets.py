import pytest
from utils import personal_bets as bets, odds_budget


def wager(mode='paper', price=-110):
    return dict(mode=mode, league='wnba', player="A'ja Wilson", game_date='2026-09-17',
                stat='PTS', side='Over', line=24.5, price=price, book='Test Book', stake=11, notes='')


def test_actual_stakes_settlement_corrections_and_modes():
    paper = bets.add_bet(wager(), 'paper')
    real = bets.add_bet(wager('real',150), 'real')
    assert bets.summary('paper')['pending'] == 11
    assert bets.summary('paper')['roi'] is None
    bets.settle_bet(paper,'win')
    bets.settle_bet(real,'loss')
    assert bets.summary('paper')['profit'] == 10
    assert bets.summary('real')['profit'] == -11
    bets.settle_bet(real,'win')
    assert bets.summary('real')['profit'] == 16.5
    bets.settle_bet(real,'push')
    assert bets.summary('real')['profit'] == 0
    assert bets.summary('real')['staked'] == 11
    bets.settle_bet(real,'void')
    assert bets.summary('real')['roi'] is None
    bets.settle_bet(paper,'pending')
    assert bets.summary('paper')['profit'] == 0
    assert bets.summary('paper')['pending'] == 11


def test_duplicate_submit_is_idempotent_and_persists():
    identifier = bets.add_bet(wager(), 'same-click')
    assert bets.add_bet(wager(), 'same-click') == identifier
    assert len(bets.list_bets()) == 1
    assert odds_budget.DB_PATH.exists()


@pytest.mark.parametrize('key,value', [('price',0),('price',float('nan')),('stake',-1),
                                      ('stake',float('inf')),('line',1.3),('book',''),
                                      ('mode','fake'),('game_date','bad')])
def test_invalid_wager_rejected(key,value):
    payload = wager()
    payload[key] = value
    with pytest.raises((ValueError,TypeError)):
        bets.add_bet(payload,'bad')
    assert not bets.list_bets()


def test_daily_and_monthly_budget_survive_connections(monkeypatch):
    for _ in range(4):
        assert odds_budget.reserve(3)
    assert not odds_budget.reserve(3)
    assert odds_budget.status()['daily'] == 12
    monkeypatch.setattr(odds_budget,'DAILY_LIMIT',999)
    monkeypatch.setattr(odds_budget,'MONTHLY_LIMIT',13)
    assert not odds_budget.reserve(3)


def test_provider_quota_blocks_requests_without_spending():
    odds_budget.report('Connected',2)
    calls=[]
    assert odds_budget.request(lambda *a,**k: calls.append(1),'https://provider/odds',{'markets':'a,b,c'}) is None
    assert not calls
    assert odds_budget.status()['daily'] == 0


def test_background_prop_reads_make_no_requests(monkeypatch):
    from utils import odds_fetcher, wnba_odds_fetcher
    def unexpected(*args,**kwargs):
        raise AssertionError('background read must not call API')
    for module in (odds_fetcher,wnba_odds_fetcher):
        monkeypatch.setattr(module,'_cache',{})
        monkeypatch.setattr(module,'API_KEY','test')
        monkeypatch.setattr(module,'_get',unexpected)
    assert odds_fetcher.get_live_odds() == {}
    assert wnba_odds_fetcher.get_live_wnba_odds() == {}
