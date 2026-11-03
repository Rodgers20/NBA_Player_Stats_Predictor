from unittest.mock import patch
import pytest
from futball import player_details as details


def test_profile_is_matched_by_provider_id_and_does_not_invent_values():
    result=details.normalize(42,{'response':[
        {'player':{'id':99,'height':'200 cm'}},
        {'player':{'id':42,'height':'179','birth':{'date':'2007-07-13'},'nationality':'Spain'}}]},
        {'response':[{'team':{'id':1,'name':'Club'},'seasons':[2024,2022,2024]},
                     {'team':{'id':2,'name':'Youth'},'seasons':[]}]})
    assert result['height']=='179'
    assert result['date_of_birth']=='2007-07-13'
    assert result['career'][0]['seasons']==[2024,2022]
    assert result['career'][1]['seasons']==[]
    assert result['market_value'] is None
    assert result['contract_end'] is None
    assert result['preferred_foot'] is None
    assert result['value_history']==[]


@pytest.fixture
def identity(tmp_path,monkeypatch):
    monkeypatch.setattr(details.config,'RAW',tmp_path)
    monkeypatch.setattr(details.players,'_by_public_id',lambda:{'public':'key'})
    monkeypatch.setattr(details.players,'_identities',lambda:{'key':{'name':'Player'}})
    monkeypatch.setattr(details,'provider_id',lambda name:42)
    monkeypatch.setattr(details.af,'available',lambda:True)
    monkeypatch.setattr(details.af,'requests_remaining',lambda:100)


def test_repeat_requests_use_disk_cache(identity):
    with patch.object(details.af,'_get',return_value={'response':[]}) as get:
        first=details.details('public')
        second=details.details('public')
    assert first==second
    assert get.call_count==2


def test_unknown_player_or_low_budget_does_not_fetch(identity,monkeypatch):
    monkeypatch.setattr(details.af,'requests_remaining',lambda:2)
    with patch.object(details.af,'_get') as get:
        assert details.details('invalid') is None
        assert details.details('public')['status']=='quota_unavailable'
    get.assert_not_called()


def test_provider_failure_preserves_partial_bio_and_caches_retry(identity):
    with patch.object(details.af,'_get',side_effect=[{'response':[{'player':{'id':42,'height':'179'}}]},RuntimeError('unavailable')]) as get:
        first=details.details('public')
        second=details.details('public')
    assert first['height']=='179'
    assert first['status']=='partial'
    assert second==first
    assert get.call_count==2
