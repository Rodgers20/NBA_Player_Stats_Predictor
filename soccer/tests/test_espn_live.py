from datetime import datetime, timezone
from unittest.mock import patch
import httpx
from futball.live import espn

NOW=datetime(2026,9,15,20,tzinfo=timezone.utc)

def payload(state='in',name='STATUS_IN_PROGRESS',home='2',away='1',day='2026-09-15'):
 return {'events':[{'id':'1','date':day+'T19:00Z','competitions':[{'status':{'type':{'name':name,'state':state,'shortDetail':'65′','completed':state=='post'}},'competitors':[{'homeAway':'away','team':{'displayName':'Away'},'score':away},{'homeAway':'home','team':{'displayName':'Home'},'score':home}]}]}]}

def parse(body):
 with patch('futball.league._to_lake_name',side_effect=lambda n,c:n):return espn.parse(body,'SP1',NOW)

def test_scores_are_sided_and_zero_is_preserved():
 r=parse(payload(home='0'))[0]
 assert r['home_goals']==0 and r['away_goals']==1 and r['is_live']
 assert r['status_label']=='65′'

def test_scheduled_and_postponed_are_not_live_or_fake_zero():
 for name in ['STATUS_SCHEDULED','STATUS_POSTPONED']:
  r=parse(payload('pre',name,home='0',away='0'))[0]
  assert not r['is_live'] and r['home_goals'] is None

def test_half_time_is_live_but_fulltime_and_suspension_are_not():
 assert parse(payload('in','STATUS_HALFTIME'))[0]['is_live']
 assert not parse(payload('post','STATUS_FULL_TIME'))[0]['is_live']
 assert not parse(payload('in','STATUS_SUSPENDED'))[0]['is_live']

def test_previous_day_only_retains_ongoing_matches():
 assert parse(payload(day='2026-09-14'))
 assert not parse(payload('post','STATUS_FULL_TIME',day='2026-09-14'))

def test_requests_shared_and_failure_retains_timestamp(tmp_path):
 espn._memory.clear()
 with patch.object(espn,'RAW',tmp_path),patch('futball.league._to_lake_name',side_effect=lambda n,c:n),patch.object(espn.httpx,'get') as get:
  get.return_value.json.return_value={'events':[]}
  a=espn.board('SP1'); b=espn.board('SP1')
  assert get.call_count==1 and a==b and not a['stale']
  espn._memory.clear()
  with patch.object(espn._cache,'read_json',return_value={'body':{'events':[]},'fetched_at':'2026-01-01T00:00:00+00:00','ttl':45}):
   get.side_effect=httpx.ConnectError('offline')
   c=espn.board('SP1')
  assert c['stale'] and c['updated_at']=='2026-01-01T00:00:00+00:00'
 espn._memory.clear()

def test_stale_matches_are_excluded_from_live_counts():
 with patch.object(espn,'snapshot',return_value={'stale':True,'matches':[{'is_live':True,'stale':True,'competition_code':'SP1'}]}):
  assert espn.counts()['total']==0
