import sys,json,math,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from threadpoolctl import threadpool_limits
from futball.api.app import app
from futball.predict import serve,features
from futball import config
import polars as pl
import numpy as np
client=TestClient(app)
checks=[]
def get(path,status=200):
 r=client.get(path); body=r.json()
 assert r.status_code==status,(path,r.status_code,body)
 assert body['success']==(status==200),(path,body)
 checks.append(path)
 return body.get('data')
with threadpool_limits(limits=2):
 get('/health'); get('/quality'); get('/competitions'); matches=get('/matches?limit=2');mid=matches[0]['match_id']
 for suffix in ['', '/stats','/events','/momentum','/passes','/shots','/heatmap','/freeze-frames','/lineup-positions','/top-players']:
  get(f'/matches/{mid}{suffix}')
 for path in ['/standings','/players/top-scorers','/players/top-assists','/players/top-contributions','/predict/metrics','/predict/competitions','/live/counts','/live/now']:
  get(path)
 samples=[]
 for code in ['E0','SP1','UCL']:
  for suffix in ['seasons','standings','matches','fixtures']:
   get(f'/leagues/{code}/{suffix}')
  teams=get(f'/predict/teams?competition_code={code}')
  bt=get(f'/predict/backtest?competition_code={code}')
  assert all(r['date']>=r['trained_before'] for r in bt)
  fixtures=get(f'/predict/fixture-list?competition_code={code}')
  for f in fixtures[:3]:
   from urllib.parse import urlencode
   p=get('/predict/by-fixture?'+urlencode({'fixture_key':f['fixture_key']}));samples.append(p)
  for h,a in zip(teams[:5],teams[5:10]):
   from urllib.parse import urlencode
   p=get('/predict/fixture?'+urlencode({'competition_code':code,'home':h,'away':a}));samples.append(p)
 for p in samples:
  assert all(math.isfinite(v) and 0<=v<=1 for v in p['win'].values())
  assert abs(sum(p['win'].values())-1)<.0002
  t=p['total_goals'];assert 0<=t['over_3_5']<=t['over_2_5']<=t['over_1_5']<=1
  assert abs(sum(x['probability'] for x in t['distribution'])-1)<.0005
  assert p['expected_goals']['total']<12
 get('/predict/by-fixture?fixture_key=E0|1900-01-01|Arsenal|Chelsea',404)
 get('/predict/fixture?competition_code=E0&home=Unknown&away=Chelsea',404)
 get('/predict/fixture?competition_code=E0&home=Arsenal&away=Arsenal',400)
 get('/matches/999999999',404);get('/matches?limit=-1',422)
 # Real-data batch invariance across fixture slots, not just toy rolling rows.
 full=serve._fixture_table();row=full.filter(pl.col('competition_code')=='E0').tail(1)
 r=row.to_dicts()[0]
 f={k:r[k] for k in ['competition_code','competition','country','season','date','home_team','away_team']}
 one=features.build(extra_fixtures=[f]).filter(pl.col('is_fixture'))
 cols=serve._bundle()['form_columns']
 np.testing.assert_allclose(serve._vector(row,cols),serve._vector(one,cols),equal_nan=True)
 print(json.dumps({'api_checks_passed':len(checks),'prediction_samples':len(samples),'real_data_batch_invariance':'passed','match_id':mid},indent=2))
