from unittest.mock import patch
import numpy as np
import polars as pl
from futball.predict import style, serve


def row(key, day, hg, ag, home='A', away='B'):
    return dict(match_key=key,date=day,home_team=home,away_team=away,season='2025/26',
                home_goals=hg,away_goals=ag,home_sot=4,away_sot=2)


def test_style_ignores_future_same_day_results_and_unplayed_rows():
    past=[row('old','2025-09-01',2,0),row('old2','2025-09-04',1,0)]
    fixture=row('target','2025-09-10',None,None)
    base=style.attach(pl.DataFrame(past+[fixture])).filter(pl.col('match_key')=='target')
    extra=[row('future','2025-09-15',9,8),row('same','2025-09-10',7,6),row('unplayed','2025-09-06',None,None)]
    other=style.attach(pl.DataFrame(past+extra+[fixture])).filter(pl.col('match_key')=='target')
    assert base.select(style.COLUMNS).to_dicts()==other.select(style.COLUMNS).to_dicts()


def test_recent_defence_and_missing_shots_remain_distinct():
    good=[row(str(i),f'2025-09-0{i}',2,0) for i in range(1,5)]
    poor=[{**r,'away_goals':4} for r in good]
    fixture=row('next','2025-09-10',None,None)
    low=style.attach(pl.DataFrame(good+[fixture])).tail(1).to_dicts()[0]
    high=style.attach(pl.DataFrame(poor+[fixture])).tail(1).to_dicts()[0]
    assert low['home_style_defence']<high['home_style_defence']
    assert low['home_style_clean_sheet']>high['home_style_clean_sheet']
    absent=style.attach(pl.DataFrame([{**r,'away_sot':None} for r in good]+[fixture])).tail(1).to_dicts()[0]
    assert absent['home_style_sot_coverage']==0
    assert low['home_style_sot_coverage']==1


class Constant:
    def __init__(self, value): self.value=value
    def predict(self, x): return np.full(len(x),self.value)


def test_draw_can_be_the_result_pick_and_top_scores_share_distribution():
    bundle={'home_goals_form':Constant(.5),'away_goals_form':Constant(.5),
            'form_columns':['x'],'rho':-.1}
    with patch.object(serve,'_bundle',return_value=bundle):
        p=serve.predict('E0','A','B',row=pl.DataFrame({'x':[1.]}))
    assert p['result_pick']=='draw'
    assert p['win']['draw']>max(p['win']['home'],p['win']['away'])
    assert p['top_scores'][0]==p['most_likely_score']
    assert len({(s['home'],s['away']) for s in p['top_scores']})==3
    assert [s['probability'] for s in p['top_scores']]==sorted([s['probability'] for s in p['top_scores']],reverse=True)
    assert abs(p['total_goals']['under_2_5']+p['total_goals']['over_2_5']-1)<1e-9


def test_elo_and_head_to_head_do_not_read_same_day_scores():
    from futball.predict import elo, features
    data=pl.DataFrame([row('prior','2025-09-01',1,0),
                       row('a-result','2025-09-10',8,0),
                       row('z-target','2025-09-10',None,None)])
    without=data.filter(pl.col('match_key')!='a-result')
    for fn,columns in [(elo.attach,['home_elo','away_elo']),
                       (features._head_to_head,['h2h_gd','h2h_meetings'])]:
        left=fn(data).filter(pl.col('match_key')=='z-target').select(columns)
        right=fn(without).filter(pl.col('match_key')=='z-target').select(columns)
        assert left.to_dicts()==right.to_dicts()
