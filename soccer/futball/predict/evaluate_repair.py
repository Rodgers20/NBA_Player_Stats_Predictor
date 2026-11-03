"""Reproducible isolated score-model comparison; never writes production models.

Run: python -m futball.predict.evaluate_repair --output /private/tmp/futball-model-review
All historical periods have been consulted previously; results are diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import date
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import PoissonRegressor
from threadpoolctl import threadpool_limits
from . import features, tune, train

FOLDS = [('2023-08-01','2024-08-01'),('2024-08-01','2025-08-01'),
         ('2025-08-01','2026-08-01'),('2026-08-01','2026-09-13')]
RHOS = [-.13,-.05,0.]

def measures(grid,h,a):
    n=len(h); p=tune.grid_to_1x2(grid)
    y=np.where(h>a,2,np.where(h==a,1,0)); picked=p.argmax(1)
    exact=grid[np.arange(n),h,a]; flat=grid.reshape(n,-1)
    modes=flat.argmax(1); actual=h*grid.shape[2]+a
    order=np.argsort(-flat,axis=1,kind='stable')[:,:3]
    over=tune.grid_to_over(grid); truth=(h+a>2.5)
    goal_h=(grid.sum(2)*np.arange(grid.shape[1])).sum(1)
    goal_a=(grid.sum(1)*np.arange(grid.shape[2])).sum(1)
    score_losses=-np.log(np.clip(exact,1e-12,1))
    result_losses=-np.log(np.clip(p[np.arange(n),y],1e-12,1))
    over_losses=-np.log(np.clip(np.where(truth,over,1-over),1e-12,1))
    bins=[]
    for lo in np.arange(0.,.6,.1):
        mask=(p[:,1]>=lo)&(p[:,1]<lo+.1)
        if mask.any(): bins.append(dict(lower=round(float(lo),1),n=int(mask.sum()),predicted=float(p[mask,1].mean()),actual=float((y[mask]==1).mean())))
    return dict(n=n,score_logloss=float(score_losses.mean()),result_logloss=float(result_losses.mean()),
        over_logloss=float(over_losses.mean()),accuracy=float((picked==y).mean()),
        top_score_accuracy=float((modes==actual).mean()),top3_score_accuracy=float((order==actual[:,None]).any(1).mean()),
        total_goals_mae=float(np.abs(goal_h+goal_a-h-a).mean()),
        modal_1_1_rate=float((modes==grid.shape[2]+1).mean()),
        draw_picks=int((picked==1).sum()),actual_draws=int((y==1).sum()),
        draw_precision=float((y[picked==1]==1).mean()) if (picked==1).any() else None,
        draw_recall=float((picked[y==1]==1).mean()),draw_calibration=bins),np.column_stack([score_losses,result_losses,over_losses])

def run(output):
    output.mkdir(parents=True,exist_ok=True)
    tuning=train._tuning(); df=features.build(include_style=True); df=df.filter(~__import__('polars').col('is_fixture'))
    dates=df['date'].to_numpy(); h=df['home_goals'].to_numpy().astype(int);a=df['away_goals'].to_numpy().astype(int)
    assert not df['home_goals'].null_count() and not df['away_goals'].null_count()
    ordinals=np.array([date.fromisoformat(d).toordinal() for d in dates])
    families={}; losses={}
    for family in ['form','market']:
        groups=tuning.get(f'{family}_groups') or (['form','elo','venue','h2h','competition']+(['market'] if family=='market' else []))
        base_params,n_iter,rho=train._cfg(tuning,family)
        X,cols,cat=tune.matrix(df,groups);Xs,style_cols,style_cat=tune.matrix(df,groups+['style'])
        valid_odds=np.isfinite(df.select(['mkt_home','mkt_draw','mkt_away']).to_numpy()).all(1)
        rows=[]
        specs=[('baseline',base_params,n_iter,None,False),
               ('recent_style',base_params,n_iter,730,True),
               ('flexible_style',dict(learning_rate=.04,max_leaf_nodes=31,min_samples_leaf=60,l2_regularization=4.,max_features=1.),200,730,True)]
        for fold,(start,end) in enumerate(FOLDS):
            fit=dates<start; test=(dates>=start)&(dates<end)
            if family=='market': test &= valid_odds
            y_h,y_a=h[test],a[test]
            for name,params,it,half,enhanced in specs:
                xx,cc=(Xs,style_cat) if enhanced else (X,cat)
                weight=None if half is None else np.exp(-np.log(2)*(date.fromisoformat(start).toordinal()-ordinals[fit])/half)
                mh=tune._reg(params,cc).set_params(max_iter=it).fit(xx[fit],h[fit],sample_weight=weight)
                ma=tune._reg(params,cc).set_params(max_iter=it).fit(xx[fit],a[fit],sample_weight=weight)
                lh=np.clip(mh.predict(xx[test]),.05,8);la=np.clip(ma.predict(xx[test]),.05,8)
                for rr in ([rho] if name=='baseline' else RHOS):
                    key=name if name=='baseline' else f'{name}:rho={rr}'
                    grid=tune.dixon_coles(tune.poisson_grid(lh,la),lh,la,rr)
                    metrics,ll=measures(grid,y_h,y_a)
                    rows.append(dict(candidate=key,fold=fold,start=start,end=end,**metrics))
                    losses[(family,key,fold)]=ll
                    print(f'{family} fold {fold} {key}: score {metrics["score_logloss"]:.4f}, result {metrics["result_logloss"]:.4f}, totals {metrics["over_logloss"]:.4f}',flush=True)
            if family=='form':
                encoder=OneHotEncoder(handle_unknown='ignore')
                cats=df.select(['home_team','away_team','competition_code']).to_numpy()
                tx=encoder.fit_transform(cats[fit]);vx=encoder.transform(cats[test])
                weight=np.exp(-np.log(2)*(date.fromisoformat(start).toordinal()-ordinals[fit])/365)
                lh=PoissonRegressor(alpha=.01,max_iter=300).fit(tx,h[fit],sample_weight=weight).predict(vx)
                la=PoissonRegressor(alpha=.01,max_iter=300).fit(tx,a[fit],sample_weight=weight).predict(vx)
                grid=tune.dixon_coles(tune.poisson_grid(lh,la),lh,la,-.05)
                metrics,ll=measures(grid,y_h,y_a);losses[(family,'team_poisson',fold)]=ll
                rows.append(dict(candidate='team_poisson',fold=fold,start=start,end=end,**metrics))
                print(f'team_poisson fold {fold}: score {metrics["score_logloss"]:.4f}',flush=True)
        families[family]=rows
    comparisons={}
    for family,rows in families.items():
        candidates=sorted({r['candidate'] for r in rows}-{'baseline'})
        def mean_loss(key,folds): return np.concatenate([losses[(family,key,f)] for f in folds]).mean(0)
        baseline=mean_loss('baseline',[0,1])
        # Select only on the two oldest origins. Later periods do not select candidates.
        eligible=[k for k in candidates if mean_loss(k,[0,1])[1]<=baseline[1]+.005 and mean_loss(k,[0,1])[2]<=baseline[2]+.005]
        selected=min(eligible,key=lambda k:mean_loss(k,[0,1])[0]) if eligible else 'baseline'
        if mean_loss(selected,[0,1])[0]>=baseline[0]: selected='baseline'
        delta=np.concatenate([losses[(family,selected,f)]-losses[(family,'baseline',f)] for f in [2,3]])
        # Paired week-block bootstrap accounts for some temporal dependence.
        test_dates=dates[(dates>=FOLDS[2][0])&(dates<FOLDS[3][1])]
        if family=='market': test_dates=dates[(dates>=FOLDS[2][0])&(dates<FOLDS[3][1])&valid_odds]
        weeks=np.array([date.fromisoformat(d).toordinal()//7 for d in test_dates])
        blocks=[delta[weeks==w] for w in np.unique(weeks)]
        rng=np.random.default_rng(20260913); boot=[]
        for _ in range(500):
            sampled=rng.integers(0,len(blocks),len(blocks));boot.append(np.concatenate([blocks[i] for i in sampled]).mean(0))
        interval=np.quantile(boot,[.025,.975],axis=0)
        repeatable=all((losses[(family,selected,f)][:,0].mean()<losses[(family,'baseline',f)][:,0].mean()) for f in [2,3])
        promote=selected!='baseline' and interval[1,0]<0 and repeatable and delta.mean(0)[1]<=.005 and delta.mean(0)[2]<=.005
        comparisons[family]=dict(selected_on_development=selected,development_baseline=baseline.tolist(),
            development_selected=mean_loss(selected,[0,1]).tolist(),later_delta=delta.mean(0).tolist(),
            later_delta_week_bootstrap_95=interval.tolist(),promotion_passed=bool(promote))
    report=dict(methodology='Historical diagnostic rolling origins; previously consulted data. Candidate selection uses folds 0/1 only. Metrics order: exact-score, result, over-2.5 log loss.',
        last_result=str(dates.max()),data_sha256=hashlib.file_digest((features.config.SILVER/'results/results.parquet').open('rb'),'sha256').hexdigest(),
        comparisons=comparisons,results=families)
    (output/'evaluation.json').write_text(json.dumps(report,indent=2))
    joblib.dump(dict(losses=losses),output/'paired-losses.joblib')
    print(json.dumps(comparisons,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    with threadpool_limits(limits=2): run(args.output)
