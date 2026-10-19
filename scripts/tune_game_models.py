"""Date-grouped tuning and untouched holdout for NBA/WNBA game forecasts.

Run python scripts/tune_game_models.py --league all. Does not fetch or spend odds credits.
Temporal evaluation: https://scikit-learn.org/1.6/modules/cross_validation.html#time-series-split
No shuffled CV, held-out early stopping or future season aggregate joins.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.game_model import MODEL_VERSION, ConstantRegressor


def date_splits(frame, holdout_fraction=.20):
    dates = np.array(sorted(frame['date'].unique()))
    if len(dates) < 60:
        raise ValueError('Need at least 60 distinct completed-game dates')
    cut = dates[int(len(dates) * (1 - holdout_fraction))]
    development = frame[frame.date < cut].copy()
    holdout = frame[frame.date >= cut].copy()
    devdates = np.array(sorted(development.date.unique()))
    folds = []
    for low, high in ((.5, .67), (.67, .84), (.84, 1.)):
        start = devdates[int(len(devdates)*low)]
        end = devdates[min(int(len(devdates)*high), len(devdates)-1)]
        train = development.index[development.date < start]
        valid = development.index[(development.date >= start) & ((development.date <= end) if high == 1 else (development.date < end))]
        if len(train) and len(valid):
            folds.append((train, valid))
    return development, holdout, folds


def make_candidate(spec, features):
    family, value, columns = spec
    select = ColumnTransformer([('features', 'passthrough', columns)], remainder='drop', verbose_feature_names_out=False)
    if family == 'constant':
        estimator = ConstantRegressor()
    elif family == 'ridge':
        estimator = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), StandardScaler(), Ridge(alpha=value))
    elif family == 'hist':
        # Disable random validation/early stopping: all model selection uses prior-date folds.
        estimator = HistGradientBoostingRegressor(max_iter=180, learning_rate=.05, max_leaf_nodes=value,
                                                  l2_regularization=30, min_samples_leaf=35,
                                                  early_stopping=False, random_state=42)
    elif family == 'xgb':
        from xgboost import XGBRegressor
        estimator = XGBRegressor(n_estimators=250, max_depth=value, learning_rate=.035,
                                 reg_lambda=30, min_child_weight=20, subsample=.85,
                                 colsample_bytree=.85, n_jobs=2, random_state=42)
    else:
        raise ValueError(family)
    return make_pipeline(select, estimator)


def errors(y, pred):
    return dict(mae=float(mean_absolute_error(y, pred)), rmse=float(np.sqrt(mean_squared_error(y, pred))))


def winner_metrics(margin, prediction, sigma):
    outcome = (np.asarray(margin) > 0).astype(int)
    prob = np.clip(ndtr(np.asarray(prediction)/sigma), .001, .999)
    bins=[]
    for lo in np.arange(0, 1, .1):
        mask=(prob >= lo)&(prob < lo+.1)
        if mask.any():
            bins.append(dict(lower=round(float(lo),1), count=int(mask.sum()), predicted=float(prob[mask].mean()), actual=float(outcome[mask].mean())))
    return dict(accuracy=float(np.mean((prob >= .5)==outcome)), log_loss=float(log_loss(outcome, prob, labels=[0,1])),
                brier=float(brier_score_loss(outcome,prob)), calibration_bins=bins)


def paired_date_bootstrap(frame, actual, chosen, baseline):
    # Confidence interval of MAE difference; resample whole dates to retain same-slate dependence.
    tmp=pd.DataFrame({'date':frame.date, 'difference':np.abs(actual-chosen)-np.abs(actual-baseline)})
    groups=[g.difference.to_numpy() for _,g in tmp.groupby('date')]
    rng=np.random.default_rng(42)
    draws=[np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(1000)]
    return [float(x) for x in np.quantile(draws,[.025,.975])]


def run(league):
    from utils.game_features import build_game_features
    logs_path=ROOT/'data'/league/'player_game_logs.csv'
    team_path=ROOT/'data'/league/'game_team_history.csv'
    logs=pd.read_csv(logs_path,low_memory=False)
    # Authoritative team history adapter, when present, supersedes partial player sums.
    fingerprint=hashlib.sha256((team_path if team_path.exists() else logs_path).read_bytes()+(ROOT/'utils/game_features.py').read_bytes()).hexdigest()
    cache=ROOT/'data'/league/f'game_features-{fingerprint[:16]}.parquet'
    if cache.exists():
        frame=pd.read_parquet(cache)
        source=team_path if team_path.exists() else logs_path
    elif team_path.exists():
        from utils.game_data import load_game_team_history
        team_games=load_game_team_history(league)
        from utils.game_features import build_matchup_features
        frame=build_matchup_features(team_games,league=league)
        source=team_path
    else:
        frame=build_game_features(logs,league=league)
        source=logs_path
    if not cache.exists():frame.to_parquet(cache,index=False)
    frame=frame.sort_values(['date','game_id']).reset_index(drop=True)
    excluded={'date','game_id','home_team','away_team','home','away','margin','total','season','home_score','away_score','league','playoffs'}
    features=[c for c in frame if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    if not features or len(frame)<250:
        raise ValueError(f'Insufficient validated features/history for {league}: {len(frame)}')
    dev, holdout, folds=date_splits(frame)
    # Core baseline uses only efficient team strength/form; larger sets must earn their place.
    from utils.game_features import BASELINE_FEATURE_COLUMNS
    core=[c for c in BASELINE_FEATURE_COLUMNS if c in features] + [c for c in ('home_pace_10','away_pace_10') if c in features]
    if not core:
        core=features[:min(10,len(features))]
    specs={'league_mean':('constant',0,core)}
    for a in (10,100,1000):
        specs[f'core_ridge_{a}']=('ridge',a,core)
        specs[f'full_ridge_{a}']=('ridge',a,features)
    for leaves in (7,15):specs[f'hist_{leaves}']=('hist',leaves,features)
    for depth in (2,3):specs[f'xgb_{depth}']=('xgb',depth,features)
    # Ablations explicitly test whether geography/workload improve generalization.
    no_schedule=[c for c in features if not any(x in c for x in ('rest','b2b','games_last4','games_last7','travel','timezone','altitude','road_trip','venue'))]
    specs['ridge_no_schedule']=('ridge',100,no_schedule)
    no_four=[c for c in features if not any(x in c for x in ('efg','tov','oreb','ft_rate'))]
    specs['ridge_no_four_factors']=('ridge',100,no_four)
    X=frame[features]
    results={}; oof={}
    for name,spec in specs.items():
        values={target:[] for target in ('margin','total')}; indexes=[]
        for train_idx,valid_idx in folds:
            indexes.extend(valid_idx)
            for target in ('margin','total'):
                model=make_candidate(spec,features)
                model.fit(X.loc[train_idx],frame.loc[train_idx,target])
                values[target].extend(model.predict(X.loc[valid_idx]))
        oof[name]={target:pd.Series(values[target],index=indexes) for target in values}
        results[name]={target:errors(frame.loc[indexes,target],values[target]) for target in values}
        print(league,name,json.dumps(results[name]),flush=True)
    selected={target:min(results,key=lambda name:results[name][target]['mae']) for target in ('margin','total')}
    # Calibration fits ONLY earlier out-of-fold results, never final holdout outcomes.
    idx=oof[selected['margin']]['margin'].index
    observed=frame.loc[idx,'margin'].to_numpy(); predicted=oof[selected['margin']]['margin'].to_numpy()
    fit=minimize_scalar(lambda s:log_loss((observed>0).astype(int),np.clip(ndtr(predicted/s),.001,.999),labels=[0,1]),bounds=(5,35),method='bounded')
    sigma=float(fit.x)
    total_idx=oof[selected['total']]['total'].index
    margin_residual=observed-predicted
    total_residual=frame.loc[total_idx,'total'].to_numpy()-oof[selected['total']]['total'].to_numpy()
    selected_models={}; predictions={}; baseline_predictions={}; baseline_names={}
    for target in ('margin','total'):
        chosen=selected[target]; model=make_candidate(specs[chosen],features)
        model.fit(X.loc[dev.index],dev[target]); predictions[target]=model.predict(X.loc[holdout.index])
        # Strongest simple baseline chosen from DEVELOPMENT folds, not final holdout.
        baseline=min([n for n in specs if n=='league_mean' or n.startswith('core_ridge')],key=lambda n:results[n][target]['mae'])
        bm=make_candidate(specs[baseline],features);bm.fit(X.loc[dev.index],dev[target])
        baseline_predictions[target]=bm.predict(X.loc[holdout.index]); baseline_names[target]=baseline
        selected_models[target]=model
    measured={target:errors(holdout[target],predictions[target]) for target in predictions}
    measured['winner']=winner_metrics(holdout.margin,predictions['margin'],sigma)
    base_measured={target:errors(holdout[target],baseline_predictions[target]) for target in predictions}
    # Baseline win probabilities calibrated on its own earlier OOF predictions.
    bp=oof[baseline_names['margin']]['margin']; by=frame.loc[bp.index,'margin']
    bs=minimize_scalar(lambda s:log_loss((by>0).astype(int),np.clip(ndtr(bp.to_numpy()/s),.001,.999),labels=[0,1]),bounds=(5,35),method='bounded').x
    base_measured['winner']=winner_metrics(holdout.margin,baseline_predictions['margin'],bs)
    intervals={'margin':[float(v) for v in np.quantile(margin_residual,[.1,.9])],
               'total':[float(v) for v in np.quantile(total_residual,[.1,.9])]}
    measured['interval_coverage_80']={target:float(np.mean((holdout[target]>=predictions[target]+intervals[target][0])&(holdout[target]<=predictions[target]+intervals[target][1]))) for target in predictions}
    # Small noninferiority tolerance accounts for noise; selected core baselines may be deployed honestly.
    promoted=all(measured[t]['mae']<=base_measured[t]['mae']*1.02 for t in ('margin','total')) and measured['winner']['log_loss']<=base_measured['winner']['log_loss']*1.02
    report=dict(version=MODEL_VERSION,league=league,created_at=datetime.now(timezone.utc).isoformat(),
                source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                games=len(frame),development_games=len(dev),holdout_games=len(holdout),
                development_through=str(dev.date.max()),holdout_from=str(holdout.date.min()),holdout_through=str(holdout.date.max()),
                features=features,core_features=core,candidates=results,selected=selected,baseline_names=baseline_names,
                holdout=measured,baseline_holdout=base_measured,promoted=bool(promoted),
                paired_mae_difference_ci95={t:paired_date_bootstrap(holdout,holdout[t].to_numpy(),predictions[t],baseline_predictions[t]) for t in predictions},
                limits=['No archived pregame injury inputs; current injury scenarios excluded from validated metrics.',
                        'No historical sportsbook quote archive; no ATS/over-under profitability or market superiority claim.',
                        'Travel inferred from schedule/team home geography, not observed flights; missing venues flagged.',
                        'Final holdout metrics measure fixed models; production refit on all completed history afterward.'])
    out=ROOT/'docs'/'game-models';out.mkdir(exist_ok=True)
    (out/f'{league}-evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    holdout_predictions=holdout[['date','game_id','home_team','away_team','margin','total']].copy()
    for target in predictions:holdout_predictions[f'predicted_{target}']=predictions[target]
    holdout_predictions.to_csv(out/f'{league}-holdout.csv',index=False)
    if promoted:
        for target,model in selected_models.items():model.fit(X,frame[target])
        artifact=dict(version=MODEL_VERSION,league=league,promoted=True,features=features,
                      margin_model=selected_models['margin'],total_model=selected_models['total'],
                      margin_sd=sigma,total_sd=float(np.std(total_residual)),interval_offsets=intervals,
                      trained_through=str(frame.date.max()),model_name=f"margin:{selected['margin']} / total:{selected['total']}",holdout=measured)
        destination=ROOT/'models'/league/'game_forecast.pkl'
        destination.parent.mkdir(parents=True,exist_ok=True)
        temp=destination.with_suffix('.tmp')
        with temp.open('wb') as f:pickle.dump(artifact,f)
        os.replace(temp,destination)
    print(json.dumps({k:report[k] for k in ('league','selected','holdout','baseline_holdout','promoted')},indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--league',choices=['nba','wnba','all'],default='all')
    args=p.parse_args()
    for league in ('nba','wnba') if args.league=='all' else (args.league,):run(league)
