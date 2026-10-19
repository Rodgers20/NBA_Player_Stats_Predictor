"""Tests focus on honest evaluation boundaries and artifact safety."""
import pickle
import numpy as np
import pandas as pd
from scripts.tune_game_models import date_splits, winner_metrics, paired_date_bootstrap
from utils.game_model import GameForecastModel, MODEL_VERSION


def test_temporal_splits_keep_dates_and_games_together():
    frame=pd.DataFrame({'date':np.repeat(pd.date_range('2024-01-01',periods=100),4),'game_id':range(400)})
    dev,holdout,folds=date_splits(frame)
    assert dev.date.max()<holdout.date.min()
    for train,valid in folds:
        assert frame.loc[train,'date'].max()<frame.loc[valid,'date'].min()
        assert set(frame.loc[valid,'date']).isdisjoint(set(holdout.date))
        assert frame.loc[valid].groupby('date').size().eq(4).all()


def test_metrics_penalize_confident_wrong_forecasts():
    actual=np.array([5,-4,9,-8])
    correct=winner_metrics(actual,actual,10)
    wrong=winner_metrics(actual,-actual,10)
    assert correct['log_loss']<wrong['log_loss']
    assert correct['brier']<wrong['brier']


def test_artifact_rejects_wrong_league_or_unpromoted(tmp_path):
    path=tmp_path/'model.pkl'
    for artifact in ({'version':MODEL_VERSION,'league':'wnba','promoted':True,'features':['x']},
                     {'version':MODEL_VERSION,'league':'nba','promoted':False,'features':['x']}):
        path.write_bytes(pickle.dumps(artifact))
        assert GameForecastModel.load('nba',path) is None


def test_mae_ci_resamples_whole_dates():
    frame=pd.DataFrame({'date':np.repeat(pd.date_range('2025-01-01',periods=10),2)})
    y=np.zeros(20)
    assert paired_date_bootstrap(frame,y,np.ones(20),np.full(20,3))==[-2.,-2.]
