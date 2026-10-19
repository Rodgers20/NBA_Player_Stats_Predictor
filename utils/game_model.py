"""Validated league-specific margin/total artifacts; never loads legacy XGB models.

Ridge/pipeline API: https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Ridge.html
Histogram boosting API: https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
"""
from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.base import BaseEstimator, RegressorMixin


class ConstantRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


MODEL_VERSION = 'game-forecast-v1'


class GameForecastModel:
    def __init__(self, artifact):
        self.artifact = artifact

    @classmethod
    def load(cls, league, path=None):
        if league not in ('nba', 'wnba'):
            raise ValueError('Unknown league')
        path = Path(path) if path else Path(__file__).resolve().parents[1] / 'models' / league / 'game_forecast.pkl'
        if not path.exists():
            return None
        with path.open('rb') as f:
            artifact = pickle.load(f)  # trusted local training artifact, never uploaded input
        if (artifact.get('version') != MODEL_VERSION or artifact.get('league') != league
                or not artifact.get('promoted') or not artifact.get('features')):
            return None
        return cls(artifact)

    def predict(self, features):
        missing = set(self.artifact['features']) - set(features)
        if missing:
            raise ValueError(f'Feature schema mismatch: {sorted(missing)}')
        frame = pd.DataFrame([{key: features[key] for key in self.artifact['features']}])
        margin = float(self.artifact['margin_model'].predict(frame)[0])
        total = float(self.artifact['total_model'].predict(frame)[0])
        sigma = float(self.artifact['margin_sd'])
        offsets = self.artifact['interval_offsets']
        return dict(margin=margin, total=total,
                    home_win_probability=float(ndtr(margin / sigma)),
                    margin_sd=sigma, total_sd=float(self.artifact['total_sd']),
                    model_name=self.artifact['model_name'], metrics=self.artifact['holdout'],
                    trained_through=self.artifact['trained_through'], version=MODEL_VERSION,
                    prediction_intervals={
                        'margin_80': [margin + v for v in offsets['margin']],
                        'total_80': [total + v for v in offsets['total']]},
                    calibration='earlier out-of-fold residuals; untouched temporal holdout',
                    feature_count=len(self.artifact['features']))
