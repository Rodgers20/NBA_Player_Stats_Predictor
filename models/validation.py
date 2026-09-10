"""Date-isolated model selection and honest holdout diagnostics."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error, r2_score, brier_score_loss


def safe_features(df):
    pattern = r"(?:rolling_avg_(?:pts|ast|reb|min|fga|fg_pct|fg3a|fg3_pct)_(?:5|10|20)|season_avg_(?:pts|ast|reb|min)|is_home|days_rest|is_back_to_back|minutes_trend)"
    return [c for c in df.select_dtypes(include=[np.number]) if re.fullmatch(pattern, c)]


def dated_frame(df, features, target):
    result = df.copy()
    result['_validation_date'] = pd.to_datetime(result['GAME_DATE'], format='mixed', errors='raise').dt.normalize()
    if result['_validation_date'].isna().any():
        raise ValueError('Missing game dates cannot be evaluated chronologically')
    keys = ['PLAYER_NAME', '_validation_date'] if 'PLAYER_NAME' in result else ['_validation_date']
    if len(keys) > 1 and result.duplicated(keys).any():
        raise ValueError('Duplicate player games: deduplicate before feature engineering')
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target])
    return result.sort_values('_validation_date', kind='stable').reset_index(drop=True)


def date_partitions(df, test_size=0.2):
    if not 0.05 <= test_size <= 0.35:
        raise ValueError('test_size must be between 0.05 and 0.35')
    dates = np.sort(df['_validation_date'].unique())
    if len(dates) < 30:
        raise ValueError('At least 30 distinct game dates are required')
    cal_start = dates[int(len(dates) * (1 - test_size - 0.2))]
    test_start = dates[int(len(dates) * (1 - test_size))]
    d = df['_validation_date']
    return df[d < cal_start], df[(d >= cal_start) & (d < test_start)], df[d >= test_start]


def expanding_folds(df, count=3):
    dates = np.sort(df['_validation_date'].unique())
    blocks = np.array_split(dates, count + 1)
    for i in range(1, len(blocks)):
        yield (np.flatnonzero(df['_validation_date'] < blocks[i][0]),
               np.flatnonzero(df['_validation_date'].isin(blocks[i])))


class RollingBaseline(RegressorMixin, BaseEstimator):
    def __init__(self, column=0):
        self.column = column

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.maximum(0, np.asarray(X)[:, self.column])


def residual_probability(prediction, line, residuals, direction='over'):
    """Smoothed empirical predictive distribution; equality is a push."""
    outcomes = np.maximum(0, np.rint(float(prediction) + np.asarray(residuals)))
    wins = outcomes > line if direction.lower() == 'over' else outcomes < line
    pushes = outcomes == line
    # A weak symmetric prior prevents finite samples producing 0% or 100%.
    denominator = len(outcomes) + 2
    return (float((wins.sum() + 1) / denominator), float(pushes.sum() / denominator))


def _summary(y, pred):
    return {'mae': float(mean_absolute_error(y, pred)),
            'r2': float(r2_score(y, pred)),
            'bias': float(np.mean(pred - y)), 'samples': len(y)}


def fit_validated(predictor, df, target, features=None, tune=False, test_size=0.2, cv=3):
    features = features or safe_features(df)
    if not features or set(features) - set(safe_features(df)):
        raise ValueError('Training requires an explicit pregame feature allowlist; snapshots and same-game stats are unsafe')
    clean = dated_frame(df, features, target)
    development, calibration, test = date_partitions(clean, test_size)
    X, y = development[features].to_numpy(), development[target].to_numpy()
    candidates = {}
    for window in (5, 10, 20):
        col = f'rolling_avg_{target.lower()}_{window}'
        if col in features:
            candidates[f'rolling_{window}'] = RollingBaseline(features.index(col))
    for alpha in ([1., 100., 1000.] if tune else [100.]):
        candidates[f'ridge_{alpha:g}'] = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    base = predictor._create_model()
    if tune and predictor.model_type == 'xgboost' and type(base).__name__ == 'XGBRegressor':
        for depth in (2, 4):
            for trees in (150, 350):
                model = clone(base).set_params(max_depth=depth, n_estimators=trees,
                    learning_rate=0.03, min_child_weight=20, subsample=0.85,
                    colsample_bytree=0.9, reg_lambda=10, n_jobs=2)
                candidates[f'xgboost_d{depth}_n{trees}'] = model
    else:
        if 'n_jobs' in base.get_params():
            base.set_params(n_jobs=2)
        candidates[predictor.model_type] = make_pipeline(StandardScaler(), base)
    folds = list(expanding_folds(development, cv))
    scores = []
    for name, estimator in candidates.items():
        errors = []
        for train_idx, valid_idx in folds:
            fitted = clone(estimator).fit(X[train_idx], y[train_idx])
            pred = np.maximum(0, fitted.predict(X[valid_idx]))
            errors.extend(np.abs(y[valid_idx] - pred))
        score = float(np.mean(errors))
        scores.append({'candidate': name, 'cv_mae': score})
        print(f'{target} {name}: chronological CV MAE {score:.4f}', flush=True)
    winner = min(scores, key=lambda item: item['cv_mae'])['candidate']
    evaluation_model = clone(candidates[winner]).fit(X, y)
    cal_pred = np.maximum(0, evaluation_model.predict(calibration[features].to_numpy()))
    residuals = calibration[target].to_numpy() - cal_pred
    yt = test[target].to_numpy()
    pt = np.maximum(0, evaluation_model.predict(test[features].to_numpy()))
    metrics = _summary(yt, pt)
    metrics.update({'selected_candidate': winner, 'selection_scores': scores,
                    'train_samples': len(development), 'test_samples': len(test),
                    'calibration_samples': len(calibration), 'features': features,
                    'excluded_incomplete_rows': len(df) - len(clean),
                    'evaluation_method': 'expanding-date CV / separate calibration / untouched final dates'})
    metrics['date_ranges'] = {
        name: [str(part['_validation_date'].min().date()), str(part['_validation_date'].max().date())]
        for name, part in [('development', development), ('calibration', calibration), ('test', test)]}
    metrics['baseline_test'] = {name: _summary(yt, estimator.predict(test[features].to_numpy()))
                                for name, estimator in candidates.items() if name.startswith('rolling_')}
    low, high = np.quantile(residuals, [0.1, 0.9])
    metrics['interval_80_coverage'] = float(np.mean((yt >= np.maximum(0, pt + low)) & (yt <= np.maximum(0, pt + high))))
    metrics['interval_80_mean_width'] = float(np.mean(np.maximum(0, pt + high) - np.maximum(0, pt + low)))
    anchor = f'rolling_avg_{target.lower()}_10'
    if anchor in test:
        lines = np.floor(test[anchor].to_numpy()) + 0.5
        probabilities = np.array([residual_probability(p, line, residuals)[0] for p, line in zip(pt, lines)])
        outcomes = yt > lines
        bins = []
        for a, b in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)):
            mask = (probabilities >= a) & (probabilities < b)
            if mask.any():
                bins.append({'count': int(mask.sum()), 'mean_probability': float(probabilities[mask].mean()),
                             'observed_rate': float(outcomes[mask].mean())})
        metrics['probability_diagnostic'] = {'threshold_source': 'pregame rolling-10 average rounded to half-point; NOT sportsbook odds',
            'brier_score': float(brier_score_loss(outcomes, probabilities)), 'bins': bins,
            'market_calibrated': False, 'roi': None}
    # Paired date-bootstrap uncertainty includes within-day correlation.
    if anchor in test:
        differences = np.abs(pt - yt) - np.abs(test[anchor].to_numpy() - yt)
        daily = pd.DataFrame({'date': test['_validation_date'].to_numpy(), 'difference': differences}).groupby('date')['difference'].agg(['sum', 'count'])
        rng = np.random.default_rng(42)
        sampled = rng.integers(0, len(daily), size=(1000, len(daily)))
        boot = daily['sum'].to_numpy()[sampled].sum(axis=1) / daily['count'].to_numpy()[sampled].sum(axis=1)
        metrics['mae_difference_vs_rolling10_95_ci'] = np.quantile(boot, [.025, .975]).tolist()
    # Selection is finished before either calibration or test outcomes are inspected.
    predictor.model = clone(candidates[winner]).fit(clean[features].to_numpy(), clean[target].to_numpy())
    predictor.scaler = FunctionTransformer().fit(clean[features].to_numpy())
    predictor.feature_columns, predictor.target = features, target
    metrics['requested_model_type'] = predictor.model_type
    predictor.model_type = 'ridge' if winner.startswith('ridge_') else ('rolling_baseline' if winner.startswith('rolling_') else predictor.model_type)
    predictor.calibration_residuals = residuals
    predictor.metrics = metrics
    predictor.metrics['production_refit'] = {'samples': len(clean), 'through': str(clean['_validation_date'].max().date()),
        'note': 'Refit on all rows after evaluation; holdout metrics measure the earlier evaluation fit. Residual calibration is approximate after refit.'}
    predictor.is_trained = True
    return metrics
