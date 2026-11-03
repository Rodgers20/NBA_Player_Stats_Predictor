import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import polars as pl
from futball.predict import features, tune, serve

class PredictionTests(unittest.TestCase):
    def test_score_grid_is_probability_distribution(self):
        lh=np.array([.05, 1.5, 8.]); la=np.array([8., 1.1, 8.])
        for rho in (0., -.16):
            g=tune.dixon_coles(tune.poisson_grid(lh,la),lh,la,rho)
            self.assertTrue(np.isfinite(g).all())
            self.assertTrue((g>=0).all())
            np.testing.assert_allclose(g.sum((1,2)),1)
            np.testing.assert_allclose(tune.grid_to_1x2(g).sum(1),1)
            self.assertTrue((tune.grid_to_over(g,1.5)>=tune.grid_to_over(g,2.5)).all())
            self.assertTrue((tune.grid_to_over(g,2.5)>=tune.grid_to_over(g,3.5)).all())

    def test_unplayed_fixtures_do_not_change_form(self):
        rows=[]
        for i in range(1,9):
            rows.append(dict(match_key=str(i),team='A',date=f'2025-01-{i:02}',season='2024',is_home=True,
                             gf=2,ga=1,points=3,shots=10,sot=4,shots_against=8,corners=5))
        fixture={**rows[-1], 'match_key':'target','date':'2025-01-12','gf':None,'ga':None,'points':0,'shots':None,'sot':None,'shots_against':None,'corners':None}
        one=features._rolling(pl.DataFrame(rows+[fixture])).filter(pl.col('match_key')=='target')
        many=features._rolling(pl.DataFrame(rows+[{**fixture,'match_key':'earlier','date':'2025-01-10'},fixture])) .filter(pl.col('match_key')=='target')
        for c in ('ppg_r5','gf_r5','rest_days','team_match_no','season_ppg','venue_gf'):
            self.assertEqual(one[c][0],many[c][0], c)
        self.assertEqual(one['ppg_r5'][0],3)
        self.assertEqual(one['rest_days'][0],4)

    def test_same_day_results_cannot_change_form(self):
        rows=[dict(match_key=str(i),team='A',date=f'2025-01-{i:02}',season='2024',is_home=True,
                   gf=2,ga=1,points=3,shots=10,sot=4,shots_against=8,corners=5) for i in range(1,4)]
        rows.append({**rows[-1],'match_key':'later','gf':0,'points':0})
        out=features._rolling(pl.DataFrame(rows)).filter(pl.col('date')=='2025-01-03')
        self.assertEqual(out['gf_r5'].to_list(),[2.,2.])

    def test_backtest_reads_frozen_artifact_not_production(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'backtest.json').write_text(json.dumps({'rows':[{'competition_code':'E0','date':'2025-10-01'}]}))
            with patch.object(serve,'MODEL_DIR',Path(d)), patch.object(serve,'predict',side_effect=AssertionError('production model used')):
                self.assertEqual(len(serve.recent_backtest('E0')),1)
                self.assertEqual(serve.recent_backtest('SP1'),[])

    #: A stub bundle whose goal models return fixed rates.
    @staticmethod
    def _bundle_stub():
        class Model:
            def __init__(self, v): self.v = v
            def predict(self, x): return np.array([self.v])
        return {'home_goals': Model(2.1), 'away_goals': Model(.8),
                'home_goals_form': Model(1.5), 'away_goals_form': Model(1.1),
                'columns': ['mkt_home'], 'form_columns': ['x'],
                'rho': -.1, 'rho_market': -.13}

    def test_served_outputs_share_grid_with_and_without_odds(self):
        """Uncorrected, every served number is that one grid and nothing else."""
        b = self._bundle_stub()
        with patch.object(serve, '_bundle', return_value=b), \
             patch.object(serve.calibration, 'load', return_value={'markets': {}}):
            for odds in (None, (2., 3., 4.)):
                p = serve.predict('E0', 'A', 'B', odds, row=pl.DataFrame({'x': [1.]}))
                self.assertAlmostEqual(sum(p['win'].values()), 1, places=3)
                self.assertAlmostEqual(sum(v['probability'] for v in p['total_goals']['distribution']), 1, places=3)
                lh, la, rho = (2.1, .8, -.13) if odds else (1.5, 1.1, -.1)
                g = tune.dixon_coles(tune.poisson_grid(np.array([lh]), np.array([la])), np.array([lh]), np.array([la]), rho)
                self.assertAlmostEqual(p['win']['home'], tune.grid_to_1x2(g)[0, 2], places=4)

    def test_calibration_shifts_the_rates_before_anything_is_derived(self):
        """A correction must move the whole grid, not one headline number.

        If it were applied to the expected total alone, the served over/under
        and the win probabilities would describe two different matches.
        """
        table = {'markets': {'goals': {'E0': {
            'market': 'goals', 'competition_code': 'E0', 'n': 400,
            'predicted_mean': 2.9, 'actual_mean': 2.61, 'factor': 0.9,
            'clipped': False, 'source': 'held-out', 'through': '2026-09-01'}}}}
        b = self._bundle_stub()
        with patch.object(serve, '_bundle', return_value=b), \
             patch.object(serve.calibration, 'load', return_value=table):
            p = serve.predict('E0', 'A', 'B', None, row=pl.DataFrame({'x': [1.]}))

        lh, la = 1.5 * 0.9, 1.1 * 0.9
        g = tune.dixon_coles(tune.poisson_grid(np.array([lh]), np.array([la])),
                             np.array([lh]), np.array([la]), -.1)
        self.assertAlmostEqual(p['expected_goals']['total'], lh + la, places=2)
        self.assertAlmostEqual(p['win']['home'], tune.grid_to_1x2(g)[0, 2], places=4)
        self.assertEqual(p['calibration']['goals']['factor'], 0.9)

    def test_a_backtest_row_is_never_logged_as_a_live_forecast(self):
        """Grading history must not create new predictions to grade."""
        b = self._bundle_stub()
        with patch.object(serve, '_bundle', return_value=b), \
             patch.object(serve.calibration, 'load', return_value={'markets': {}}), \
             patch.object(serve.calibration, 'record', side_effect=AssertionError('logged')):
            serve.predict('E0', 'A', 'B', None, row=pl.DataFrame({'x': [1.]}))

if __name__=='__main__': unittest.main()
