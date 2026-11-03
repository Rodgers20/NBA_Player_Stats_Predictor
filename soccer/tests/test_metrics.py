import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from futball.predict import serve

class MetricsTests(unittest.TestCase):
    def test_competition_metrics_do_not_include_other_leagues(self):
        rows=[{'competition_code':'E0','date':'2025-09-01','actual_result':'home','win':{'home':.7,'draw':.2,'away':.1}},
              {'competition_code':'SP1','date':'2025-09-02','actual_result':'away','win':{'home':.7,'draw':.2,'away':.1}}]
        with tempfile.TemporaryDirectory() as d,patch.object(serve,'MODEL_DIR',Path(d)):
            Path(d,'metrics.json').write_text(json.dumps({'n_test':2}))
            Path(d,'backtest.json').write_text(json.dumps({'rows':rows}))
            report=serve.metrics('E0')
            self.assertEqual(report['n_test'],2)
            self.assertEqual(report['competition_metrics']['n'],1)
            self.assertEqual(report['competition_metrics']['accuracy'],1)
            self.assertAlmostEqual(report['competition_metrics']['logloss'],.3566749439)
            self.assertEqual(serve.metrics('UCL')['competition_metrics']['n'],0)
            self.assertIsNone(serve.metrics('UCL')['competition_metrics']['accuracy'])
