import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import polars as pl
import joblib
from futball.cache import file_cache
from futball.storage import atomic_path, write_text_atomic
from futball import league
from futball.predict import serve

class CacheTests(unittest.TestCase):
    def test_atomic_update_refreshes_cached_result(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'value'
            write_text_atomic(p,'old')
            calls=[]
            @file_cache(p)
            def read():
                calls.append(1)
                return p.read_text()
            self.assertEqual(read(),'old');self.assertEqual(read(),'old')
            write_text_atomic(p,'new')
            self.assertEqual(read(),'new');self.assertEqual(len(calls),2)

    def test_calendar_rollover_invalidates_without_file_change(self):
        values=iter([1,2])
        @file_cache(daily=True)
        def read(): return next(values)
        with patch('futball.cache._today',return_value='2026-09-12'):
            self.assertEqual(read(),1);self.assertEqual(read(),1)
        with patch('futball.cache._today',return_value='2026-09-13'):
            self.assertEqual(read(),2)

    def test_failed_publish_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'value';p.write_text('good')
            with self.assertRaises(RuntimeError):
                with atomic_path(p) as tmp:
                    tmp.write_text('partial')
                    self.assertEqual(p.read_text(),'good')
                    raise RuntimeError('failed writer')
            self.assertEqual(p.read_text(),'good')
            self.assertEqual(list(Path(d).iterdir()),[p])

    def test_missing_file_creation_invalidates_cached_empty_value(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'value'
            @file_cache(p)
            def read(): return p.read_text() if p.exists() else None
            self.assertIsNone(read());write_text_atomic(p,'ready');self.assertEqual(read(),'ready')

    def test_league_reader_reloads_replaced_parquet(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'results.parquet'
            with patch.object(league,'RESULTS',p),patch.object(league.config,'require_volume'):
                for value in (1,2):
                    with atomic_path(p) as tmp:
                        pl.DataFrame({'season':['2025/26'],'value':[value]}).write_parquet(tmp)
                    self.assertEqual(league._results()['value'][0],value)
            league._results.cache_clear()

    def test_model_reader_reloads_replaced_bundle(self):
        with tempfile.TemporaryDirectory() as d,patch.object(serve,'MODEL_DIR',Path(d)):
            for value in (1,2):
                with atomic_path(Path(d)/'models.joblib') as tmp: joblib.dump({'version':value},tmp)
                self.assertEqual(serve._bundle()['version'],value)
        serve._bundle.cache_clear()
