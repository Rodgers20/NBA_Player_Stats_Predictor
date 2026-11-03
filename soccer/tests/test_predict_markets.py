"""Corners, shots on target, the explanations, and the self-correction loop."""
import json
import unittest

import numpy as np

from futball.predict import calibration, reasons, serve


def _skip_without_models(test):
    try:
        serve._bundle()
    except serve.ModelsUnavailable:
        test.skipTest("models not trained; run `python -m futball.predict.train`")


class CountMarketTests(unittest.TestCase):
    def setUp(self):
        _skip_without_models(self)
        self.p = serve.predict("SP1", "Betis", "Villarreal")

    def test_both_count_markets_are_served_for_a_domestic_league(self):
        for market in ("corners", "sot"):
            with self.subTest(market=market):
                self.assertTrue(self.p[market]["available"], self.p[market].get("note"))

    def test_over_and_under_are_a_partition(self):
        for market in ("corners", "sot"):
            total = self.p[market]["total"]
            with self.subTest(market=market):
                self.assertAlmostEqual(total["over"] + total["under"], 1.0, places=6)

    def test_the_match_total_is_the_two_sides_added(self):
        """Within rounding: the total is rounded from the unrounded rates, not
        from the two already-rounded sides, so the last penny can differ."""
        for market in ("corners", "sot"):
            block = self.p[market]
            with self.subTest(market=market):
                self.assertAlmostEqual(
                    block["total"]["expected"],
                    block["home"]["expected"] + block["away"]["expected"],
                    delta=0.02,
                )

    def test_each_side_carries_a_team_line(self):
        for market in ("corners", "sot"):
            for side in ("home", "away"):
                with self.subTest(market=market, side=side):
                    block = self.p[market][side]
                    self.assertGreaterEqual(block["line"], 1)
                    self.assertTrue(0.0 <= block["at_least"] <= 1.0)

    def test_a_higher_threshold_is_never_more_likely(self):
        rows = self.p["corners"]["home"]["distribution"]
        probabilities = [r["probability"] for r in rows]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_both_teams_is_no_likelier_than_either_alone(self):
        """P(both reach N) must not exceed P(one of them reaches N)."""
        for market in ("corners", "sot"):
            both = self.p[market]["both_teams"]
            if not both:
                continue
            for side in ("home", "away"):
                # Compare at the SAME threshold: each side's own `line` is
                # rounded from its own rate and is usually a different number.
                alone = next(
                    (d["probability"] for d in self.p[market][side]["distribution"]
                     if d["count"] == both["threshold"]),
                    None,
                )
                with self.subTest(market=market, side=side):
                    self.assertIsNotNone(alone)
                    self.assertLessEqual(both["probability"], alone + 1e-9)


class ChampionsLeagueTests(unittest.TestCase):
    """The competition that publishes neither must say so, not extrapolate."""

    def setUp(self):
        _skip_without_models(self)
        teams = serve.teams("UCL")
        if len(teams) < 2:
            self.skipTest("no Champions League teams cached")
        self.p = serve.predict("UCL", teams[0], teams[1])

    def test_count_markets_are_absent_rather_than_guessed(self):
        for market in ("corners", "sot"):
            with self.subTest(market=market):
                self.assertFalse(self.p[market]["available"])
                self.assertIn("does not publish", self.p[market]["note"])

    def test_the_result_forecast_still_works(self):
        self.assertAlmostEqual(sum(self.p["win"].values()), 1.0, places=3)

    def test_the_absence_is_listed_as_a_limitation(self):
        joined = " ".join(self.p["limitations"]).lower()
        self.assertIn("corner lines are unavailable", joined)


class ReasonTests(unittest.TestCase):
    def setUp(self):
        _skip_without_models(self)
        self.p = serve.predict("SP1", "Betis", "Villarreal")

    def test_every_market_is_explained(self):
        markets = {r["market"] for r in self.p["reasons"]}
        self.assertLessEqual({"result", "goals", "corners", "sot"}, markets)

    def test_the_result_reason_names_the_side_the_model_picked(self):
        text = next(r["text"] for r in self.p["reasons"] if r["market"] == "result")
        pick = self.p["result_pick"]
        expected = {"home": self.p["home_team"], "away": self.p["away_team"]}.get(pick)
        if expected:
            self.assertIn(expected, text)

    def test_the_goals_reason_agrees_with_the_forecast(self):
        text = next(r["text"] for r in self.p["reasons"] if r["market"] == "goals")
        side = "over" if self.p["total_goals"]["over_2_5"] >= 0.5 else "under"
        self.assertIn(f"{side} 2.5", text)

    def test_the_corner_reason_quotes_the_served_numbers(self):
        text = next(r["text"] for r in self.p["reasons"] if r["market"] == "corners")
        self.assertIn(f"{self.p['corners']['total']['expected']:.1f}", text)

    def test_reasons_are_sentences_not_fragments(self):
        for r in self.p["reasons"]:
            with self.subTest(market=r["market"]):
                self.assertTrue(r["text"].endswith("."))
                self.assertGreater(len(r["text"]), 40)


class CalibrationTests(unittest.TestCase):
    def test_a_correction_is_bounded(self):
        for raw in (0.1, 0.5, 1.0, 2.0, 50.0):
            factor, clipped = calibration._clip(raw)
            with self.subTest(raw=raw):
                self.assertGreaterEqual(factor, 1 - calibration.MAX_SHIFT)
                self.assertLessEqual(factor, 1 + calibration.MAX_SHIFT)
                self.assertEqual(clipped, not (1 - calibration.MAX_SHIFT <= raw <= 1 + calibration.MAX_SHIFT))

    def test_measurement_needs_enough_settled_predictions(self):
        measured = calibration.measure()
        for market, codes in measured.items():
            for code, c in codes.items():
                with self.subTest(market=market, code=code):
                    self.assertGreaterEqual(c.n, calibration.MIN_MATCHES)

    def test_the_factor_points_the_right_way(self):
        """Predicting too high must pull the rate down, not up."""
        for codes in calibration.measure().values():
            for code, c in codes.items():
                with self.subTest(code=code, market=c.market):
                    if c.clipped:
                        continue
                    if c.predicted_mean > c.actual_mean:
                        self.assertLess(c.factor, 1.0)
                    elif c.predicted_mean < c.actual_mean:
                        self.assertGreater(c.factor, 1.0)

    def test_a_lean_is_named_from_the_factor(self):
        made_up = calibration.Correction(
            market="corners", competition_code="E0", n=200,
            predicted_mean=10.4, actual_mean=9.6, factor=0.923,
            clipped=False, source="held-out", through="2026-09-01")
        self.assertEqual(made_up.lean, "over")

    def test_a_corrupt_served_log_line_is_skipped_not_fatal(self):
        original = calibration.SERVED_LOG
        try:
            path = original.parent / "served_test.jsonl"
            path.write_text('{"competition_code":"E0"}\n{ this is not json\n')
            calibration.SERVED_LOG = path
            calibration._from_served()  # must not raise
        finally:
            calibration.SERVED_LOG = original
            (original.parent / "served_test.jsonl").unlink(missing_ok=True)


class CalibrationAffectsServingTests(unittest.TestCase):
    def setUp(self):
        _skip_without_models(self)

    def test_the_applied_correction_is_published(self):
        p = serve.predict("SP1", "Betis", "Villarreal")
        self.assertIn("calibration", p)
        for market in ("goals", "corners", "sot"):
            c = p["calibration"].get(market)
            if c:
                with self.subTest(market=market):
                    self.assertEqual(c["competition_code"], "SP1")
                    self.assertGreaterEqual(c["n"], calibration.MIN_MATCHES)

    def test_a_shifted_rate_is_explained_rather_than_silent(self):
        p = serve.predict("SP1", "Betis", "Villarreal")
        shifted = [c for c in p["calibration"].values()
                   if c and abs(c["factor"] - 1) >= 0.005]
        texts = {r["market"] for r in p["reasons"]}
        if shifted:
            self.assertIn("calibration", texts)


if __name__ == "__main__":
    unittest.main()
