"""Position resolution and secondary-role ranking.

The bug these lock down: the front end matched positions by substring, looking
for "back", "midfield", "wing" and "forward". None of those words appear in the
current-season feeds, so `Offence`, `Defence`, `Attacker` and `Defender` all fell
through to a default of **MF** — Haaland drawn as a midfielder, and every
centre-back with him.
"""
import unittest

from futball import marts, players, positions


class ResolutionTests(unittest.TestCase):
    def test_the_values_that_rendered_as_MF_now_resolve_correctly(self):
        cases = [
            ("Offence", "FW"), ("Attacker", "FW"), ("Forward", "FW"),
            ("Defence", "DF"), ("Defender", "DF"),
            ("Midfield", "MF"), ("Midfielder", "MF"),
            ("Goalkeeper", "GK"),
        ]
        for value, code in cases:
            with self.subTest(value=value):
                resolved = positions.resolve(value)
                self.assertIsNotNone(resolved, f"{value} did not resolve")
                self.assertEqual(resolved.code, code)

    def test_granular_statsbomb_roles_resolve_to_specific_badges(self):
        for value, code in [
            ("Right Wing", "RW"), ("Left Back", "LB"), ("Center Forward", "ST"),
            ("Center Attacking Midfield", "CAM"), ("Right Wing Back", "RWB"),
        ]:
            with self.subTest(value=value):
                resolved = positions.resolve(value)
                self.assertEqual(resolved.code, code)
                self.assertEqual(resolved.tier, "specific")

    def test_a_coarse_source_never_produces_a_specific_badge(self):
        """"Offence" must not become "ST" — the feed never claimed that much."""
        for value in ("Offence", "Attacker", "Defence", "Midfield"):
            with self.subTest(value=value):
                self.assertEqual(positions.resolve(value).tier, "coarse")

    def test_an_unknown_value_resolves_to_nothing_not_a_default(self):
        for value in ("nonsense", "", None, "Substitute", "  "):
            with self.subTest(value=value):
                self.assertIsNone(positions.resolve(value))

    def test_resolution_is_case_and_whitespace_insensitive(self):
        self.assertEqual(positions.resolve("  RIGHT   WING "), positions.resolve("Right Wing"))

    def test_every_position_the_event_data_emits_resolves(self):
        try:
            rows = marts._rows(
                "select distinct position from events where position is not null")
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"event lake unavailable: {exc}")
        unresolved = [
            r["position"] for r in rows
            if r["position"] != "Substitute" and positions.resolve(r["position"]) is None
        ]
        self.assertEqual(unresolved, [], f"unmapped positions: {unresolved}")

    def test_pitch_coordinates_are_inside_the_pitch(self):
        for value in positions.known_values():
            with self.subTest(value=value):
                p = positions.resolve(value)
                self.assertTrue(0.0 <= p.x <= 1.0)
                self.assertTrue(0.0 <= p.y <= 1.0)

    def test_the_pitch_runs_from_goalkeeper_back_to_forward(self):
        """y increases toward a team's own goal, so roles order correctly."""
        y = {c: positions.resolve(v).y for v, c in
             [("Goalkeeper", "GK"), ("Center Back", "CB"),
              ("Center Midfield", "CM"), ("Center Forward", "ST")]}
        self.assertGreater(y["GK"], y["CB"])
        self.assertGreater(y["CB"], y["CM"])
        self.assertGreater(y["CM"], y["ST"])

    def test_left_and_right_roles_sit_on_opposite_flanks(self):
        self.assertLess(positions.resolve("Left Wing").x, 0.5)
        self.assertGreater(positions.resolve("Right Wing").x, 0.5)


class UsageRankingTests(unittest.TestCase):
    def setUp(self):
        try:
            marts.connection()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"event lake unavailable: {exc}")

    def test_positions_come_back_most_used_first(self):
        rows = marts.player_positions(1460)  # Saka: several roles at Euro 2024
        if len(rows) < 2:
            self.skipTest("player has a single position")
        matches = [r["matches"] for r in rows]
        self.assertEqual(matches, sorted(matches, reverse=True))

    def test_substitute_is_not_treated_as_a_position(self):
        rows = marts._rows(
            "select distinct player_id from events where position = 'Substitute' limit 1")
        if not rows:
            self.skipTest("no substitute rows")
        got = marts.player_positions(rows[0]["player_id"])
        self.assertNotIn("Substitute", [r["position"] for r in got])


class ProfileDetailTests(unittest.TestCase):
    def setUp(self):
        from futball import clubs
        if not clubs.directory():
            self.skipTest("club directory not cached")
        players.refresh_caches()

    def test_haaland_is_a_forward_not_a_midfielder(self):
        """The reported bug, as a test."""
        hits = players.search("Haaland")
        if not hits:
            self.skipTest("Haaland not in the cached boards")
        detail = players.profile(hits[0]["player_id"])["position_detail"]
        self.assertEqual(detail["primary"]["code"], "FW")

    def test_secondary_shares_are_ordered_and_bounded(self):
        hits = players.search("Saka")
        if not hits:
            self.skipTest("Saka not in the cached boards")
        detail = players.profile(hits[0]["player_id"])["position_detail"]
        shares = [s["share"] for s in detail["secondary"]]
        self.assertEqual(shares, sorted(shares, reverse=True))
        for s in shares:
            self.assertGreater(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_a_secondary_never_duplicates_the_primary(self):
        for query in ("Saka", "Yamal", "Mbapp"):
            hits = players.search(query)
            if not hits:
                continue
            detail = players.profile(hits[0]["player_id"]).get("position_detail")
            if not detail:
                continue
            with self.subTest(player=query):
                codes = [s["code"] for s in detail["secondary"]]
                self.assertNotIn(detail["primary"]["code"], codes)
                self.assertEqual(len(codes), len(set(codes)))

    def test_a_player_with_no_event_data_has_no_secondaries(self):
        hits = players.search("Haaland")
        if not hits:
            self.skipTest("Haaland not in the cached boards")
        detail = players.profile(hits[0]["player_id"])["position_detail"]
        self.assertEqual(detail["secondary"], [])


if __name__ == "__main__":
    unittest.main()
