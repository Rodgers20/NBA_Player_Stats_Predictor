"""Directory listing that survives exFAT.

The T7 is exFAT, which cannot store extended attributes natively. macOS
therefore writes a companion AppleDouble file — `._2025-26.txt` beside
`2025-26.txt` — carrying the xattrs it could not store inline. Those companions
match every glob the data layer uses, and their contents are binary, so
`glob("*.txt")` started handing the parsers 4 KB of AppleDouble header and
`read_text()` raised UnicodeDecodeError mid-refresh.

This is a property of the filesystem, not of any one source, so the guard lives
in one place and every directory scan goes through it.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from futball.storage import data_files, is_sidecar


APPLEDOUBLE_MAGIC = bytes.fromhex("0005160700020000") + b"Mac OS X" + b"\x00" * 16


class SidecarDetectionTests(unittest.TestCase):
    def test_appledouble_companions_are_sidecars(self):
        for name in ("._2025-26.txt", "._E0.csv", "._results.parquet", "._12345.json"):
            with self.subTest(name=name):
                self.assertTrue(is_sidecar(Path(name)))

    def test_macos_metadata_files_are_sidecars(self):
        for name in (".DS_Store", ".Spotlight-V100", ".Trashes", ".fseventsd"):
            with self.subTest(name=name):
                self.assertTrue(is_sidecar(Path(name)))

    def test_partial_writes_are_sidecars(self):
        # atomic_path leaves these behind if a process is killed mid-write.
        self.assertTrue(is_sidecar(Path(".results.parquet.abc123.partial")))

    def test_real_data_files_are_not_sidecars(self):
        for name in ("2025-26.txt", "E0.csv", "results.parquet", "12345.json",
                     "._not_really.txt".replace("._", ""), "a._b.txt"):
            with self.subTest(name=name):
                self.assertFalse(is_sidecar(Path(name)))

    def test_a_leading_dot_alone_does_not_make_a_sidecar(self):
        # A dotfile that is genuinely ours should still be visible if asked for.
        self.assertFalse(is_sidecar(Path(".gitkeep")))


class DataFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _write(self, name, data=b"real content"):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_appledouble_companions_are_excluded(self):
        self._write("2025-26.txt")
        self._write("2024-25.txt")
        self._write("._2025-26.txt", APPLEDOUBLE_MAGIC)
        self._write("._2024-25.txt", APPLEDOUBLE_MAGIC)

        found = [p.name for p in data_files(self.dir, "*.txt")]
        self.assertEqual(found, ["2024-25.txt", "2025-26.txt"])

    def test_results_are_sorted(self):
        for name in ("2025-26.txt", "2018-19.txt", "2021-22.txt"):
            self._write(name)
        found = [p.name for p in data_files(self.dir, "*.txt")]
        self.assertEqual(found, sorted(found))

    def test_nested_patterns_are_filtered_at_every_level(self):
        self._write("2627/E0.csv")
        self._write("2627/._E0.csv", APPLEDOUBLE_MAGIC)
        self._write("._2627/E0.csv")  # a sidecar-named *directory* component
        found = [str(p.relative_to(self.dir)) for p in data_files(self.dir, "*/*.csv")]
        self.assertEqual(found, ["2627/E0.csv"])

    def test_empty_files_are_excluded_when_asked(self):
        self._write("full.csv")
        self._write("empty.csv", b"")
        self.assertEqual(
            [p.name for p in data_files(self.dir, "*.csv", non_empty=True)],
            ["full.csv"])

    def test_empty_files_are_kept_by_default(self):
        self._write("full.csv")
        self._write("empty.csv", b"")
        self.assertEqual(
            [p.name for p in data_files(self.dir, "*.csv")],
            ["empty.csv", "full.csv"])

    def test_a_missing_directory_yields_nothing_rather_than_raising(self):
        self.assertEqual(list(data_files(self.dir / "nope", "*.txt")), [])

    def test_directories_matching_the_pattern_are_excluded(self):
        (self.dir / "a_directory.json").mkdir()
        self._write("real.json")
        self.assertEqual([p.name for p in data_files(self.dir, "*.json")], ["real.json"])

    def test_stem_parsing_stays_safe(self):
        """The concrete crash this prevents in statsbomb.py: int(p.stem)."""
        self._write("12345.json")
        self._write("._12345.json", APPLEDOUBLE_MAGIC)
        ids = sorted(int(p.stem) for p in data_files(self.dir, "*.json"))
        self.assertEqual(ids, [12345])


if __name__ == "__main__":
    unittest.main()
