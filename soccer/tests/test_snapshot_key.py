import json
from pathlib import Path

import pytest

from futball.snapshot_key import snapshot_file

ROWS = json.loads((Path(__file__).parent / "fixtures" / "snapshot_keys.json").read_text())


@pytest.mark.parametrize("row", ROWS, ids=[r["file"] for r in ROWS])
def test_snapshot_file(row: dict) -> None:
    assert snapshot_file(row["path"], row["query"]) == row["file"]
