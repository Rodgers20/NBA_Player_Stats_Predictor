import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { snapshotKey } from "./snapshot-key.mjs";

const rows = JSON.parse(
  readFileSync(new URL("../../tests/fixtures/snapshot_keys.json", import.meta.url), "utf8"),
);

for (const r of rows) {
  test(`${r.path} ${JSON.stringify(r.query)}`, () => {
    assert.equal(snapshotKey(r.path, r.query), r.file);
  });
}
