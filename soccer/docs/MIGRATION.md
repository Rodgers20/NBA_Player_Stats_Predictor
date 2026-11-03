# Storage migration — September 13, 2026

## Everyday use

Open `/Users/rodgersbahati/Developer/Projects/FUTBALL` in VS Code.
This is the actual source directory, including the existing Git history and
uncommitted work. The frontend retains its existing nested Git repository.
`FUTBALL.code-workspace` opens this same directory.

Connect the Samsung T7. In one terminal:

```bash
cd ~/Developer/Projects/FUTBALL
make api
```

In another terminal:

```bash
cd ~/Developer/Projects/FUTBALL
make web
```

Open http://localhost:3000. Edit API keys in the root `.env` file and restart
the API. `.env` is ignored by Git and loads independently of the working
directory. Process environment variables override it.

Persistent data is under `/Volumes/T7/FUTBALL-data`: `raw`, `lake`, `models`,
and `cache`. Code, Python dependencies, frontend dependencies and build output
are on the Mac. `make data` checks the data location. Startup fails when the
data folder is unavailable; no replacement data folder is created locally.
Stop the app and data jobs before ejecting the T7 in Finder.

## Cleanup status

Migration and cleanup are complete. The original disk image is unmounted,
and both redundant external folders have been moved to the backup locations
below. The API and dashboard responded successfully with the old image
unmounted. Any terminal previously inside the old image should run
`cd ~/Developer/Projects/FUTBALL` before further commands.

## What was preserved

- The original source was copied, including both Git repositories and
  uncommitted/untracked files. Existing Developer-folder notes were retained.
- Every copied source file (excluding regenerable dependencies and caches)
  was verified by SHA-256 before migration-specific edits.
- Data files were checksum-verified. Raw data, processed tables and trained
  models were checked again before the old image was detached.
- Python was recreated at its new location with the existing dependency
  versions. Frontend dependencies were copied and production output rebuilt.
- The old `app` shortcut was removed. Its target was
  `/Volumes/FUTBALL/code/futball`; the workspace now points to `.`.
- The original disk-image folder is retained at
  `/Volumes/T7/FUTBALL-data/backups/legacy-disk-image`.
- The former `/Volumes/T7/projects/FUTBALL` notes are retained at
  `/Volumes/T7/FUTBALL-data/backups/legacy-project-notes`.

The disk image is a rollback copy, not an active dependency. No original
code or data was permanently deleted. Checksum inventories are in
`/Volumes/T7/FUTBALL-data/migration-records`.

## Validation

- 74 regression tests and 15 subtests passed.
- 69 real-data API checks and 21 prediction samples passed.
- All 15 data-quality checks passed.
- Python dependency consistency, frontend lint, TypeScript and production
  build passed.
- Browser checks passed across 11 primary routes, 33 links, all three
  competitions, mobile navigation/layout, search, CSV download, preferences,
  replay streaming and archive tabs. No JavaScript errors were reported.

This migration preserved model artifacts; it did not retrain or change their
predictions. Existing prediction limitations still apply.

## Recovering the original copy

Stop the migrated app first. The original image can be mounted with:

```bash
hdiutil attach /Volumes/T7/FUTBALL-data/backups/legacy-disk-image/futball.sparsebundle -mountpoint /Volumes/FUTBALL
```

Its original code is then at `/Volumes/FUTBALL/code/futball` and original data
at `/Volumes/FUTBALL`. Use a separate terminal without an exported
`FUTBALL_DATA_ROOT` override when running that original code. Do not overwrite
the new project with the backup: changes made after migration exist only in
the new project. Normal use does not require mounting this image.
