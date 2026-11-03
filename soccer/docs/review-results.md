# FUTBALL review results — 12 September 2026

The dashboard and prediction pipeline were repaired and validated on the T7-hosted project. Existing source files and model/data artifacts were backed up before replacement. Changes are left uncommitted so the pre-existing frontend work remains available for review.

## Design and functionality

- Retained the dark stadium palette and existing typefaces; improved muted-text contrast, heading visibility, focus indicators, touch targets and mobile prediction-card layout.
- Added working mobile navigation and team search. Competition selection in the predictor now updates the URL and the surrounding app.
- Replaced placeholder Reports with season summaries and CSV exports; Alerts now shows schedule updates. Settings saves table-density preferences on the device.
- Replaced the inactive upgrade promotion with a link to model performance. Removed the inactive theme button; the supported dark appearance is described in Settings.
- Added a retryable application error boundary and explicit prediction loading/error states.
- Routed browser API, image and replay requests through the dashboard origin, including WebSocket upgrades. Image URLs now match between server and client rendering.
- Fixed dropdown clipping and keyboard state handling. 3D views detect WebGL before mounting, offer 2D fallbacks, and render on demand.

## Prediction fixes and evaluation

Unplayed fixtures previously occupied rolling-history slots and counted as losses. They no longer alter team form, rest days or season averages. Date batches exclude same-day results from rolling statistics. A real-data check confirms that an individual fixture receives the same features when built alone or in a batch.

All served result probabilities, totals, BTTS and scorelines now derive from the same normalized Dixon–Coles goal grid. Valid odds are inputs to the goal regressors; the independent classifier blend no longer overrides win probabilities. The total-goals chart includes a 7+ bin, preserving the probability tail. Scheduled fixture dates are used for rest-day features; unknown fixtures and teams are rejected.

Held-out predictions are saved **before** fitting the production models on all results. The backtest reads those saved predictions, rather than evaluating the all-data production model on its own training matches.

A bounded validation search compared the incumbent with more strongly regularized configurations, using combined 1X2 and over-2.5 log-loss. Selected configurations use 150 boosting iterations for form-only and 200 for odds-informed goals, with rho −0.10 and −0.13 respectively. Iteration counts remain those selected on validation.

| Historical diagnostic | Accuracy | Log-loss |
|---|---:|---:|
| Frequency baseline, all 2,017 evaluation matches | 44.2% | 1.0700 |
| Form-only goal model, same 2,017 matches | 52.2% | 0.9907 |
| Served forecast, all 2,017 matches | 52.8% | 0.9822 |
| Served forecast, 1,875 matches with odds | — | 0.9814 |
| Closing-odds market, same 1,875 matches | 53.2% | 0.9796 |

The served forecast uses odds-informed goals where odds exist and form-only goals otherwise. Multiclass Brier score: **0.5847**. The UI exposes sample counts, calibration bins, methodology, and the latest training-result date (**7 September 2026**).

A separate audit boundary of **25 August 2026** was fixed before the refresh. The 95 newly fetched evaluation matches after that boundary produced **53.7% accuracy** and **0.9935 log-loss**. This is a small, provisional sample. No settings were selected on it. The older historical test window had been consulted during earlier development and again during these repairs, so it is explicitly described as diagnostic rather than untouched.

These results beat a simple frequency baseline but do **not** demonstrate superiority over the closing-odds market or guarantee future predictive accuracy. Closing odds are not necessarily available days before kick-off. A most-likely scoreline remains just one possible outcome.

## Data and repeatable checks

The refresh retrieved all 10 requested result files, 190 rolling fixture rows, and five league season schedules. The results layer now contains 28,322 matches across six competitions used for training. Fixed redirect handling and made `make refresh` refresh fixture schedules as well as results.

Verified:

- 19 regression tests passed, covering models, refresh, cache invalidation, atomic writes, scoped metrics, and replay lifecycle.
- 15/15 event-data quality checks passed, including score reconciliation.
- 69 API checks passed across Premier League, La Liga and Champions League; 21 sample predictions passed probability, range and ordering checks.
- TypeScript and production build passed; ESLint passed with zero errors and warnings.
- All 11 main routes returned 200; 33 distinct internal links had no failures.
- Additional browser checks covered five views each for La Liga and Champions League.
- Team search, CSV download, saved preferences, mobile navigation and all seven archive tabs worked.
- Desktop (1440px) and mobile (375px) checks found no horizontal overflow on the sampled mobile routes; no uncaught JavaScript or hydration errors were observed.
- Replay delivered sequenced messages through the same-origin WebSocket proxy. A browser with WebGL disabled rendered the fallback without mounting a canvas.

Run `make check` for tests, data quality, lint, TypeScript and a production build. Run `.venv/bin/python tests/api_smoke.py` for the real-data API checks. `make train` refits the saved configuration; `make tune` searches on validation before refitting. The API detects replaced result, fixture and model files automatically; date-dependent prediction caches also expire at calendar rollover. Writers publish these files atomically so readers do not see partial output.

## Remaining limits

Live indicators on league pages are inferred from schedules; verified in-play scores require provider support. Match updates are an on-page schedule view, not background notifications. Club/player images can fall back to initials when providers have no image. Champions League and player-stat coverage depends on the available upstream data; absent data is not invented.

Code backups: `review-backup-20260912/` in the repository. Data/model backups: `/Volumes/FUTBALL/review-backup-20260912/`. Current metrics, calibration and frozen predictions: `/Volumes/FUTBALL/models/predict/`.


## Follow-up reliability pass

Two reproduced defects were repaired: cached results stayed unchanged after replacing their file, and a disconnected idle WebSocket remained blocked on its message queue. File-version-aware caches now reload automatically. Failed artifact writes preserve the previous file. Replay handlers monitor both incoming disconnects and outgoing messages, cancel pending tasks on exit, and unsubscribe promptly.

Replay sequence numbers now remain monotonic across restarts. A slow consumer is asked to reconnect instead of silently losing its subscription. Requests for expired history or an unknown match are explicitly rejected. The browser suppresses duplicate messages, avoids callbacks from obsolete sockets, and waits for a new playback to start before subscribing to its backlog.

The prediction page now separates the selected competition’s held-out results from the all-competition aggregate:

| Displayed competition | Held-out matches | Served accuracy | Log-loss |
|---|---:|---:|---:|
| Premier League | 407 | 48.2% | 1.0209 |
| La Liga | 416 | 52.2% | 0.9699 |
| Champions League | 142 | 54.9% | 0.9931 |

These use the existing frozen evaluation predictions. This pass did not retune on these test results or change model weights. The league breakdown makes weaker performance visible rather than hiding it behind the aggregate score.

Follow-up verification: 19 backend tests, the 69-check real-data API smoke suite, lint and production build passed. Chrome verified all three competition summaries, the 375px mobile layout, two replay start/stop cycles, and reconnect sequencing (12 → 13). All disconnected subscribers were released, with no JavaScript or hydration errors observed.
