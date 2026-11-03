# FUTBALL reliability and design review — 2026-09-12

Direction approved by the user: preserve the dark football dashboard, improve usability and repair model evaluation before tuning. Alternatives considered: cosmetic-only polish leaves reliability defects; a full rebuild adds avoidable scope.

## Design

- Preserve Archivo/Public Sans/Plex Mono typography and stadium palette. Improve muted-text contrast, focus visibility, touch targets and narrow-screen layouts.
- Provide keyboard-operable mobile navigation, team search, CSV season reports, schedule updates and persistent table-density preferences. Replace the inactive upgrade promotion with access to model evidence.
- Exclude unplayed and same-day matches from rolling form history. Derive result probabilities, scorelines, totals and BTTS from one normalized goal distribution, including a 7+ total-goal bin.
- Save held-out predictions before the production refit. Clearly disclose the reused historical test window. Tune only on validation; report baseline log-loss, multiclass Brier score, confidence calibration and comparisons on matching odds-available rows.
- Reject unknown fixtures and consistently return API error envelopes. Keep errors distinct from empty data in prediction panels.

## Implementation and verification sequence

1. Add model regression tests; fix fixture contamination and probability consistency.
2. Implement frozen evaluation artifacts and diagnostic metrics.
3. Repair shell interactions and unfinished pages; preserve existing uncommitted user work.
4. Refresh available source data, evaluate fixed configurations on validation, retain the better configuration and refit.
5. Run model/API tests, TypeScript, production build, route/link and browser checks at desktop and mobile sizes.
6. Record measured outcomes and remaining provider/data limitations. Do not claim independent predictive certification from the reused historical hold-out.
