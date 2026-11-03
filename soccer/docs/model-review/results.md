# Prediction repair results

## Decision

Keep the existing fitted production models. Recent-form feature experiments and
simpler team attack/defence models did not demonstrate a repeatable improvement.
The production bundle's SHA-256 remains:
`33fc4690c9f6415a67d9a0ce3d55e63b0e1f10b84b9d7a94d996358a0de14827`.
It matches the migration checksum inventory; no fitted weights were replaced.

This is not an accuracy improvement claim. More varied score predictions alone
were not a reliable measure of a better model.

## What changed in the application

- The goals card displays the more likely side. Over probability .55 becomes
  **Over 2.5 goals — 55%**; over probability .43 becomes
  **Under 2.5 goals — 57%**. Exactly .50 displays both sides with no lean.
- The forecast displays its highest-probability home/draw/away result, and its
  three leading scorelines with individual probabilities. Draw is eligible.
- Score and outcome probabilities remain derived from the same normalized grid;
  under 2.5 is the complement of over 2.5.
- Forecasts show model training date, latest prior result per team, and current
  season match counts. Goal estimates are explicitly labeled as model estimates.
- Elo and head-to-head now exclude all same-date results, matching the existing
  rolling-form rule. A regression test covers same-date fixture contamination.
- Added opt-in, causal recency/season/defence/opponent-adjusted features for
  reproducible model comparisons. They are not enabled in the retained model.
- Model-dependent fixture caches now invalidate when the model bundle changes.

## Experiment design and findings

The runner is `futball/predict/evaluate_repair.py`. It writes only to an explicit
output directory and never overwrites production model files. It compares:

1. Existing tuned goal model.
2. Existing architecture plus attack/defence features and two-year sample decay.
3. A more flexible boosted model with those features and decay.
4. A regularized team/competition Poisson baseline with one-year decay.

Feature experiments test Dixon-Coles rho values -0.13, -0.05 and 0. For each
origin, fitting uses strictly earlier matches. Candidate selection uses the
2023/24 and 2024/25 origins; later diagnostics use 2025/26 and early 2026/27.
All historical periods were previously consulted, so none is represented as an
untouched final test. Closing-odds comparisons use only odds-available matches.

Development exact-score log loss (lower is better) selected the existing model
for both form-only (2.93493) and odds-informed (2.90121) forecasts. No candidate
passed the selection gate, so no production replacement was warranted.

Illustrative 2025/26 diagnostic results, form-only, 1,879 matches:

| Candidate | Exact-score log loss | Top-score accuracy | Top-three accuracy | 1–1 modal frequency |
|---|---:|---:|---:|---:|
| Existing model | 2.93318 | 12.08% | 32.52% | 75.63% |
| Recent style, rho -0.05 | 2.94298 | 11.60% | 30.49% | 65.89% |
| Flexible style, rho -0.05 | 2.94833 | 11.28% | 30.55% | 63.76% |
| Team Poisson | 2.95757 | 11.81% | 30.49% | 91.33% |

The candidate with stronger recent-form weighting did improve the early 2026/27
form-only sample, but failed on older periods. Selecting it on that small sample
would disregard the selection protocol. These results do not establish that the
current model is optimal; they reject these particular alternatives.

`evaluation.json` includes result/totals losses, goal error, draw precision/recall,
draw calibration bins and counts for every candidate and origin. The retained
baseline has no draw top picks in these form-only windows. That limitation is
reported rather than obscured with a draw quota. A synthetic low-scoring test
verifies that the serving layer can select a draw when it is most probable.

The selected model is the baseline itself, so its paired improvement and interval
are exactly zero. No statistical improvement is claimed. The runner supports
paired week-block bootstrap comparisons when a candidate passes selection.

## Data audit

Stored results contain 28,352 matches, no duplicate match keys, no future-dated
results, and no team appearing in two played matches on the same date. Therefore
the same-date feature fix does not change the historical experiment inputs.

The four current-season Barcelona results cited in the investigation match their
stored source CSV. This checks transformation integrity; it is not independent
verification of the provider's facts. Latest stored result is 2026-09-12; retained
model training data ends 2026-09-07. Serving uses newer prior results for form.

Current-season shots-on-target are missing for 5 La Liga and 7 Premier League
matches. There are no current-season Champions League results in this snapshot.
Lineups, injuries and tactical formations are not established by these result
files. The models should not be described as knowing those factors.

See `data-audit.json` for counts and source provenance. Detailed experiment
artifacts are reproducible with:

```bash
.venv/bin/python -m futball.predict.evaluate_repair --output /private/tmp/futball-model-review
```

## Final verification (September 14)

- 175 pytest tests passed, 1 skipped, 479 subtests passed. The suite includes
  other existing project work; the prediction repair cases passed.
- Real-data API smoke checks: 69 endpoints/cases, 21 prediction samples,
  and batch invariance passed.
- Production build and TypeScript passed. Lint has zero errors and six warnings
  in other existing components (unused variable and image-element warnings).
- Browser checked current scheduled Villarreal–Betis (Over 2.5, 61%) and
  Getafe–Malaga (Under 2.5, 58%), leading scores and freshness information.
- Browser-only response mocks verified precisely .55 -> Over 55%,
  .43 -> Under 57%, .50 -> no lean, and a draw as the leading result.
- At 375px width, no horizontal overflow; no JavaScript errors.
- Added the explicit loopback development origin 127.0.0.1 after the dev server
  rejected JavaScript requests from that address. The browser then loaded normally.

Browser mocks were used only inside the automated test browser; production
probabilities and data were not changed. See `browser-results.json`.
