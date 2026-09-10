# Honest model tuning and prop evaluation

Approved approach (2026-09-10): repair validation and market calculations before tuning.

Use only lagged player statistics and pregame schedule features. Season-end defensive,
pace, and position snapshots are excluded from training unless historical availability
can be established. Deduplicate player games before calculating lagged features.
Keep low-minute outcomes; actual minutes cannot decide retrospective eligibility.

Select models against rolling-average baselines using expanding date folds on the
earliest 60% of dates. Reserve the next 20% for residual calibration and the final
20% for one final evaluation. Fit preprocessing inside each fold. Report MAE, bias,
sample counts, date ranges, prediction-interval coverage, and probability diagnostics.
Synthetic evaluation thresholds are explicitly not historical sportsbook markets;
no betting ROI or market calibration claims can be made from those thresholds.

Save the chosen configuration refit on all available historical rows, retaining the
separate holdout report and calibration residuals. State that production refitting
changes the fitted model and that the report measures the earlier evaluation fit.
Do not claim that a more complex model always wins. Preserve existing artifacts in
a backup before replacing them. Fail clearly on invalid dates or missing features.

Build next-game lagged features including the latest completed game. Use the same
feature definitions as training. Display actual market lines and prices consistently;
exclude unpriced/synthetic lines from actionable EV rankings. Treat pushes separately.
Keep historical hit rates distinct from estimated probabilities and avoid lock claims.

Validation: temporal isolation, future-outcome perturbations, duplicate handling,
next-game feature parity, probability/push/odds arithmetic, missing markets, existing
league regression tests, and reproducible NBA/WNBA training reports.
