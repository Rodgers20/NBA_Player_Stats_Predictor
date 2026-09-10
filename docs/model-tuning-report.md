# Model tuning report — 2026-09-10

Six player models were selected with expanding date validation, separately calibrated, evaluated on the final held-out dates, and refit for local production. All six selected Ridge with alpha=100 over the tested XGBoost and rolling-average candidates. Previous model files are preserved under each league’s models/backups directory.

## Measured prediction errors

Mean absolute error (MAE), in units of the predicted stat; lower is better. These results cover recorded player appearances, including low-minute games, not just players with sportsbook markets.

| League | Stat | Tuned MAE | Rolling 5 | Rolling 10 | Rolling 20 | Change vs rolling 10 | Test rows |
|---|---|---:|---:|---:|---:|---:|---:|
| NBA | PTS | 4.536 | 4.629 | 4.592 | 4.670 | 1.23% lower | 13,997 |
| NBA | AST | 1.273 | 1.297 | 1.287 | 1.307 | 1.10% lower | 13,997 |
| NBA | REB | 1.845 | 1.898 | 1.882 | 1.902 | 1.94% lower | 13,997 |
| WNBA | PTS | 4.232 | 4.355 | 4.259 | 4.310 | 0.64% lower | 2,818 |
| WNBA | AST | 1.248 | 1.300 | 1.264 | 1.257 | 1.27% lower | 2,818 |
| WNBA | REB | 1.633 | 1.689 | 1.640 | 1.652 | 0.45% lower | 2,818 |

These are modest improvements. Date-bootstrap 95% intervals support a reduction versus rolling 10 for NBA PTS/AST/REB and WNBA AST. The intervals for WNBA PTS and REB include zero: those gains are not established. This is one temporal holdout, not proof of universal superiority. Old random-split scores are not comparable.

## Uncertainty and probability diagnostics

| League | Stat | MAE difference vs rolling 10: 95% CI | Coverage of nominal 80% interval | Brier score* |
|---|---|---|---:|---:|
| NBA | PTS | [-0.0872, -0.0261] | 78.8% | 0.2412 |
| NBA | AST | [-0.0228, -0.0057] | 78.8% | 0.2264 |
| NBA | REB | [-0.0498, -0.0234] | 80.1% | 0.2343 |
| WNBA | PTS | [-0.0879, 0.0300] | 81.2% | 0.2477 |
| WNBA | AST | [-0.0285, -0.0038] | 81.5% | 0.2333 |
| WNBA | REB | [-0.0300, 0.0168] | 82.3% | 0.2373 |

*Brier diagnostics use a pregame rolling-average half-point threshold, not archived bookmaker lines. They do not establish market calibration, betting win rate, ROI, profitability, or individual-player interval coverage. No such results are claimed.

## Evaluation dates

| League | Development (selection only) | Calibration | Untouched final test |
|---|---|---|---|
| NBA | 2023-10-26 to 2025-10-23 | 2025-10-24 to 2026-02-07 | 2026-02-08 to 2026-06-13 |
| WNBA | 2024-05-16 to 2025-08-15 | 2025-08-16 to 2026-06-08 | 2026-06-09 to 2026-08-09 |

Training, calibration, and test sets never share a date. Five expanding folds select the configuration within development dates. Scaling is fitted inside each fold. Same-game PF, numeric IDs, actual minutes, and season-end defensive/pace snapshots cannot enter the feature allowlist. Same-game minutes neither filter nor weight evaluation rows. Exact duplicate raw rows are removed before rolling features; conflicting identified duplicates fail. The newer NBA cache contained 215 partial live-feed duplicates alongside game-ID box scores. The identified observations supersede those partial rows; targets were not used to choose between them. Cached engineered columns are discarded and rebuilt from raw box scores.

Production artifacts were refit on all available rows after the evaluation. The reported scores measure the earlier evaluation fit, not that final refit. Residual-based probability estimates are approximate after refitting; they remain explicitly unverified against real markets.

## Application behavior

- Next-game features include the latest completed game and exclude games on/after the target date.
- NBA recommendations use actual quoted PTS/AST/REB lines and side-specific prices. Both sides are evaluated; the best positive estimated EV per player/stat is retained. Missing markets produce no recommendation.
- EV handles pushes separately; positive American odds use the correct implied probability. Historical hit rates and charts retain low-minute outcomes.
- WNBA synthetic and uncalibrated combo props remain descriptive analysis with no price-based EV. Historical 10/10 does not become a 100% forecast. Only residual-supported quoted props receive estimated EV.
- NBA alternate-line and parlay recommendations are withheld because their prior prices/probabilities were synthetic or unvalidated. The game-score heuristic itself remains unchanged; invented numeric confidence percentages have been removed.
- Existing user prediction histories and caches were not deliberately rewritten.

## Limits and reproduction

NBA local training data ends 2026-06-13; WNBA ends 2026-08-09. No fresh data was downloaded. These artifacts cannot claim to incorporate subsequent games, trades, injuries, or role changes. A running dashboard must reload/restart to replace models already cached in memory. Historical sportsbook snapshots and prospective settled results are still required to measure betting performance.

Run the same tuning procedure with:

```sh
python3 scripts/train_improved_models.py --league nba --tune
python3 scripts/train_improved_models.py --league wnba --tune
python3 -m pytest -q
```

Environment: pandas 2.2.3, NumPy 2.2.4, scikit-learn 1.6.1. Random seed: 42. Raw source SHA-256 hashes:

- data/nba/engineered_data.parquet: `69a2aca2bd18e59d5797717ea7158048d243a924fce8357255aa326ebdbc0e44`
- data/wnba/player_game_logs.csv: `c35b7656831490ed85805078c5bdbebeb2ce7a83260bac00baaff3b3b45220d1`

Per-model JSON reports next to the artifacts contain feature names, candidate CV scores, sample counts, ranges, and reliability bins. A source-controlled snapshot is included in model-validation-results.json.

Validation completed: 120 tests passed, including temporal isolation, duplicate resolution, market arithmetic, model roundtrips, and rendering missing EV/calibration labels. Python compilation passed with macOS metadata sidecars excluded.
