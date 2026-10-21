# Game forecasting audit and release — October 10, 2026

The old game engine was a hand-weighted heuristic, with its XGBoost weight set to zero. Its rest calculation used the gap between the previous two games rather than the scheduled game and last completed game; pace was always the default, WNBA predictions skipped injuries, and training/runtime feature definitions disagreed. Player sums could include duplicate WNBA rows. Those paths have been corrected or replaced by the shared pregame feature/artifact pipeline.

## Measured release

| League | Held-out games | Winner accuracy | Margin MAE | Total MAE | Winner log loss |
|---|---:|---:|---:|---:|---:|
| NBA | 713 | 70.7% | 11.90 | 14.80 | 0.5779 |
| WNBA | 195 | 69.7% | 9.72 | 14.12 | 0.5733 |

MAE is average absolute error in points. NBA holdout: January 15–June 13, 2026. WNBA holdout: July 4–October 7, 2026. These are fixed temporal backtests, not live betting records. Full machine-readable reports and individual game predictions are in `docs/game-models/`.

Candidates include league mean, core/form Ridge, richer regularized Ridge, histogram gradient boosting, XGBoost and feature ablations. Three earlier date-grouped forward folds select margin and total models independently; all final-date games remain together. Imputation/scaling fits only training folds. Probability scale and 80% interval offsets come from earlier out-of-fold residuals. The final holdout is excluded from parameter/model selection. After evaluation, the chosen model is refitted on completed history for future forecasts.

NBA selected regularized full-feature Ridge (alpha 1000) for both outcomes. Its total MAE improved from the development-selected core baseline's 15.06 to 14.80; the date-bootstrap 95% interval for the MAE difference is approximately −0.47 to −0.03 points. Its margin MAE of 11.90 versus 11.84 does **not** demonstrate better margin accuracy. WNBA selected core Ridge (alpha 100 margin, alpha 10 total); richer candidates did not earn selection. Promotion checks enforce no more than 2% degradation versus the selected simple baseline; passing is not proof of superiority.

## Inputs and source quality

The shared feature builder evaluates home/away, 5/10/20-game offense and defense per estimated possession, pace normalized for overtime, recent margins, home/road form, season-regressed efficiency, shooting efficiency, turnovers, offensive rebounding, free-throw rate, Elo/opponent strength, rest, back-to-backs and games in the preceding four/seven days. Geography has deliberately limited NBA arena-city approximations and explicit missingness; it is not observed flight or sleep data. WNBA travel and unknown/neutral venues remain missing rather than fabricated. An exact arena-specific effect has not been established.

Official free `LeagueGameLog` team records replaced incomplete player-derived team scores: 3,941 NBA games across 2023–24 through 2025–26, and initially 916 WNBA games across 2024–26. Ten missing NBA regular-season games were supplemented from existing cached direct Kaggle team totals with explicit provenance. WNBA operational history was subsequently refreshed through October 9 (918 games); the published evaluation and fitted artifact retain their October 7 cutoff. History refresh merges recent seasons by default rather than discarding older training records.

NBA preseason/exhibition games are excluded from training. Current preseason forecasts are explicitly qualified as outside the validated regular/playoff sample. Current injuries are applied to both leagues as scenario adjustments using player production, minutes, playmaking/rebounding and defensive proxies, with recent participation accounting to limit double counting. These adjustments are **not historically validated**: there is no complete timestamped pregame injury archive. Stale player histories and unmatched/unknown data remain visible. No historical absence is inferred from a future box score.

## Honest performance and operations

The Model Performance page separates fixed holdout evaluation, timestamp-verified legacy saved calls and current-version prospective forecasts. The new forecast ledger stores only a first-published prediction with an explicit future tipoff, model version and current quote snapshot. Grading attaches completed team scores without altering the original forecast. Static export records and grades the ledger; a static site still requires refreshed data and redeployment. Local workflow changes require pushing and configured secrets before unattended production refresh can be claimed.

No comprehensive historical pregame sportsbook quote archive was available. Therefore ATS, over/under profitability, ROI, closing-line value and superiority to betting markets remain unmeasured. More inputs did not automatically improve forecasts, and these results cannot establish that this model beats every competing model.

## Primary references

- Free official endpoint client contract: https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguegamelog.md
- Advanced team box scores: https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/boxscoreadvancedv3.md
- Official NBA injury reports: https://official.nba.com/nba-injury-report-2025-26-season/
- Official WNBA injury reports: https://www.wnba.com/wnba-injury-report
- Historical odds access and coverage: https://the-odds-api.com/historical-odds-data/
- Temporal validation: https://scikit-learn.org/1.6/modules/cross_validation.html#time-series-split
- Ridge: https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Ridge.html
- Histogram boosting: https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- XGBoost 2.1.4 fixes compatibility with scikit-learn 1.6: https://xgboost.readthedocs.io/en/stable/changes/v2.1.0.html
