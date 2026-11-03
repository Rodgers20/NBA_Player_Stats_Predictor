# Prediction review and proposed repair

Status: approved and implemented. Evaluation retained the existing fitted models; see results.md for evidence and shipped changes.

## Observations from the current project

- Saved diagnostic evaluation: 2,017 matches, 1,321 home picks, 696 away picks,
  zero draw picks; actual draws 501 (24.84%). Mean predicted draw probability
  25.91%. Aggregate draw frequency is therefore not evidence of systematic
  draw-probability underestimation; conditional calibration needs examination.
- Current form-only model across 2,572 possible pairings in the three displayed
  competitions: 1,754 modal scores are 1–1 (68.20%). These are possible pairings,
  not a representative scheduled-match accuracy test. None has draw as its
  highest result probability.
- Recent stored results extend through 2026-09-12; the model metrics record a
  last training result of 2026-09-07. Serving features read newer results, while
  fitted model parameters remain from the earlier refit.
- Existing features include previous 5/10-match goals for/against, shots,
  venue form, Elo and head-to-head. They do not explicitly model player changes,
  tactical formations or pressing/high-line behavior. These should not be
  claimed from result data alone.
- The goal-model tuner selects result and over-2.5 log loss, not exact-score
  likelihood. A global Dixon-Coles correction increases 0–0 and 1–1 mass.
- Current form-only examples for today's date: Levante–Barcelona 0–2 (10.08%),
  away win 66.48%, over 2.5 63.46%; Getafe–La Coruna 1–1 (13.59%), over 2.5
  43.83%; Barcelona–Getafe 3–0 (11.84%). Scheduled dates/odds can differ.

The actual scheduled-fixture API also returns Levante–Barcelona on 2026-09-13
as 0–2 (11.33%), Barcelona win 73.46%, over 2.5 64.69%. For Getafe–La Coruna
on that date it returns 1–1 (14.86%), home 40.76%, draw 31.81%, away 27.44%,
and over 2.5 39.12%. The reported 1–1 for Levante–Barcelona is not reproduced
by the current backend; browser freshness remains part of the investigation.

## Alternatives

1. Presentation-only: show multiple score probabilities and clearer labels.
   Helps interpretation but does not address predictive weaknesses.
2. Recommended: audit data freshness and provenance, improve measured recent
   attack/defence features, compare score-distribution models chronologically,
   and improve probability presentation. Preserves the existing API architecture.
3. Full lineup/tactical model: requires reliable historical pre-match lineup,
   injury, manager and tactical data with timestamps. Defer until that evidence
   exists; club-name rules are not a substitute.

## Recommended design

Keep the current production model as the baseline and rollback artifact. Run
candidate experiments separately from production data/model publication.

Audit current-season source coverage, team-name mappings, missing measurements,
fixture dates/odds and model freshness. Verify the cited recent results against
their source data before using them as evidence. Surface the latest result date,
training cutoff and number of available prior games in each forecast.

Compare the current features with time-decayed attack and defence, current-season
scoring/conceding with shrinkage toward longer history, shots on target against,
clean sheets/failure-to-score and opponent/venue-adjusted form. Use only matches
strictly before kickoff. Do not force monotonic reactions to a small scoring streak
or label teams permanently defensive/attacking. Include missing-data indicators.

Benchmark the current goal model against a regularized time-decayed team
attack/defence Poisson baseline and an enhanced feature-based candidate. Tune
score likelihood alongside outcome and totals quality; test low-score correction
strength instead of manually reducing 1–1. Examine whether a more flexible score
distribution improves diagnostics before increasing model complexity.

Evaluate with rolling chronological origins. Select on development windows,
then report a separately reserved later window. Previously consulted historical
periods remain diagnostic, not untouched evidence. Report exact-score log loss,
top-one/top-three score accuracy, goal error, outcome/over-2.5 log loss, draw
calibration by probability bin, draw precision/recall, and counts. Compare on
identical available-information subsets and include uncertainty in differences.
Promote only a candidate with repeatable score-distribution improvement and no
material outcome/totals regression; otherwise retain the baseline and document it.

Result picks use the maximum of home/draw/away probabilities, with draw eligible.
Do not impose a draw quota. Show the leading three scorelines and probabilities,
explain that one scoreline differs from the sum of all win or over outcomes,
and label model goal expectations clearly. Keep all served probabilities derived
from the same normalized score distribution.

Tests cover date leakage, fixture batch invariance, unknown teams/missing history,
draw selection, score-ranking and total-probability consistency. Verify the actual
scheduled-fixture API and browser view before final publication. Preserve existing
user edits and API keys. No accuracy guarantee is implied by passing software tests.

Probability-evaluation reference: [scikit-learn calibration documentation](https://sklearn.org/stable/modules/calibration.html). Proper scoring rules and calibration curves answer different questions from top-pick accuracy.
