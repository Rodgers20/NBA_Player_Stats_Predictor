# Approved prediction repair: implementation plan

Approved by the user, including showing the more likely over/under 2.5 side.
The optional writing-plans skill is not installed; this file records the plan.

1. Preserve baseline model artifacts; audit raw-to-processed results and dates.
2. Add causal recent attack/defence features without altering existing feature columns.
3. Run rolling-origin comparisons with a fixed candidate set and score likelihood,
   outcome/totals loss, draw calibration, and score-ranking metrics. Keep the current
   model unless evidence supports promotion. Historical windows are diagnostic.
4. Return top-three scorelines, result pick, data/training dates, and history counts.
5. Show Over 2.5 or Under 2.5 with complementary probabilities; handle 50/50 explicitly.
6. Test temporal isolation, probability consistency, API and rendered browser behavior.
7. Record experiment results, chosen model decision and limitations; restart the app.
