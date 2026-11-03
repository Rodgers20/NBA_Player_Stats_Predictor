"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { CalendarClock } from "lucide-react";
import { api } from "@/lib/api";
import { Panel } from "@/components/shell/Panel";
import { displayTeam, fmtDate, shortDate } from "@/lib/format";
import { Select } from "@/components/ui/Select";
import { TeamCrest } from "@/components/TeamCrest";
import { CountMarketCard } from "@/components/predict/CountMarketCard";
import { WhyTheModelPicked } from "@/components/predict/WhyTheModelPicked";
import { MatchdayBoard } from "@/components/predict/MatchdayBoard";
import { CompetitionCrest } from "@/components/CompetitionCrest";
import { kickoffTime } from "@/lib/format";
import type { BacktestRow, League } from "@/lib/types";

export function PredictionsPanel({
  leagues, initialCompetition,
}: { leagues: League[]; initialCompetition: string }) {
  const router = useRouter();
  const code = initialCompetition;
  const [requestedFixture, setFixtureKey] = useState<string>("");

  const fixtures = useQuery({
    queryKey: ["predict-fixtures", code],
    queryFn: () => api.predictFixtureList(code, 21),
  });

  const fixtureKey = fixtures.data?.some(f => f.fixture_key === requestedFixture)
    ? requestedFixture : fixtures.data?.[0]?.fixture_key ?? "";

  const prediction = useQuery({
    queryKey: ["predict-fixture", fixtureKey],
    queryFn: () => api.predictByFixture(fixtureKey),
    enabled: Boolean(fixtureKey),
  });

  const metrics = useQuery({ queryKey: ["predict-metrics", code], queryFn: () => api.predictMetrics(code) });
  const backtest = useQuery({
    queryKey: ["predict-backtest", code],
    queryFn: () => api.predictBacktest(code, 12),
  });

  const p = fixtures.data?.some(f => f.fixture_key === fixtureKey) ? prediction.data : undefined;
  const maxBar = useMemo(
    () => Math.max(...(p?.total_goals.distribution.map((d) => d.probability) ?? [1])),
    [p],
  );
  const selected = fixtures.data?.find((f) => f.fixture_key === fixtureKey);
  const over = p?.total_goals.over_2_5 ?? 0.5;
  const goalsLabel = over === 0.5 ? "Over / Under 2.5 goals" : over > 0.5 ? "Over 2.5 goals" : "Under 2.5 goals";
  const goalsProbability = Math.max(over, 1 - over);
  const resultPick = p?.result_pick ?? (p ? (Object.entries(p.win).sort((a, b) => b[1] - a[1])[0][0] as "home" | "draw" | "away") : undefined);

  return (
    <div className="space-y-4">
      <Panel title="This matchday">
        <MatchdayBoard competitionCode={code} />
      </Panel>

      <Panel title="Fixture Predictor">
        <div className="grid gap-3 sm:grid-cols-[minmax(0,240px)_minmax(0,1fr)]">
          <Field label="Competition">
            <Select
              ariaLabel="Competition"
              value={code}
              onChange={(next) => { setFixtureKey(""); router.push(`/predictions?competition=${encodeURIComponent(next)}`); }}
              options={leagues.map((l) => ({
                value: l.code, label: l.name, hint: l.country,
                icon: <CompetitionCrest code={l.code} size={16} />,
              }))}
            />
          </Field>
          <Field label="Fixture">
            {fixtures.isLoading ? (
              <div className="rounded-lg border border-stroke bg-elevated px-2.5 py-2 text-[12.5px] text-ink-muted">
                Loading fixtures…
              </div>
            ) : fixtures.isError ? (<p role="alert" className="text-sm text-live">{fixtures.error.message}</p>) : (fixtures.data?.length ?? 0) === 0 ? (
              <div className="flex items-center gap-2 rounded-lg border border-stroke bg-elevated px-2.5 py-2 text-[12.5px] text-ink-muted">
                <CalendarClock className="w-3.5 h-3.5" aria-hidden />
                No scheduled fixtures in the next three weeks for this competition.
              </div>
            ) : (
              <Select
                ariaLabel="Fixture"
                value={fixtureKey}
                onChange={setFixtureKey}
                options={fixtures.data!.map((f) => ({
                  value: f.fixture_key,
                  label: `${displayTeam(f.home_team)} v ${displayTeam(f.away_team)}`,
                  labelNode: <FixtureLabel home={f.home_team} away={f.away_team} />,
                  hint: `${fmtDate(f.date)} · ${kickoffTime(f.kickoff_utc, f.date, f.time)}`,
                }))}
              />
            )}
          </Field>
        </div>

        {prediction.isFetching && fixtureKey && <p role="status" className="mt-3 text-sm text-ink-secondary">Calculating forecast…</p>}
        {prediction.isError && (
          <p role="alert" className="mt-3 text-[12px] text-live">{(prediction.error as Error).message}</p>
        )}

        {p && (
          <div className="mt-4 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
              <h3 className="flex min-w-0 items-center gap-1.5 font-display text-[14px] font-semibold sm:text-[15px]">
                <TeamCrest name={p.home_team} size={20} />
                <span className="truncate" title={p.home_team}>{displayTeam(p.home_team)}</span>
                <span className="text-ink-muted">v</span>
                <TeamCrest name={p.away_team} size={20} />
                <span className="truncate" title={p.away_team}>{displayTeam(p.away_team)}</span>
              </h3>
              <span className="whitespace-nowrap text-[11px] text-ink-muted">
                {selected ? `${shortDate(selected.date)}${selected.time ? ` · ${selected.time}` : ""}` : ""}
              </span>
            </div>
            {p.fixture?.referee && (
              <p className="-mt-2 text-[10.5px] text-ink-muted">Referee: {p.fixture.referee}</p>
            )}

            <div>
              <div className="mb-1.5 flex flex-wrap gap-2 items-baseline justify-between">
                <h4 className="text-[11px] uppercase tracking-wider text-ink-muted">Match Result</h4>
                <span className="text-[10.5px] italic text-ink-muted">{p.basis}</span>
              </div>
              <div className="flex h-8 overflow-hidden rounded-lg">
                <Segment value={p.win.home} color="var(--color-home)" />
                <Segment value={p.win.draw} color="var(--color-form-d)" />
                <Segment value={p.win.away} color="var(--color-away)" />
              </div>
              <div className="mt-1.5 flex justify-between text-[11px] text-ink-secondary">
                <span>{displayTeam(p.home_team)} {(p.win.home * 100).toFixed(0)}%</span>
                <span>Draw {(p.win.draw * 100).toFixed(0)}%</span>
                <span>{displayTeam(p.away_team)} {(p.win.away * 100).toFixed(0)}%</span>
              </div>
            </div>

            {resultPick && <p className="text-sm text-ink-secondary">
              Most likely outcome: <strong className="text-ink">{resultPick === "draw" ? "Draw" : displayTeam(resultPick === "home" ? p.home_team : p.away_team)}</strong>
              {` · ${(p.win[resultPick] * 100).toFixed(0)}%`}
            </p>}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Model goal estimate" value={p.expected_goals.total.toFixed(2)}
                    sub={`${p.expected_goals.home.toFixed(2)} – ${p.expected_goals.away.toFixed(2)}`} />
              <Stat label={goalsLabel} value={`${(goalsProbability * 100).toFixed(0)}%`}
                    sub={over === 0.5 ? "Even chance · no lean" : `Other side ${((1 - goalsProbability) * 100).toFixed(0)}%`} />
              <Stat label="Both to score" value={`${(p.total_goals.btts * 100).toFixed(0)}%`} sub="BTTS" />
              <Stat label="Likeliest score"
                    value={`${p.most_likely_score.home}–${p.most_likely_score.away}`}
                    sub={`${(p.most_likely_score.probability * 100).toFixed(1)}%`} />
            </div>

            <div className="rounded-lg border border-stroke bg-elevated p-3">
              <h4 className="text-[11px] uppercase tracking-wider text-ink-muted">Leading scorelines</h4>
              <div className="mt-2 flex flex-wrap gap-x-6 gap-y-2">
                {(p.top_scores ?? [p.most_likely_score]).map(score => (
                  <span key={`${score.home}-${score.away}`} className="font-mono text-sm text-ink">
                    {score.home}–{score.away} <span className="text-ink-secondary">{(score.probability * 100).toFixed(1)}%</span>
                  </span>
                ))}
              </div>
              <p className="mt-2 text-xs text-ink-muted">Each score is one possibility. A win or goals forecast combines many scorelines, so its probability can be much higher.</p>
            </div>

            <div className="grid gap-3 lg:grid-cols-2">
              <CountMarketCard title="Corners" market={p.corners} unit="corners" />
              <CountMarketCard title="Shots on target" market={p.sot} unit="on target" />
            </div>

            <WhyTheModelPicked prediction={p} />
            {p.context && <div className="text-xs text-ink-muted space-y-1">
              <p>Model trained through {p.context.model_trained_through ?? "unknown"}. Recent form uses results before {p.context.forecast_date}.</p>
              <p>{displayTeam(p.home_team)}: latest result {p.context.home_last_result ?? "unavailable"} · {p.context.home_season_matches ?? 0} season matches.
                {" "}{displayTeam(p.away_team)}: latest result {p.context.away_last_result ?? "unavailable"} · {p.context.away_season_matches ?? 0} season matches.</p>
            </div>}
            <div>
              <h4 className="mb-2 text-[11px] uppercase tracking-wider text-ink-muted">
                Total goals distribution
              </h4>
              <div className="flex items-end gap-1.5">
                {p.total_goals.distribution.map((d) => (
                  <div key={d.goals} className="flex flex-1 flex-col items-center gap-1">
                    <span className="font-mono text-[9px] tnum text-ink-muted">
                      {(d.probability * 100).toFixed(0)}%
                    </span>
                    <div className="flex h-20 w-full items-end">
                      <div
                        className="w-full rounded-t transition-[height] duration-500"
                        style={{
                          height: `${Math.max((d.probability / maxBar) * 100, 2)}%`,
                          background: "var(--color-series-3)",
                        }}
                      />
                    </div>
                    <span className="font-mono text-[10px] tnum text-ink-secondary">{d.goals === 7 ? "7+" : d.goals}</span>
                  </div>
                ))}
              </div>
            </div>

            {p.fixture?.odds?.home && (
              <p className="text-[11px] text-ink-muted">
                Market odds for this fixture: {p.fixture.odds.home} / {p.fixture.odds.draw} /{" "}
                {p.fixture.odds.away} — used as inputs to the goal model.
              </p>
            )}
          </div>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Model Performance">
          {metrics.data ? (
            <div className="space-y-3">
              {metrics.data.competition_metrics && (
                <div className="rounded-lg border border-pitch/25 bg-pitch/5 p-4" data-testid="competition-performance">
                  <h3 className="text-sm font-semibold">{leagues.find(l => l.code === code)?.name ?? code} · served forecast</h3>
                  {metrics.data.competition_metrics.n > 0 ? <>
                    <dl className="mt-3 grid grid-cols-2 gap-3 text-xs">
                      <div><dt className="text-ink-secondary">Accuracy</dt><dd className="mt-1 font-mono text-lg">{((metrics.data.competition_metrics.accuracy ?? 0) * 100).toFixed(1)}%</dd></div>
                      <div><dt className="text-ink-secondary">Log-loss</dt><dd className="mt-1 font-mono text-lg">{metrics.data.competition_metrics.logloss?.toFixed(4)}</dd></div>
                    </dl>
                    <p className="mt-3 text-xs leading-relaxed text-ink-secondary">{metrics.data.competition_metrics.n.toLocaleString()} held-out matches · {metrics.data.competition_metrics.from} to {metrics.data.competition_metrics.through}. Historical diagnostic; performance varies by competition.</p>
                  </> : <p className="mt-2 text-xs text-ink-secondary">No held-out predictions are available for this competition.</p>}
                </div>
              )}
              <h3 className="text-xs font-semibold text-ink-secondary">All training competitions</h3>
              <table className="w-full text-[12px] tnum">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wider text-ink-muted">
                    <th className="py-1 text-left font-medium">Forecast</th>
                    <th className="py-1 text-right font-medium">Accuracy</th>
                    <th className="py-1 text-right font-medium">Log-loss</th>
                  </tr>
                </thead>
                <tbody>
                  <Row label="Closing-odds market" a={metrics.data.market_accuracy} l={metrics.data.market_logloss} />
                  <Row label="Served forecast" a={metrics.data.blend_accuracy} l={metrics.data.blend_logloss} />
                  <Row label="Model (form only)" a={metrics.data.formonly_accuracy} l={metrics.data.formonly_logloss} />
                  <Row label="Over 2.5 (odds model)" a={metrics.data.over25_accuracy} l={metrics.data.over25_logloss} />
                </tbody>
              </table>
              {metrics.data.baseline_logloss != null && <p className="text-xs text-ink-secondary">Historical-frequency baseline log-loss: {metrics.data.baseline_logloss.toFixed(4)}. Lower is better.</p>}
              <dl className="grid grid-cols-2 gap-y-1 text-[11.5px]">
                <dt className="text-ink-muted">Training matches</dt>
                <dd className="text-right tnum">{metrics.data.n_train.toLocaleString()}</dd>
                <dt className="text-ink-muted">Held-out matches</dt>
                <dd className="text-right tnum">{metrics.data.n_test.toLocaleString()}</dd>
                {metrics.data.n_features != null && (
                  <>
                    <dt className="text-ink-muted">Features</dt>
                    <dd className="text-right tnum">{metrics.data.n_features}</dd>
                  </>
                )}
                {metrics.data.dixon_coles_rho != null && (
                  <>
                    <dt className="text-ink-muted">Dixon-Coles ρ</dt>
                    <dd className="text-right tnum">{metrics.data.dixon_coles_rho.toFixed(2)}</dd>
                  </>
                )}
              </dl>
              {metrics.data.feature_groups && (
                <div className="flex flex-wrap gap-1">
                  {metrics.data.feature_groups.map((g) => (
                    <span key={g} className="rounded px-1.5 py-0.5 text-[10px] font-medium bg-elevated text-ink-secondary">
                      {g}
                    </span>
                  ))}
                </div>
              )}
              <p className="text-xs leading-relaxed text-ink-secondary">Market scores use {metrics.data.market_comparison_n?.toLocaleString() ?? "odds-available"} matches with odds. The served forecast’s log-loss on those same matches is {metrics.data.served_logloss_odds_subset?.toFixed(4) ?? "not recorded"}.</p>
              <details className="rounded-lg border border-stroke p-3">
                <summary className="cursor-pointer text-sm text-ink-secondary">Evaluation method and calibration</summary>
                <p className="my-3 text-xs leading-relaxed text-ink-secondary">{metrics.data.methodology_note}</p>
                {metrics.data.brier_score != null && <p className="mb-3 text-xs text-ink-secondary">Multiclass Brier score: {metrics.data.brier_score.toFixed(4)} (lower is better).</p>}
                <table className="w-full text-xs"><thead><tr><th className="text-left">Matches</th><th className="text-right">Mean confidence</th><th className="text-right">Actual accuracy</th></tr></thead><tbody>{metrics.data.calibration?.map(bin => <tr key={bin.from} className="border-t border-stroke"><td className="py-2">{bin.n}</td><td className="text-right">{(bin.confidence*100).toFixed(1)}%</td><td className="text-right">{(bin.accuracy*100).toFixed(1)}%</td></tr>)}</tbody></table>
              </details>
              <p className="rounded-lg bg-void/60 p-3 text-[11.5px] leading-relaxed text-ink-secondary">
                {metrics.data.verdict}
                
                <span className="mt-2 block">Latest training result: {metrics.data.last_training_result ?? "Not recorded"}. Probabilities express uncertainty; the likeliest score is still only one possible outcome.</span>
              </p>
            </div>
          ) : (
            <p className="py-8 text-center text-[12px] text-ink-muted">
              {metrics.isError ? (metrics.error as Error).message : "Loading…"}
            </p>
          )}
        </Panel>

        <Panel title="Held-out match predictions">
          {backtest.isLoading ? (
            <p className="py-8 text-center text-[12px] text-ink-muted">Loading…</p>
          ) : backtest.isError ? (<p role="alert" className="py-6 text-sm text-live">{backtest.error.message}</p>) : (backtest.data?.length ?? 0) === 0 ? (
            <p className="py-8 text-center text-[12px] text-ink-muted">
              No completed matches to backtest for this competition.
            </p>
          ) : (
            <>
              <p className="mb-2 text-[11px] text-ink-muted">
                These predictions were saved from models trained only on matches before the evaluation window. They are historical diagnostics, not live betting results.
                {" "}
                <span className="text-ink-secondary">
                  {backtest.data!.filter((b) => b.correct).length}/{backtest.data!.length} correct.
                </span>
              </p>
              <ul className="max-h-[320px] space-y-1.5 overflow-y-auto pr-1">
                {backtest.data!.map((b, i) => (
                  <BacktestRowItem key={i} row={b} />
                ))}
              </ul>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}

function Row({ label, a, l }: { label: string; a: number; l: number }) {
  return (
    <tr className="border-t border-stroke/60">
      <td className="py-1.5 text-ink-secondary">{label}</td>
      <td className="py-1.5 text-right">{(a * 100).toFixed(1)}%</td>
      <td className="py-1.5 text-right font-mono">{l.toFixed(4)}</td>
    </tr>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[10px] uppercase tracking-wider text-ink-muted">{label}</span>
      {children}
    </label>
  );
}

function Segment({ value, color }: { value: number; color: string }) {
  return (
    <div
      className="grid place-items-center text-[10.5px] font-semibold text-[#04160C] transition-[flex-grow] duration-500"
      style={{ flexGrow: Math.max(value, 0.02), background: color }}
    >
      {value > 0.12 && `${(value * 100).toFixed(0)}%`}
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg bg-elevated/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-ink-muted">{label}</div>
      <div className="mt-1 font-display text-[20px] font-bold tnum">{value}</div>
      {sub && <div className="mt-0.5 font-mono text-[10.5px] tnum text-ink-muted">{sub}</div>}
    </div>
  );
}


/**
 * One graded historical prediction.
 *
 * This was a five-column table with a 440px floor, which on a phone meant
 * scrolling the whole page sideways to read the pick. Clubs are their crests
 * here — at this size a name is two wrapped lines that the badge already says.
 */
function BacktestRowItem({ row }: { row: BacktestRow }) {
  const leanedOver = row.over_2_5 > 0.5;
  const leanCorrect = row.actual_over_2_5 === leanedOver;
  return (
    <li className="rounded-md bg-elevated px-2.5 py-2">
      <div className="flex items-center gap-2">
        <span className="w-11 shrink-0 text-[10px] tnum text-ink-muted">{shortDate(row.date)}</span>
        <span className="flex min-w-0 flex-1 items-center gap-1.5" title={`${row.home_team} v ${row.away_team}`}>
          <TeamCrest name={row.home_team} size={16} />
          <span className="font-mono text-[12px] tnum text-ink">
            {row.actual_home}&ndash;{row.actual_away}
          </span>
          <TeamCrest name={row.away_team} size={16} />
        </span>
        <span
          className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium"
          style={
            row.correct
              ? { background: "color-mix(in oklab, var(--color-form-w) 20%, transparent)", color: "var(--color-form-w)" }
              : { background: "color-mix(in oklab, var(--color-form-l) 16%, transparent)", color: "var(--color-form-l)" }
          }
        >
          {row.predicted}
        </span>
      </div>
      <p className="mt-1 text-[10px] text-ink-muted">
        <span className={leanCorrect ? "text-ink-secondary" : undefined}>
          {row.over_2_5 === 0.5 ? "O/U" : leanedOver ? "Over" : "Under"} 2.5 ·{" "}
          {(Math.max(row.over_2_5, 1 - row.over_2_5) * 100).toFixed(0)}%
        </span>
        {" — actual "}
        {row.actual_over_2_5 ? "over" : "under"}
      </p>
    </li>
  );
}


/**
 * A fixture with each club's crest immediately before its own name.
 *
 * The option used to carry the home crest alone, so the away side had no badge
 * anywhere — including the closed trigger, which is where a reader confirms
 * what they picked. Putting both badges together at the left fixed the absence
 * but not the reading: a pair of crests ahead of "Malaga v Villarreal" belongs
 * to neither club. Each badge now sits with the name it identifies.
 */
function FixtureLabel({ home, away }: { home: string; away: string }) {
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <TeamCrest name={home} size={16} />
      <span className="truncate">{displayTeam(home)}</span>
      <span className="shrink-0 text-ink-muted">v</span>
      <TeamCrest name={away} size={16} />
      <span className="truncate">{displayTeam(away)}</span>
    </span>
  );
}
