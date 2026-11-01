"use client";

import { useQuery } from "@tanstack/react-query";
import { cbbApi } from "@/lib/api";
import { mark, pct } from "@/lib/cbb.mjs";
import type { CbbCalibrationBin, CbbModelPerformance, CbbRec } from "@/lib/cbb-types";
import { CbbHeader, CbbLogo, ErrorBox, Loading, Mark } from "./parts";

const BREAK_EVEN = 0.524; // win rate needed at -110 to cover the vig
const rec = (r: CbbRec | null | undefined) => (r ? `${r.wins.toLocaleString()}–${r.losses.toLocaleString()}` : "—");
const rate = (r: CbbRec | null | undefined) => pct(r?.rate, 1);

function Panel({ title, children, note }: { title: string; children: React.ReactNode; note?: string }) {
  return <section aria-label={title} className="ql-panel min-w-0 p-4 sm:p-5"><h2 className="ql-section-title mb-3">{title}</h2>{children}{note && <p className="mt-3 text-xs leading-snug text-text-sec">{note}</p>}</section>;
}
function Big({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return <div><dt className="ql-data-label">{label}</dt><dd className="ql-metric mt-1">{value}</dd>{sub && <dd className="mt-1 text-xs text-text-sec">{sub}</dd>}</div>;
}

/** Predicted vs actual win rate per probability bin; the diagonal is a perfectly calibrated model. */
function Calibration({ bins }: { bins: CbbCalibrationBin[] }) {
  const S = 260, P = 28, x = (v: number) => P + v * (S - P - 8), y = (v: number) => S - P - v * (S - P - 8);
  const max = Math.max(...bins.map(b => b.n));
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto w-full max-w-sm" role="img" aria-label="Calibration chart: predicted win probability against actual win rate. Points near the diagonal mean the probabilities are honest.">
      {[0, 0.25, 0.5, 0.75, 1].map(t => <g key={t}><line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke="var(--color-border-sub)" /><text x={x(0) - 4} y={y(t) + 3} textAnchor="end" fontSize="8" fill="var(--color-text-sec)">{t * 100}%</text><text x={x(t)} y={S - P + 12} textAnchor="middle" fontSize="8" fill="var(--color-text-sec)">{t * 100}</text></g>)}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--color-lock-gold)" strokeDasharray="4 3" />
      {bins.map(b => <circle key={b.predicted} cx={x(b.predicted)} cy={y(b.actual)} r={3 + 5 * Math.sqrt(b.n / max)} fill="var(--color-teal-400)" fillOpacity=".75"><title>{`Predicted ${pct(b.predicted)}, won ${pct(b.actual)} (${b.n} games)`}</title></circle>)}
      <text x={S / 2} y={S - 2} textAnchor="middle" fontSize="8" fill="var(--color-text-sec)">Predicted home win %</text>
    </svg>
  );
}

function Body({ d }: { d: CbbModelPerformance }) {
  const hold = d.backtest?.holdout;
  const m = hold?.model ?? d.backtest?.model;
  const base = hold ?? d.backtest;
  const vs = d.vs_market;
  const live = d.track_record.live;
  return (
    <div className="space-y-4">
      {vs && (
        <div className="ql-panel border-lock-gold/40 p-4 sm:p-5" role="note">
          <p className="ql-kicker !text-lock-gold">The plain version</p>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-text-pri">
            Over {vs.n_games.toLocaleString()} games with sportsbook lines, the model picked {rate(vs.model_winner)} of winners and covered the spread {rate(vs.ats)} of the time ({rec(vs.ats)}).
            The book picked {rate(vs.market_winner)} of winners. Beating the spread needs about {pct(BREAK_EVEN, 1)} to cover the vig, so the model does not beat the market. Treat its picks as a second opinion, not an edge.
          </p>
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        {m && base && (
          <Panel title="Backtest" note={`Walk-forward: each game is called by a model that only knew earlier results.${hold ? ` Holdout is season ${hold.from_season}, which the rating rule was not tuned on.` : ""}`}>
            <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <Big label="Winners picked" value={pct(m.win_accuracy, 1)} sub={`${base.n_games.toLocaleString()} games`} />
              <Big label="Margin error" value={`${m.margin_mae.toFixed(1)} pts`} sub="mean absolute" />
              <Big label="Total error" value={`${m.total_mae.toFixed(1)} pts`} sub="mean absolute" />
            </dl>
            <p className="mt-3 text-xs text-text-sec">Ratings alone: {pct(hold?.ratings_only.win_accuracy ?? d.backtest?.ratings_only.win_accuracy, 1)} winners. Always picking the home team: {pct(base.home_team_always_wins, 1)}. Log loss {m.log_loss.toFixed(3)}, Brier {m.brier.toFixed(3)}.</p>
          </Panel>
        )}
        {vs && (
          <Panel title="Versus the sportsbook" note={`Seasons ${vs.seasons.join("–")}. Pushes are not counted.`}>
            <div className="overflow-x-auto"><table className="w-full text-sm">
              <thead><tr className="text-left"><th className="ql-data-label py-1.5">Measure</th><th className="ql-data-label py-1.5 text-right">Model</th><th className="ql-data-label py-1.5 text-right">Book</th></tr></thead>
              <tbody className="font-mono text-xs tabular-nums">
                <tr className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm">Winners picked</td><td className="text-right">{rate(vs.model_winner)}</td><td className="text-right">{rate(vs.market_winner)}</td></tr>
                <tr className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm">Margin error</td><td className="text-right">{vs.model_margin_mae.toFixed(2)}</td><td className="text-right">{vs.market_margin_mae.toFixed(2)}</td></tr>
                <tr className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm">Total error</td><td className="text-right">{vs.model_total_mae.toFixed(2)}</td><td className="text-right">{vs.market_total_mae.toFixed(2)}</td></tr>
                <tr className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm">Against the spread</td><td className="text-right">{rec(vs.ats)} · {rate(vs.ats)}</td><td className="text-right text-text-sec">≥ {pct(BREAK_EVEN, 1)} to profit</td></tr>
                <tr className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm">Over / under</td><td className="text-right">{rec(vs.ou)} · {rate(vs.ou)}</td><td className="text-right text-text-sec">≥ {pct(BREAK_EVEN, 1)} to profit</td></tr>
              </tbody>
            </table></div>
          </Panel>
        )}
        {vs && vs.ats_by_edge.length > 0 && (
          <Panel title="Does a bigger disagreement help?" note="Spread record when the model differs from the line by at least this many points. A real edge would climb with the gap; this stays flat.">
            <div className="overflow-x-auto"><table className="w-full text-xs">
              <thead><tr className="text-left"><th className="ql-data-label py-1.5">Gap</th><th className="ql-data-label py-1.5 text-right">Record</th><th className="ql-data-label py-1.5 text-right">Cover</th><th className="ql-data-label py-1.5 text-right">Games</th></tr></thead>
              <tbody className="font-mono tabular-nums">{vs.ats_by_edge.map(e => <tr key={e.min_edge} className="border-t border-ink/[0.06]"><td className="py-1.5">{e.min_edge}+ pts</td><td className="text-right">{rec(e)}</td><td className="text-right">{rate(e)}</td><td className="text-right text-text-sec">{e.n.toLocaleString()}</td></tr>)}</tbody>
            </table></div>
          </Panel>
        )}
        <Panel title="By season">
          <div className="overflow-x-auto"><table className="w-full text-xs">
            <thead><tr className="text-left"><th className="ql-data-label py-1.5">Season</th><th className="ql-data-label py-1.5 text-right">Winners</th><th className="ql-data-label py-1.5 text-right">Margin err</th>{vs && <th className="ql-data-label py-1.5 text-right">ATS</th>}</tr></thead>
            <tbody className="font-mono tabular-nums">{d.by_season.map(s => {
              const ats = vs?.by_season.find(x => x.season === s.season)?.ats;
              return <tr key={s.season} className="border-t border-ink/[0.06]"><td className="py-1.5">{s.season - 1}–{String(s.season).slice(2)}</td><td className="text-right">{pct(s.win_accuracy, 1)}</td><td className="text-right">{s.margin_mae.toFixed(1)}</td>{vs && <td className="text-right">{ats ? `${rec(ats)} · ${rate(ats)}` : "—"}</td>}</tr>;
            })}</tbody>
          </table></div>
        </Panel>
        {d.calibration.length > 0 && (
          <Panel title="Calibration" note="Teal dots are probability bins sized by games; the gold line is perfect. When the model says 70%, the home team wins about that often.">
            <Calibration bins={d.calibration} />
          </Panel>
        )}
        <Panel title="Logged before tip-off">
          {live.games > 0
            ? <dl className="grid grid-cols-3 gap-4"><Big label="Winners" value={rec(live.winner)} sub={rate(live.winner)} /><Big label="Spread" value={rec(live.ats)} sub={rate(live.ats)} /><Big label="Totals" value={rec(live.total)} sub={rate(live.total)} /></dl>
            : <p className="text-sm text-text-sec">No game has finished since the daily job began logging calls before tip-off, so there is no live record yet. Everything above is backtested.</p>}
        </Panel>
      </div>
      {d.recent.length > 0 && (
        <Panel title="Last graded games">
          <ul className="divide-y divide-ink/[0.07]">
            {d.recent.map(g => (
              <li key={g.id} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 py-2.5 text-sm sm:grid-cols-[84px_minmax(0,1fr)_auto]">
                <span className="col-span-2 font-mono text-[10px] text-text-sec sm:col-span-1">{g.date}</span>
                <span className="flex min-w-0 items-center gap-2">
                  <CbbLogo team={g.away} size={20} /><span className="truncate">{g.away.short} <b className="font-mono">{g.result?.away_score}</b></span>
                  <span className="text-text-sec">@</span>
                  <CbbLogo team={g.home} size={20} /><span className="truncate">{g.home.short} <b className="font-mono">{g.result?.home_score}</b></span>
                </span>
                {g.result && <span className="flex items-center gap-2 font-mono text-[10px] text-text-sec">W <Mark kind={mark(g.result.winner_correct)} /> ATS <Mark kind={mark(g.result.ats_correct, g.result.ats_result)} /> O/U <Mark kind={mark(g.result.total_correct, g.result.total_result)} /></span>}
              </li>
            ))}
          </ul>
        </Panel>
      )}
      {d.notes.length > 0 && <ul className="space-y-1 text-xs text-text-sec">{d.notes.map(n => <li key={n}>{n}</li>)}</ul>}
    </div>
  );
}

export function CbbModel() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["cbb-model"], queryFn: cbbApi.model, staleTime: 600_000 });
  return (
    <div className="ql-page">
      <CbbHeader title="Model record" sub={data ? `Version ${data.model.version}, trained on ${data.model.n_games.toLocaleString()} games.` : undefined} />
      {isPending && <Loading label="Loading model record" />}
      {isError && <ErrorBox onRetry={() => void refetch()}>The model record is unavailable.</ErrorBox>}
      {data && <Body d={data} />}
    </div>
  );
}
