"use client";

import { useQuery } from "@tanstack/react-query";
import { nhlApi } from "@/lib/api";
import { MARKETS, pct } from "@/lib/nhl.mjs";
import { Panel } from "@/components/games/shared/Panel";
import type { NhlCalBin, NhlMarketKey, NhlModelPerformance, NhlPropPerf } from "@/lib/nhl-types";
import { ErrorBox, Loading, Mark, NhlHeader, FRAME } from "./parts";

function Big({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return <div><dt className="ql-data-label">{label}</dt><dd className="ql-metric mt-1">{value}</dd>{sub && <dd className="mt-1 text-xs text-text-sec">{sub}</dd>}</div>;
}

/** Predicted vs actual rate per probability bin; the gold diagonal is a perfectly calibrated model. */
export function Calibration({ bins, label }: { bins: NhlCalBin[]; label: string }) {
  const S = 260, P = 28, x = (v: number) => P + v * (S - P - 8), y = (v: number) => S - P - v * (S - P - 8);
  const max = Math.max(1, ...bins.map(b => b.n));
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto w-full max-w-sm" role="img" aria-label={`Calibration chart: predicted ${label} against actual rate. Points near the diagonal mean the probabilities are honest.`}>
      {[0, 0.25, 0.5, 0.75, 1].map(t => <g key={t}><line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke="var(--color-border-sub)" /><text x={x(0) - 4} y={y(t) + 3} textAnchor="end" fontSize="8" fill="var(--color-text-sec)">{t * 100}%</text><text x={x(t)} y={S - P + 12} textAnchor="middle" fontSize="8" fill="var(--color-text-sec)">{t * 100}</text></g>)}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--color-lock-gold)" strokeDasharray="4 3" />
      {bins.map(b => <circle key={b.predicted} cx={x(b.predicted)} cy={y(b.actual)} r={3 + 5 * Math.sqrt(b.n / max)} fill="var(--color-teal-400)" fillOpacity=".75"><title>{`Predicted ${pct(b.predicted)}, actual ${pct(b.actual)} (${b.n.toLocaleString()} samples)`}</title></circle>)}
      <text x={S / 2} y={S - 2} textAnchor="middle" fontSize="8" fill="var(--color-text-sec)">Predicted {label} %</text>
    </svg>
  );
}

function PropPanel({ k, p, rec }: { k: NhlMarketKey; p: NhlPropPerf; rec?: NhlModelPerformance["record"]["props"][NhlMarketKey] }) {
  const label = MARKETS.find(m => m.key === k)!.label;
  const base = Object.entries(p.baselines)[0];
  return (
    <Panel title={label} note={base ? `Lower log loss and Brier are better. Baseline "${base[0].replace(/_/g, " ")}": log loss ${base[1].log_loss.toFixed(3)}, Brier ${base[1].brier.toFixed(3)}.` : undefined}>
      <dl className="grid grid-cols-3 gap-4">
        <Big label="Log loss" value={p.model.log_loss.toFixed(3)} sub={`${p.n.toLocaleString()} player games`} />
        <Big label="Brier" value={p.model.brier.toFixed(3)} />
        {p.top_decile && <Big label="Top 10% picks" value={pct(p.top_decile.hit_rate, 1)} sub={`predicted ${pct(p.top_decile.predicted, 1)}, base ${pct(p.top_decile.base_rate, 1)}`} />}
      </dl>
      {p.calibration.length > 0 && <div className="mt-4"><Calibration bins={p.calibration} label={`${label.toLowerCase()} probability`} /></div>}
      {p.lines.length > 1 && (
        <table className="mt-3 w-full text-xs"><thead><tr className="text-left"><th className="ql-data-label py-1.5">Line</th><th className="ql-data-label py-1.5 text-right">Hit rate</th><th className="ql-data-label py-1.5 text-right">Log loss</th><th className="ql-data-label py-1.5 text-right">Samples</th></tr></thead>
          <tbody className="font-mono tabular-nums">{p.lines.map(l => <tr key={l.line} className="border-t border-ink/[0.06]"><td className="py-1.5">{l.line}</td><td className="text-right">{pct(l.hit_rate, 1)}</td><td className="text-right">{l.log_loss.toFixed(3)}</td><td className="text-right text-text-sec">{l.n.toLocaleString()}</td></tr>)}</tbody></table>
      )}
      {rec && rec.graded > 0 && <p className="mt-3 text-xs text-text-sec">Logged live: {rec.graded.toLocaleString()} graded{rec.top_decile_hit_rate != null && `, top-decile hit rate ${pct(rec.top_decile_hit_rate, 1)}`}{rec.avg_p != null && ` (average probability ${pct(rec.avg_p, 1)})`}.</p>}
    </Panel>
  );
}

function Body({ d }: { d: NhlModelPerformance }) {
  const g = d.game, r = d.record;
  const base = Object.entries(g.baselines);
  return (
    <div className="space-y-4">
      <div className="ql-panel border-lock-gold/40 p-4 sm:p-5" role="note">
        <p className="ql-kicker !text-lock-gold">The plain version</p>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-text-pri">
          Over {g.n.toLocaleString()} backtested games the model picked {pct(g.model.accuracy, 1)} of winners. NHL games are close to coin flips, so even a good model sits in the high 50s.
          Prop numbers are probabilities, not sportsbook lines: they are only useful next to a price you can actually get.{g.note ? ` ${g.note}` : ""}
        </p>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Game winner backtest" note={`Walk-forward over ${g.seasons.join(", ")}: each game is called by a model that only knew earlier results.`}>
          <dl className="grid grid-cols-3 gap-4">
            <Big label="Winners picked" value={pct(g.model.accuracy, 1)} sub={`${g.n.toLocaleString()} games`} />
            <Big label="Log loss" value={g.model.log_loss.toFixed(3)} />
            <Big label="Brier" value={g.model.brier.toFixed(3)} />
          </dl>
          <table className="mt-4 w-full text-xs"><thead><tr className="text-left"><th className="ql-data-label py-1.5">Baseline</th><th className="ql-data-label py-1.5 text-right">Winners</th><th className="ql-data-label py-1.5 text-right">Log loss</th><th className="ql-data-label py-1.5 text-right">Brier</th></tr></thead>
            <tbody className="font-mono tabular-nums">{base.map(([k, b]) => <tr key={k} className="border-t border-ink/[0.06]"><td className="py-1.5 font-sans capitalize">{k.replace(/_/g, " ")}</td><td className="text-right">{pct(b.accuracy, 1)}</td><td className="text-right">{b.log_loss.toFixed(3)}</td><td className="text-right">{b.brier.toFixed(3)}</td></tr>)}</tbody></table>
          <p className="mt-3 text-xs text-text-sec">Projected goals per game {g.total_goals.mean_pred.toFixed(2)} vs {g.total_goals.mean_actual.toFixed(2)} actual; total-goals error {g.total_goals.mae.toFixed(2)} on average.</p>
        </Panel>
        <Panel title="Calibration" note="Teal dots are probability bins sized by games; the gold line is perfect. When the model says 60%, that side should win about 60% of the time.">
          <Calibration bins={g.calibration} label="home win" />
        </Panel>
        <Panel title="Logged before puck drop">
          {r.games.graded > 0
            ? <dl className="grid grid-cols-2 gap-4"><Big label="Winners" value={`${r.games.correct}–${r.games.graded - r.games.correct}`} sub={pct(r.games.rate, 1)} /></dl>
            : <p className="text-sm text-text-sec">No game has finished since the daily job began logging calls before puck drop, so there is no live record yet.</p>}
        </Panel>
        {r.recent.length > 0 && (
          <Panel title="Last graded games">
            <ul className="divide-y divide-ink/[0.07]">{r.recent.map(x => (
              <li key={x.game_id} className="flex items-center justify-between gap-3 py-2 text-sm">
                <span className="w-24 shrink-0 font-mono text-[11px] text-text-sec">{x.date}</span>
                <span className="min-w-0 flex-1">Pick <b>{x.pick}</b> <span className="font-mono text-xs text-text-sec">{pct(x.p)}</span> · won by {x.result}</span>
                <Mark kind={x.correct ? "hit" : "miss"} />
              </li>
            ))}</ul>
          </Panel>
        )}
      </div>
      <h2 className="ql-section-title pt-2">Prop backtest</h2>
      <div className="grid gap-4 lg:grid-cols-2">
        {MARKETS.map(m => d.props[m.key] && <PropPanel key={m.key} k={m.key} p={d.props[m.key]!} rec={r.props[m.key]} />)}
      </div>
    </div>
  );
}

export function NhlModel() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["nhl-model"], queryFn: nhlApi.model, staleTime: 600_000 });
  return (
    <div className={FRAME}>
      <NhlHeader kicker="MODEL" title="Model record" sub="How the game and prop models did in a walk-forward backtest and on calls logged before puck drop." />
      {isPending && <Loading label="Loading model record" />}
      {isError && <ErrorBox onRetry={() => void refetch()}>The model record is unavailable.</ErrorBox>}
      {data && <Body d={data} />}
    </div>
  );
}
