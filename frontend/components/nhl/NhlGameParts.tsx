import { Panel } from "@/components/games/shared/Panel";
import { VenueCard } from "@/components/games/VenueCard";
import { SIDE_COLOR, num, signed } from "@/lib/nhl.mjs";
import type { NhlFactor, NhlGameDetail, NhlSide } from "@/lib/nhl-types";

const longDate = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });

export function Factors({ g }: { g: NhlGameDetail }) {
  if (!g.factors.length) return null;
  const max = Math.max(1, ...g.factors.map(f => Math.abs(f.pct)));
  const show = (v: NhlFactor["home"]) => (v == null ? "—" : String(v));
  return (
    <Panel title="Why the model leans this way" note={`Each bar is how many points of home win probability the factor moves, toward ${g.game.home.abbr} (right) or ${g.game.away.abbr} (left). An approximate attribution, not an exact split.`}>
      <ul className="space-y-3">
        {g.factors.map(f => (
          <li key={f.key} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 text-sm">
            <span className="truncate text-text-pri">{f.label} <span className="font-mono text-[11px] text-text-sec">{g.game.away.abbr} {show(f.away)} · {g.game.home.abbr} {show(f.home)}</span></span>
            <span className="font-mono text-xs tabular-nums">{signed(f.pct)} pts</span>
            <div className="relative col-span-2 h-2 rounded-sm bg-surface-hover" aria-hidden>
              <span className="absolute inset-y-0 left-1/2 w-px bg-border-card" />
              <span className="absolute inset-y-0 rounded-sm" style={{ background: f.pct >= 0 ? SIDE_COLOR.home : SIDE_COLOR.away, width: `${(Math.abs(f.pct) / max) * 50}%`, [f.pct >= 0 ? "left" : "right"]: "50%" }} />
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

export function RecentForm({ g }: { g: NhlGameDetail }) {
  const side = (s: NhlSide) => {
    const f = g.form[s], t = g.game[s];
    if (!f.last10.length) return null;
    const wins = f.last10.filter(r => r === "W").length;
    // Oldest to newest: the data lists newest first.
    const bars = [...f.last10].reverse().map((r, i) => ({ win: r === "W", tip: `${r === "OTL" ? "OT loss" : r === "W" ? "Win" : "Loss"} (${f.last10.length - i} games ago)` }));
    return <VenueCard key={s} title={`${t.short} last ${f.last10.length} (${f.record_l10})`} games={f.last10.length} wins={wins} losses={f.last10.length - wins}
      scores={num(f.gf)} allows={num(f.ga)} margin={f.gf != null && f.ga != null ? f.gf - f.ga : null} bars={bars} />;
  };
  return (
    <Panel title="Recent form" note="Last 10 games before puck drop, oldest to newest (teal win, rose loss or overtime loss). Scores and allows are goals per game.">
      <div className="grid gap-3 md:grid-cols-2">{side("away")}{side("home")}</div>
      <dl className="mt-3 grid grid-cols-2 gap-3 text-xs">
        {(["away", "home"] as const).map(s => (
          <div key={s}><dt className="ql-data-label">{g.game[s].abbr} SHOTS FOR / AGAINST</dt><dd className="font-mono tabular-nums text-text-pri">{num(g.form[s].sf)} / {num(g.form[s].sa)} per game</dd></div>
        ))}
      </dl>
    </Panel>
  );
}

export function RestGoalies({ g }: { g: NhlGameDetail }) {
  const goalies = g.game.goalies;
  const rest = (s: NhlSide) => {
    const r = g.rest[s];
    return r.days == null ? "—" : `${r.days} ${r.days === 1 ? "day" : "days"}${r.b2b ? " (back-to-back)" : ""}`;
  };
  return (
    <Panel title="Rest and goalies">
      <table className="w-full text-sm">
        <thead><tr className="text-left"><th className="p-0"><span className="sr-only">Measure</span></th>{(["away", "home"] as const).map(s => <th key={s} className="ql-data-label pb-2 text-right">{g.game[s].abbr}</th>)}</tr></thead>
        <tbody className="font-mono text-xs tabular-nums">
          <tr className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm text-text-sec">Rest</td>{(["away", "home"] as const).map(s => <td key={s} className={`text-right ${g.rest[s].b2b ? "text-lock-gold" : "text-text-pri"}`}>{rest(s)}</td>)}</tr>
          <tr className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm text-text-sec">Goalie</td>{(["away", "home"] as const).map(s => <td key={s} className="text-right text-text-pri">{goalies?.[s]?.name ?? "—"}{goalies?.[s]?.projected ? "*" : ""}</td>)}</tr>
          <tr className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm text-text-sec">Save %</td>{(["away", "home"] as const).map(s => <td key={s} className="text-right text-text-pri">{goalies?.[s]?.sv_pct != null ? goalies[s]!.sv_pct!.toFixed(3).replace(/^0/, "") : "—"}</td>)}</tr>
          <tr className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm text-text-sec">Elo</td>{(["away", "home"] as const).map(s => <td key={s} className="text-right text-text-pri">{g.elo[s] ?? "—"}</td>)}</tr>
        </tbody>
      </table>
      {(goalies?.home?.projected || goalies?.away?.projected) && <p className="mt-3 text-xs text-text-sec">* Projected starter: the goalie who started most recently. It is not confirmed until the boxscore posts.</p>}
    </Panel>
  );
}

export function H2H({ g }: { g: NhlGameDetail }) {
  if (!g.h2h.games.length) return null;
  const { home, away } = g.game;
  return (
    <Panel title="Head to head" note="Recent meetings, newest first.">
      <p className="mb-2 text-sm font-bold">{away.short} {g.h2h.away_wins} – {g.h2h.home_wins} {home.short}</p>
      <ul className="divide-y divide-ink/[0.07]">
        {g.h2h.games.map(m => (
          <li key={`${m.date}-${m.home}`} className="flex items-center justify-between gap-3 py-2 text-sm">
            <span className="w-24 shrink-0 font-mono text-[11px] text-text-sec">{longDate(m.date)}</span>
            <span className="min-w-0 flex-1 tabular-nums"><span className={m.away_score > m.home_score ? "font-bold" : "text-text-sec"}>{m.away} {m.away_score}</span> <span className="text-text-sec">@</span> <span className={m.home_score > m.away_score ? "font-bold" : "text-text-sec"}>{m.home} {m.home_score}</span></span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
