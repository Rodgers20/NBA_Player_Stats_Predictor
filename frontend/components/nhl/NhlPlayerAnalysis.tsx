"use client";

import { MARKETS, num, pct } from "@/lib/nhl.mjs";
import type { NhlMarketKey, NhlPlayerDetail } from "@/lib/nhl-types";
import { AbbrLogo, Headshot } from "./parts";
import { NhlNextGame } from "./NhlNextGame";
import { NhlPlayerChart } from "./NhlPlayerChart";

const day = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
const STAT_COLS = [["goals", "G"], ["assists", "A"], ["points", "P"], ["sog", "SOG"]] as const;

export function PlayerHeader({ d }: { d: NhlPlayerDetail }) {
  const s = d.season;
  const tiles = [["GP", s.gp], ["G", s.goals], ["A", s.assists], ["PTS", s.points], ["SOG", s.sog], ["TOI", s.toi != null ? num(s.toi) : "—"], ["PPG", s.pp_goals], ["SH%", s.shooting_pct != null ? pct(s.shooting_pct, 1) : "—"]];
  return (
    <>
      <header className="flex items-center gap-4">
        <Headshot src={d.player.headshot} name={d.player.name} size={84} />
        <div className="min-w-0">
          <h2 className="ql-heading truncate" style={{ fontSize: "clamp(26px, 5vw, 40px)" }}>{d.player.name}</h2>
          <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-text-sec"><AbbrLogo abbr={d.player.team} size={20} /><span>{d.player.team} · {d.player.pos}</span><span className="font-mono text-xs">this season</span></p>
        </div>
      </header>
      <section aria-label="Season totals" className="grid grid-cols-4 gap-2 sm:grid-cols-8">
        {tiles.map(([k, v]) => <div key={k} className="ql-panel p-2.5"><p className="ql-data-label">{k}</p><p className="ql-metric mt-1 !text-xl">{v}</p></div>)}
      </section>
    </>
  );
}

function HitRates({ d }: { d: NhlPlayerDetail }) {
  const rows = MARKETS.flatMap(m => (d.hit_rates[m.key as NhlMarketKey] ?? []).map((r, i) => ({ m, r, first: i === 0 })));
  if (!rows.length) return null;
  return (
    <section aria-label="Hit rates" className="ql-panel overflow-x-auto">
      <h3 className="ql-section-title px-4 pt-4" style={{ fontSize: 18 }}>Hit rates over each line</h3>
      <table className="mt-2 w-full min-w-[420px] text-sm">
        <thead><tr className="border-b border-border-card text-left">{["Market", "Line", "L5", "L10", "L20", "Season"].map((h, i) => <th key={h} scope="col" className={`ql-data-label px-3 py-2.5 ${i > 1 ? "text-right" : ""}`}>{h}</th>)}</tr></thead>
        <tbody>{rows.map(({ m, r, first }) => (
          <tr key={`${m.key}-${r.line}`} className="border-b border-ink/[0.06] last:border-0">
            <td className="px-3 py-2 font-bold">{first ? m.label : ""}</td><td className="px-3 py-2 font-mono text-xs">{r.line}</td>
            {[r.l5, r.l10, r.l20, r.season].map((v, i) => <td key={i} className={`px-3 py-2 text-right font-mono text-xs tabular-nums ${v != null && v >= 0.7 ? "text-teal-300" : v != null && v <= 0.3 ? "text-rose-300" : "text-text-soft"}`}>{pct(v)}</td>)}
          </tr>
        ))}</tbody>
      </table>
    </section>
  );
}

function Averages({ d }: { d: NhlPlayerDetail }) {
  const wins = [["Last 5", d.rolling.l5], ["Last 10", d.rolling.l10], ["Last 20", d.rolling.l20], ["Season", d.rolling.season]] as const;
  const splits = [["Home", d.splits.home], ["Away", d.splits.away]] as const;
  return (
    <section aria-label="Averages and splits" className="ql-panel overflow-x-auto p-4">
      <h3 className="ql-section-title mb-2" style={{ fontSize: 18 }}>Per-game averages and splits</h3>
      <table className="w-full min-w-[360px] text-sm">
        <thead><tr className="text-left"><th className="ql-data-label py-1.5">Window</th>{STAT_COLS.map(([, h]) => <th key={h} className="ql-data-label py-1.5 text-right">{h}</th>)}<th className="ql-data-label py-1.5 text-right">TOI</th></tr></thead>
        <tbody className="font-mono text-xs tabular-nums">
          {wins.map(([label, r]) => <tr key={label} className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm">{label}</td>{STAT_COLS.map(([k]) => <td key={k} className="text-right">{num(r[k], 2)}</td>)}<td className="text-right text-text-sec">{num(r.toi)}</td></tr>)}
          {splits.map(([label, r]) => r && <tr key={label} className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm">{label} <span className="text-[10px] text-text-sec">({r.gp} gp)</span></td>{STAT_COLS.map(([k]) => <td key={k} className="text-right">{num(r[k], 2)}</td>)}<td /></tr>)}
        </tbody>
      </table>
    </section>
  );
}

function GameLog({ d }: { d: NhlPlayerDetail }) {
  return (
    <section aria-label="Game log" className="ql-panel overflow-x-auto">
      <table className="w-full text-sm">
        <thead><tr className="border-b border-border-card text-left">{["Date", "Opponent", "Result", "TOI", "G", "A", "PTS", "SOG"].map(h => <th key={h} scope="col" className="ql-data-label px-3 py-3">{h}</th>)}</tr></thead>
        <tbody>{d.log.slice(0, 40).map(g => (
          <tr key={g.game_id} className="border-b border-ink/[0.06] last:border-0">
            <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-text-sec">{day(g.date)}</td>
            <td className="whitespace-nowrap px-3 py-2">{g.home ? "vs" : "@"} {g.opp}</td>
            <td className={`px-3 py-2 font-mono text-xs font-bold ${g.result === "W" ? "text-teal-300" : "text-rose-300"}`}>{g.result ?? "–"}</td>
            <td className="px-3 py-2 font-mono text-xs text-text-sec">{g.toi}</td>
            {[g.goals, g.assists, g.points, g.sog].map((v, i) => <td key={i} className="px-3 py-2 font-mono text-xs tabular-nums">{v}</td>)}
          </tr>
        ))}</tbody>
      </table>
    </section>
  );
}

/** Everything about one player: next-game projections, the log chart, hit rates, splits and the game log. */
export function NhlPlayerAnalysis({ d }: { d: NhlPlayerDetail }) {
  return (
    <div className="space-y-6">
      <PlayerHeader d={d} />
      <NhlNextGame d={d} />
      {d.log.length > 0 ? <NhlPlayerChart d={d} /> : <p role="status" className="ql-panel p-5 text-sm text-text-sec">No games logged yet.</p>}
      <div className="grid gap-4 lg:grid-cols-2"><HitRates d={d} /><Averages d={d} /></div>
      <GameLog d={d} />
    </div>
  );
}
