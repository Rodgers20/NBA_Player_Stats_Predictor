"use client";

import { useState } from "react";
import { MARKETS, defaultLine, hitSummary, lineLabel, linesFor, logValue, pct, windowGames } from "@/lib/nhl.mjs";
import type { NhlMarketKey, NhlPlayerDetail } from "@/lib/nhl-types";

const WINDOWS = [{ n: 5, label: "Last 5" }, { n: 10, label: "Last 10" }, { n: 20, label: "Last 20" }, { n: 0, label: "Season" }] as const;
const day = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { month: "numeric", day: "numeric", timeZone: "UTC" });

/** Game-log bars against a chosen line, with a market tab, a line selector and a last 5 / 10 / 20 / season window. */
export function NhlPlayerChart({ d }: { d: NhlPlayerDetail }) {
  const [market, setMarket] = useState<NhlMarketKey>("sog");
  const [picked, setPicked] = useState<Record<string, number>>({});
  const [win, setWin] = useState<number>(10);
  const lines = linesFor(d, market);
  const line = picked[market] && lines.includes(picked[market]) ? picked[market] : defaultLine(lines);
  const games = windowGames(d.log, win, d.season.gp);
  const values = games.map(g => logValue(g, market));
  const top = Math.max(1, line ?? 0, ...values.map(v => v ?? 0)) + (market === "sog" ? 1 : 0);
  const counts = WINDOWS.map(w => ({ ...w, ...hitSummary(windowGames(d.log, w.n, d.season.gp).map(g => logValue(g, market)), line ?? 0) }));
  const label = MARKETS.find(m => m.key === market)!;
  const tall = 150;
  return (
    <section aria-label="Game log chart" className="ql-panel p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="ql-section-title" style={{ fontSize: 18 }}>{label.label} by game</h3>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Market">
          {MARKETS.map(m => <button key={m.key} type="button" aria-pressed={market === m.key} onClick={() => setMarket(m.key)} className={`ql-chip ${market === m.key ? "ql-chip-active" : ""}`}>{m.short}</button>)}
        </div>
      </div>
      {lines.length > 1 && (
        <div className="mt-3 flex flex-wrap items-center gap-2" role="group" aria-label="Line">
          <span className="ql-data-label">LINE</span>
          {lines.map(l => <button key={l} type="button" aria-pressed={line === l} onClick={() => setPicked(p => ({ ...p, [market]: l }))} className={`ql-chip ${line === l ? "ql-chip-active" : ""}`}>{l} <span className="opacity-70">({lineLabel(market, l)})</span></button>)}
        </div>
      )}
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4" role="group" aria-label="Games shown">
        {counts.map(c => (
          <button key={c.label} type="button" aria-pressed={win === c.n} disabled={c.n === 0} onClick={() => setWin(c.n)}
            aria-label={`${c.label}: ${c.rate == null ? "no games" : `${c.hits} of ${c.n} games over ${line}, ${Math.round(c.rate * 100)} percent`}`}
            className={`rounded-lg border px-3 py-2 text-left transition-colors disabled:opacity-40 ${win === c.n ? "border-teal-400/70 bg-teal-400/10" : "border-border-sub bg-surface-hover hover:border-teal-400/40"}`}>
            <span className="ql-data-label block">{c.label}</span>
            <span className={`block font-mono text-lg font-bold tabular-nums ${c.rate != null && c.rate >= 0.7 ? "text-teal-300" : c.rate != null && c.rate <= 0.4 ? "text-rose-300" : "text-text-pri"}`}>{c.rate == null ? "—" : pct(c.rate)}</span>
            <span className="block font-mono text-[10px] text-text-sec">{c.hits}/{c.n}</span>
          </button>
        ))}
      </div>
      {games.length === 0 || line == null ? <p role="status" className="mt-4 text-sm text-text-sec">No games in this window.</p> : (
        <>
          <div className="relative mt-4 flex items-end gap-[3px]" style={{ height: tall + 18 }} role="img"
            aria-label={`${label.label} in the last ${games.length} games, oldest to newest, against a line of ${line}`}>
            <span aria-hidden className="absolute inset-x-0 z-10 border-t border-dashed border-lock-gold" style={{ bottom: `${(line / top) * tall + 14}px` }} />
            <span aria-hidden className="absolute right-0 z-10 -translate-y-full rounded bg-bg-root/80 px-1 font-mono text-[10px] font-bold text-lock-gold" style={{ bottom: `${(line / top) * tall + 14}px` }}>{line}</span>
            {games.map((g, i) => {
              const v = values[i];
              const hit = v != null && v > line;
              return (
                <div key={g.game_id} className="flex min-w-0 flex-1 flex-col items-center justify-end" title={`${day(g.date)} ${g.home ? "vs" : "@"} ${g.opp}: ${v ?? "–"} ${label.unit}${hit ? " (over)" : ""}`}>
                  <span className="font-mono text-[9px] tabular-nums text-text-sec">{games.length <= 21 ? v ?? "" : ""}</span>
                  <span className={`w-full rounded-sm ${hit ? "bg-teal-400/85" : "bg-rose-400/65"}`} style={{ height: `${Math.max(3, ((v ?? 0) / top) * tall)}px` }} />
                  <span className="h-3.5 font-mono text-[8px] leading-[14px] text-text-sec">{games.length <= 12 ? g.opp : ""}</span>
                </div>
              );
            })}
          </div>
          <p className="mt-2 text-xs text-text-sec">Teal bars cleared {line} ({lineLabel(market, line)}), red bars did not. The dashed gold line is the threshold. Oldest game on the left.</p>
        </>
      )}
    </section>
  );
}
