"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { TeamLogo } from "@/components/teams/TeamLogo";
import { GAMELOG_URL, parseGamelog, windowAverages } from "@/lib/cbb-gamelog.mjs";
import type { CbbGameLogRow } from "@/lib/cbb-gamelog.mjs";
import { formatStat, seasonLabel, statValue } from "@/lib/cbb-players.mjs";
import type { CbbStatLine } from "@/lib/cbb-types";
import { Headshot } from "./Headshot";

export interface SheetPlayer {
  id: string; name: string; pos?: string | null; team?: string | null; abbr?: string | null; logo?: string | null;
  season: number; line?: CbbStatLine; note?: ReactNode;
}

const TILES = ["pts", "reb", "ast", "min", "stl", "blk"] as const;
const WINDOWS = [{ n: 5, label: "Last 5" }, { n: 10, label: "Last 10" }, { n: 0, label: "Season" }] as const;
const CHART_GAMES = 20;

async function fetchLog(id: string, season: number): Promise<CbbGameLogRow[]> {
  const res = await fetch(`${GAMELOG_URL}/${id}/gamelog?region=us&lang=en&contentorigin=espn&season=${season}`);
  if (!res.ok) throw new Error(`ESPN answered ${res.status}`);
  return parseGamelog(await res.json());
}

const day = (iso: string) => new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/New_York" });

/** Full-screen college player view. Owns one history entry so the back button closes it, like the NBA player sheet. */
export function CbbPlayerSheet({ player, onClose }: { player: SheetPlayer; onClose: () => void }) {
  const back = useRef<HTMLButtonElement>(null);
  const closeRef = useRef(onClose);
  useEffect(() => { closeRef.current = onClose; }, [onClose]);
  useEffect(() => {
    const returnFocus = document.activeElement as HTMLElement | null;
    const { overflow } = document.documentElement.style;
    document.documentElement.style.overflow = "hidden";
    window.history.pushState({ ...window.history.state, cbbSheet: true }, "");
    const onPop = () => closeRef.current();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") window.history.back(); };
    window.addEventListener("popstate", onPop);
    window.addEventListener("keydown", onKey);
    back.current?.focus({ preventScroll: true });
    return () => {
      window.removeEventListener("popstate", onPop);
      window.removeEventListener("keydown", onKey);
      document.documentElement.style.overflow = overflow;
      returnFocus?.focus({ preventScroll: true });
    };
  }, []);

  const log = useQuery({ queryKey: ["cbb-gamelog", player.id, player.season], queryFn: () => fetchLog(player.id, player.season), staleTime: 300_000 });
  const [win, setWin] = useState<number>(10);
  const games = useMemo(() => log.data ?? [], [log.data]);
  const averages = useMemo(() => windowAverages(games, win), [games, win]);
  const chart = useMemo(() => games.slice(0, CHART_GAMES).reverse(), [games]);
  const top = Math.max(1, ...chart.map(g => g.pts ?? 0));

  return createPortal(
    <div role="dialog" aria-modal="true" aria-label={`${player.name} stats`} className="ql-enter fixed inset-0 z-[91] overflow-y-auto overscroll-contain bg-[var(--color-bg-root)]">
      <div className="sticky top-0 z-20 flex items-center gap-3 border-b border-ink/10 bg-bg-root/85 px-3 py-2.5 backdrop-blur-xl sm:px-6">
        <button ref={back} type="button" onClick={() => window.history.back()}
          className="inline-flex items-center gap-2 rounded-full px-3 py-2 text-sm font-semibold text-text-pri hover:bg-ink/5 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
          <ArrowLeft size={18} aria-hidden="true" /> Back
        </button>
        <p className="truncate text-sm font-bold text-text-sec">{player.name}</p>
      </div>
      <div className="ql-page space-y-6">
        <header className="flex items-center gap-4">
          <Headshot id={player.id} name={player.name} size={84} />
          <div className="min-w-0">
            <h2 className="ql-heading truncate" style={{ fontSize: "clamp(26px, 5vw, 40px)" }}>{player.name}</h2>
            <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-text-sec">
              {player.logo && <TeamLogo team={{ abbrev: player.abbr ?? "", name: player.team ?? "", logo: player.logo }} size={20} />}
              <span>{[player.team, player.pos].filter(Boolean).join(" · ")}</span>
              <span className="font-mono text-xs">{seasonLabel(player.season)}</span>
            </p>
            {player.note && <p className="mt-2 text-sm text-text-sec">{player.note}</p>}
          </div>
        </header>

        {player.line && (
          <section aria-label="Season averages" className="grid grid-cols-3 gap-2 sm:grid-cols-6">
            {TILES.map(k => (
              <div key={k} className="ql-panel p-3">
                <p className="ql-data-label">{k.toUpperCase()}</p>
                <p className="ql-metric mt-1">{formatStat(statValue(player.line, k), k)}</p>
              </div>
            ))}
          </section>
        )}

        {log.isPending && <div className="ql-panel h-48 animate-pulse" aria-busy="true" aria-label="Loading game log" />}
        {log.isError && <p role="alert" className="ql-panel p-5 text-sm text-text-sec">Game log is unavailable right now. <button type="button" className="underline" onClick={() => void log.refetch()}>Try again</button></p>}
        {log.data && games.length === 0 && <p role="status" className="ql-panel p-5 text-sm text-text-sec">No games logged in {seasonLabel(player.season)} yet.</p>}

        {games.length > 0 && <>
          <section aria-label="Recent form" className="ql-panel p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="ql-section-title" style={{ fontSize: 18 }}>Points by game</h3>
              <div className="flex gap-2" role="group" aria-label="Averaging window">
                {WINDOWS.map(w => (
                  <button key={w.label} type="button" aria-pressed={win === w.n} onClick={() => setWin(w.n)} className={`ql-chip ${win === w.n ? "ql-chip-active" : ""}`}>{w.label}</button>
                ))}
              </div>
            </div>
            <div className="mt-4 flex h-32 items-end gap-1" role="img" aria-label={`Points in the last ${chart.length} games, oldest to newest`}>
              {chart.map(g => (
                <div key={g.id} className="flex min-w-0 flex-1 flex-col items-center justify-end gap-1" title={`${day(g.date)} ${g.home ? "vs" : "@"} ${g.opp}: ${g.pts ?? "–"} pts`}>
                  <span className="font-mono text-[9px] tabular-nums text-text-sec">{g.pts ?? ""}</span>
                  <span className={`w-full rounded-sm ${g.result === "W" ? "bg-teal-400/80" : "bg-rose-400/70"}`} style={{ height: `${Math.max(3, ((g.pts ?? 0) / top) * 88)}px` }} />
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-text-sec">Teal bars are wins, red bars are losses.</p>
            <dl className="mt-4 grid grid-cols-3 gap-2 sm:grid-cols-7">
              {(["pts", "reb", "ast", "min", "stl", "blk", "tov"] as const).map(k => (
                <div key={k}><dt className="ql-data-label">{k.toUpperCase()}</dt><dd className="font-mono text-base font-bold tabular-nums text-text-pri">{formatStat(averages[k], k)}</dd></div>
              ))}
            </dl>
            <p className="mt-2 text-xs text-text-sec">Average over {averages.games} {averages.games === 1 ? "game" : "games"}.</p>
          </section>

          <section aria-label="Game log" className="ql-panel overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr className="border-b border-border-card text-left">
                {["Date", "Opponent", "Result", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TO", "3PM", "FG"].map(h => <th key={h} scope="col" className="ql-data-label px-3 py-3">{h}</th>)}
              </tr></thead>
              <tbody>
                {games.map(g => (
                  <tr key={g.id} className="border-b border-ink/[0.06] last:border-0">
                    <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-text-sec">{day(g.date)}</td>
                    <td className="whitespace-nowrap px-3 py-2 text-text-pri">{g.home ? "vs" : "@"} {g.opp}</td>
                    <td className={`whitespace-nowrap px-3 py-2 font-mono text-xs font-bold ${g.result === "W" ? "text-teal-300" : "text-rose-300"}`}>{g.result} {g.score}</td>
                    {[g.min, g.pts, g.reb, g.ast, g.stl, g.blk, g.tov, g.tpm].map((v, i) => <td key={i} className="px-3 py-2 font-mono text-xs tabular-nums">{v ?? "–"}</td>)}
                    <td className="whitespace-nowrap px-3 py-2 font-mono text-xs tabular-nums">{g.fgm}-{g.fga}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>}
      </div>
    </div>,
    document.body,
  );
}
