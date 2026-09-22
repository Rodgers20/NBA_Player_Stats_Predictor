"use client";

import { useState } from "react";
import useSWR from "swr";
import { fetchPlayers, fetchPlayerChart, fetchPlayerStats } from "@/lib/api";
import { PlayerBarChart } from "@/components/charts/PlayerBarChart";
import { TrendLineChart } from "@/components/charts/TrendLineChart";
import { cn, STAT_COLORS } from "@/lib/utils";

const STATS = ["PTS", "AST", "REB", "FG3M", "STL", "BLK"];

export default function AnalysisPage() {
  const [query, setQuery]   = useState("");
  const [player, setPlayer] = useState("");
  const [stat, setStat]     = useState("PTS");
  const [chart, setChart]   = useState<"bar" | "line">("bar");

  const { data: playerList } = useSWR(
    query.length >= 2 ? ["players", query] : null,
    () => fetchPlayers(query)
  );

  const { data: chartData, isLoading: chartLoading } = useSWR(
    player ? ["chart", player, stat] : null,
    () => fetchPlayerChart(player, stat, 20)
  );

  const { data: statsData } = useSWR(
    player ? ["stats", player] : null,
    () => fetchPlayerStats(player)
  );

  return (
    <div className="px-4 pt-3 max-w-2xl mx-auto">
      <h1 className="text-xl font-extrabold text-[#f0f4ff] mb-4">Player Analysis</h1>

      {/* Search */}
      <div className="relative mb-4">
        <input
          value={query}
          onChange={(e) => { setQuery(e.target.value); setPlayer(""); }}
          placeholder="Search player…"
          className="w-full glass px-4 py-3 text-sm text-[#f0f4ff] placeholder-[#8ca0c0] focus:outline-none focus:ring-1 focus:ring-[#2dd4bf]/40 rounded-xl"
        />
        {playerList && playerList.players.length > 0 && !player && (
          <ul className="absolute z-20 top-full mt-1 w-full glass rounded-xl overflow-hidden">
            {playerList.players.slice(0, 8).map((p) => (
              <li key={p}>
                <button
                  onClick={() => { setPlayer(p); setQuery(p); }}
                  className="w-full text-left px-4 py-2.5 text-sm text-[#f0f4ff] hover:bg-white/5 transition-colors"
                >
                  {p}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {!player && (
        <div className="glass p-6 text-center text-sm text-[#8ca0c0]">
          Search for a player to see their stats and charts.
        </div>
      )}

      {player && (
        <>
          {/* Player header */}
          {statsData && (
            <div className="glass p-4 mb-4 flex items-center justify-between">
              <div>
                <p className="text-base font-bold text-[#f0f4ff]">{statsData.player}</p>
                <p className="text-xs text-[#8ca0c0]">
                  {statsData.team} · {statsData.position} · {statsData.games_played} GP
                </p>
              </div>
              <div className="flex gap-3 text-right">
                <div>
                  <p className="text-lg font-extrabold text-[#2dd4bf]">{statsData.season_avgs["PTS"] ?? "—"}</p>
                  <p className="text-[10px] text-[#8ca0c0] uppercase">PPG</p>
                </div>
                <div>
                  <p className="text-lg font-extrabold text-[#f97066]">{statsData.season_avgs["AST"] ?? "—"}</p>
                  <p className="text-[10px] text-[#8ca0c0] uppercase">APG</p>
                </div>
                <div>
                  <p className="text-lg font-extrabold text-[#a78bfa]">{statsData.season_avgs["REB"] ?? "—"}</p>
                  <p className="text-[10px] text-[#8ca0c0] uppercase">RPG</p>
                </div>
              </div>
            </div>
          )}

          {/* Stat selector */}
          <div className="flex gap-2 overflow-x-auto pb-1 mb-3">
            {STATS.map((s) => {
              const color = STAT_COLORS[s] ?? "#2dd4bf";
              return (
                <button
                  key={s}
                  onClick={() => setStat(s)}
                  style={stat === s ? { color, borderColor: `${color}50`, background: `${color}15` } : {}}
                  className={cn(
                    "flex-shrink-0 text-xs font-semibold px-3 py-1.5 rounded-full border transition-colors",
                    stat === s ? "" : "border-white/10 text-[#8ca0c0]"
                  )}
                >
                  {s}
                </button>
              );
            })}
          </div>

          {/* Chart toggle */}
          <div className="flex gap-2 mb-3">
            {(["bar", "line"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setChart(t)}
                className={cn(
                  "text-xs font-medium px-3 py-1 rounded-full border transition-colors",
                  chart === t
                    ? "bg-[#2dd4bf]/15 border-[#2dd4bf]/40 text-[#2dd4bf]"
                    : "border-white/10 text-[#8ca0c0]"
                )}
              >
                {t === "bar" ? "Bar" : "Line"}
              </button>
            ))}
          </div>

          {/* Chart */}
          <div className="glass p-4 mb-4">
            {chartLoading && <div className="h-48 animate-pulse" />}
            {!chartLoading && chartData && (
              <>
                <div className="flex justify-between items-baseline mb-3">
                  <span className="text-xs font-semibold text-[#8ca0c0] uppercase tracking-wide">{stat} — Last 20 games</span>
                  <span className="text-xs text-[#8ca0c0]">
                    Avg {chartData.avg} · L5 {chartData.l5_avg}
                    {chartData.line != null && ` · Line ${chartData.line}`}
                  </span>
                </div>
                {chart === "bar" ? (
                  <PlayerBarChart games={chartData.games} stat={stat} line={chartData.line} height={200} />
                ) : (
                  <TrendLineChart games={chartData.games} stat={stat} line={chartData.line} height={200} />
                )}
              </>
            )}
          </div>

          {/* L5 vs Season */}
          {statsData && (
            <div className="glass p-4">
              <p className="text-xs font-semibold text-[#8ca0c0] uppercase tracking-wide mb-3">Season vs L5 Averages</p>
              <div className="flex flex-col gap-2">
                {STATS.map((s) => {
                  const season = statsData.season_avgs[s];
                  const l5     = statsData.l5_avgs[s];
                  const color  = STAT_COLORS[s] ?? "#2dd4bf";
                  if (season == null) return null;
                  const pct = season > 0 ? Math.min(100, (l5 / season) * 100) : 0;
                  return (
                    <div key={s} className="flex items-center gap-3">
                      <span className="text-xs font-semibold w-10" style={{ color }}>{s}</span>
                      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${pct}%`, background: color }}
                        />
                      </div>
                      <span className="text-xs text-[#8ca0c0] w-16 text-right">
                        {l5} <span className="text-[10px]">/ {season}</span>
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
