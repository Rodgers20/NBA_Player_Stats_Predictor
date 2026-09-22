"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "motion/react";
import { Star } from "lucide-react";
import { api } from "@/lib/api";
import { usePrefs } from "@/store/prefs";
import { STAT_COLORS } from "@/lib/utils";
import { PlayerBarChart }  from "@/components/charts/PlayerBarChart";
import { TrendLineChart }  from "@/components/charts/TrendLineChart";

const STATS = ["PTS", "AST", "REB", "FG3M", "STL", "BLK"];

export default function AnalysisPage() {
  const { followedPlayers, followPlayer, unfollowPlayer } = usePrefs();
  const [query, setQuery]   = useState("");
  const [player, setPlayer] = useState("");
  const [stat, setStat]     = useState("PTS");
  const [chartType, setChartType] = useState<"bar" | "line">("bar");

  const { data: playerList } = useQuery({
    queryKey: ["players", query],
    queryFn:  () => api.players(query),
    enabled:  query.length >= 2,
  });

  const { data: chartData, isPending: chartPending } = useQuery({
    queryKey: ["chart", player, stat],
    queryFn:  () => api.playerChart(player, stat, 20),
    enabled:  !!player,
  });

  const { data: statsData } = useQuery({
    queryKey: ["stats", player],
    queryFn:  () => api.playerStats(player),
    enabled:  !!player,
  });

  const isFollowed = followedPlayers.includes(player);

  return (
    <div className="px-4 pt-3 max-w-2xl mx-auto">
      <h1 className="text-2xl font-black tracking-tight mb-4"
          style={{ color: "var(--color-text-pri)", fontFamily: "var(--font-display)" }}>
        Player Analysis
      </h1>

      {/* Search */}
      <div className="relative mb-4">
        <input
          value={query}
          onChange={(e) => { setQuery(e.target.value); setPlayer(""); }}
          placeholder="Search player…"
          className="w-full glass px-4 py-3 text-sm focus:outline-none"
          style={{ color: "var(--color-text-pri)" }}
        />
        {playerList && playerList.players.length > 0 && !player && (
          <ul className="absolute z-20 top-full mt-1 w-full glass rounded-xl overflow-hidden">
            {playerList.players.slice(0, 8).map((p) => (
              <li key={p}>
                <button
                  onClick={() => { setPlayer(p); setQuery(p); }}
                  className="w-full text-left px-4 py-2.5 text-sm hover:bg-white/5 transition-colors"
                  style={{ color: "var(--color-text-pri)" }}
                >
                  {p}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {!player && (
        <div className="glass p-6 text-center text-sm" style={{ color: "var(--color-text-sec)" }}>
          {followedPlayers.length > 0 ? (
            <>
              <p className="mb-3">Your followed players:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {followedPlayers.map((p) => (
                  <button key={p} onClick={() => { setPlayer(p); setQuery(p); }}
                    className="text-xs px-3 py-1.5 rounded-full border border-white/10 hover:bg-white/5"
                    style={{ color: "var(--color-teal-400)" }}>
                    {p}
                  </button>
                ))}
              </div>
            </>
          ) : (
            "Search for a player to see their stats and charts."
          )}
        </div>
      )}

      {player && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          {/* Player header */}
          {statsData && (
            <div className="glass p-4 mb-4 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <p className="text-base font-bold" style={{ color: "var(--color-text-pri)" }}>{statsData.player}</p>
                  <button onClick={() => isFollowed ? unfollowPlayer(player) : followPlayer(player)}>
                    <Star size={14} fill={isFollowed ? "#f59e0b" : "none"} color={isFollowed ? "#f59e0b" : "#8ca0c0"} />
                  </button>
                </div>
                <p className="text-xs" style={{ color: "var(--color-text-sec)" }}>
                  {statsData.team} · {statsData.position} · {statsData.games_played} GP
                </p>
              </div>
              <div className="flex gap-3 text-right">
                {[["PTS", "#14b8a6"], ["AST", "#f97066"], ["REB", "#a78bfa"]].map(([s, c]) => (
                  <div key={s}>
                    <p className="text-lg font-extrabold" style={{ color: c }}>{statsData.season_avgs[s] ?? "—"}</p>
                    <p className="text-[10px]" style={{ color: "var(--color-text-sec)" }}>{s}PG</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Stat + chart type selectors */}
          <div className="flex gap-2 overflow-x-auto pb-1 mb-3">
            {STATS.map((s) => {
              const c = STAT_COLORS[s] ?? "var(--color-teal-400)";
              return (
                <button key={s} onClick={() => setStat(s)}
                  className="flex-shrink-0 text-xs font-semibold px-3 py-1.5 rounded-full border transition-colors"
                  style={stat === s
                    ? { color: c, borderColor: `color-mix(in srgb, ${c} 40%, transparent)`, background: `color-mix(in srgb, ${c} 12%, transparent)` }
                    : { color: "var(--color-text-sec)", borderColor: "rgba(255,255,255,0.1)" }
                  }>
                  {s}
                </button>
              );
            })}
          </div>
          <div className="flex gap-2 mb-3">
            {(["bar", "line"] as const).map((t) => (
              <button key={t} onClick={() => setChartType(t)}
                className="text-xs font-medium px-3 py-1 rounded-full border transition-colors"
                style={chartType === t
                  ? { color: "var(--color-teal-400)", borderColor: "color-mix(in srgb, var(--color-teal-400) 40%, transparent)", background: "color-mix(in srgb, var(--color-teal-400) 10%, transparent)" }
                  : { color: "var(--color-text-sec)", borderColor: "rgba(255,255,255,0.1)" }
                }>
                {t === "bar" ? "Bar" : "Line"}
              </button>
            ))}
          </div>

          {/* Chart */}
          <div className="glass p-4 mb-4">
            {chartPending && <div className="h-48 animate-pulse" />}
            {!chartPending && chartData && (
              <>
                <div className="flex justify-between items-baseline mb-3">
                  <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--color-text-sec)" }}>
                    {stat} — Last 20 games
                  </span>
                  <span className="text-xs" style={{ color: "var(--color-text-sec)" }}>
                    Avg {chartData.avg} · L5 {chartData.l5_avg}
                    {chartData.line != null && ` · Line ${chartData.line}`}
                  </span>
                </div>
                {chartType === "bar"
                  ? <PlayerBarChart games={chartData.games} stat={stat} line={chartData.line} height={200} />
                  : <TrendLineChart games={chartData.games} stat={stat} line={chartData.line} height={200} />
                }
              </>
            )}
          </div>

          {/* L5 vs Season bars */}
          {statsData && (
            <div className="glass p-4">
              <p className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--color-text-sec)" }}>
                Season vs L5 Averages
              </p>
              <div className="flex flex-col gap-2">
                {STATS.map((s) => {
                  const season = statsData.season_avgs[s];
                  const l5     = statsData.l5_avgs[s];
                  const c      = STAT_COLORS[s] ?? "#2dd4bf";
                  if (!season) return null;
                  const pct = Math.min(100, (l5 / season) * 100);
                  return (
                    <div key={s} className="flex items-center gap-3">
                      <span className="text-xs font-semibold w-10" style={{ color: c }}>{s}</span>
                      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.05)" }}>
                        <motion.div
                          className="h-full rounded-full"
                          style={{ background: c }}
                          initial={{ width: 0 }}
                          animate={{ width: `${pct}%` }}
                          transition={{ duration: 0.6 }}
                        />
                      </div>
                      <span className="text-xs w-16 text-right" style={{ color: "var(--color-text-sec)" }}>
                        {l5} <span className="text-[10px]">/ {season}</span>
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
