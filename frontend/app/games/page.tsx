"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { GamesResponse } from "@/lib/types";
import { fmtOdds } from "@/lib/utils";

export default function GamesPage() {
  const { data, isPending, isError } = useQuery<GamesResponse>({
    queryKey:        ["games"],
    queryFn:         api.games,
    refetchInterval: 300_000,
  });

  return (
    <div className="px-4 pt-3 max-w-2xl mx-auto">
      <div className="mb-4">
        <h1 className="text-2xl font-black tracking-tight"
            style={{ color: "var(--color-text-pri)", fontFamily: "var(--font-display)" }}>
          Today&apos;s Games
        </h1>
        {data?.target_date && (
          <p className="text-xs mt-0.5" style={{ color: "var(--color-text-sec)" }}>{data.target_date}</p>
        )}
      </div>

      {isPending && (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="glass h-24 animate-pulse" />
          ))}
        </div>
      )}

      {isError && (
        <div className="glass p-4 text-center text-sm" style={{ color: "var(--color-danger)" }}>
          Backend offline — run <code className="font-mono text-xs bg-white/5 px-1.5 py-0.5 rounded">uvicorn api.main:app --port 8000</code>
        </div>
      )}

      {!isPending && data?.games?.length === 0 && (
        <div className="glass p-6 text-center text-sm" style={{ color: "var(--color-text-sec)" }}>
          No games scheduled for today.
        </div>
      )}

      {!isPending && data && data.games.length > 0 && (
        <div className="flex flex-col gap-3">
          {data.games.map((g) => (
            <div key={g.matchup} className="glass p-4 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="flex flex-col items-center gap-0.5 flex-1">
                  <span className="text-lg font-extrabold" style={{ color: "var(--color-text-pri)" }}>{g.away_team}</span>
                  <span className="text-xs" style={{ color: "var(--color-text-sec)" }}>Away</span>
                </div>
                <div className="flex flex-col items-center">
                  <span className="text-xs font-semibold" style={{ color: "var(--color-teal-400)" }}>@</span>
                  {g.game_time && <span className="text-[10px] mt-0.5" style={{ color: "var(--color-text-sec)" }}>{g.game_time}</span>}
                </div>
                <div className="flex flex-col items-center gap-0.5 flex-1">
                  <span className="text-lg font-extrabold" style={{ color: "var(--color-text-pri)" }}>{g.home_team}</span>
                  <span className="text-xs" style={{ color: "var(--color-text-sec)" }}>Home</span>
                </div>
              </div>

              {(g.spread != null || g.total != null || g.home_ml != null) && (
                <div className="grid grid-cols-3 gap-2 text-center border-t pt-3" style={{ borderColor: "var(--color-border-sub)" }}>
                  {[
                    ["Spread", g.spread != null ? (g.spread > 0 ? `+${g.spread}` : String(g.spread)) : "—"],
                    ["Total",  g.total != null ? `O/U ${g.total}` : "—"],
                    ["ML",     `${fmtOdds(g.away_ml)} / ${fmtOdds(g.home_ml)}`],
                  ].map(([label, value]) => (
                    <div key={label} className="flex flex-col items-center gap-0.5">
                      <span className="text-sm font-bold" style={{ color: "var(--color-text-pri)" }}>{value}</span>
                      <span className="text-[10px] uppercase tracking-wide" style={{ color: "var(--color-text-sec)" }}>{label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
