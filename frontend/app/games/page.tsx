"use client";

import useSWR from "swr";
import { fetchGames } from "@/lib/api";
import type { GamesResponse } from "@/lib/types";
import { fmtOdds } from "@/lib/utils";

export default function GamesPage() {
  const { data, isLoading, error } = useSWR<GamesResponse>("games", fetchGames, {
    refreshInterval: 300_000,
  });

  return (
    <div className="px-4 pt-3 max-w-2xl mx-auto">
      <div className="mb-4">
        <h1 className="text-xl font-extrabold text-[#f0f4ff]">Today&apos;s Games</h1>
        {data?.target_date && (
          <p className="text-xs text-[#8ca0c0] mt-0.5">{data.target_date}</p>
        )}
      </div>

      {isLoading && (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="glass h-24 animate-pulse" />
          ))}
        </div>
      )}

      {error && (
        <div className="glass p-4 text-center text-sm text-[#f87171]">
          Failed to load games — is the backend running?
        </div>
      )}

      {!isLoading && data?.games?.length === 0 && (
        <div className="glass p-6 text-center text-sm text-[#8ca0c0]">
          No games scheduled for today.
        </div>
      )}

      {!isLoading && data && data.games.length > 0 && (
        <div className="flex flex-col gap-3">
          {data.games.map((g) => (
            <GameCard key={g.matchup} game={g} />
          ))}
        </div>
      )}
    </div>
  );
}

function GameCard({ game }: { game: GamesResponse["games"][0] }) {
  return (
    <div className="glass p-4 flex flex-col gap-3">
      {/* Teams */}
      <div className="flex items-center justify-between">
        <div className="flex flex-col items-center gap-0.5 flex-1">
          <span className="text-lg font-extrabold text-[#f0f4ff]">{game.away_team}</span>
          <span className="text-xs text-[#8ca0c0]">Away</span>
        </div>
        <div className="flex flex-col items-center">
          <span className="text-xs font-semibold text-[#2dd4bf]">@</span>
          {game.game_time && (
            <span className="text-[10px] text-[#8ca0c0] mt-0.5">{game.game_time}</span>
          )}
        </div>
        <div className="flex flex-col items-center gap-0.5 flex-1">
          <span className="text-lg font-extrabold text-[#f0f4ff]">{game.home_team}</span>
          <span className="text-xs text-[#8ca0c0]">Home</span>
        </div>
      </div>

      {/* Odds */}
      {(game.spread != null || game.total != null || game.home_ml != null) && (
        <div className="grid grid-cols-3 gap-2 text-center border-t border-white/5 pt-3">
          <OddsPill label="Spread" value={game.spread != null ? (game.spread > 0 ? `+${game.spread}` : String(game.spread)) : "—"} />
          <OddsPill label="Total" value={game.total != null ? `O/U ${game.total}` : "—"} />
          <OddsPill label="ML" value={`${fmtOdds(game.away_ml)} / ${fmtOdds(game.home_ml)}`} />
        </div>
      )}
    </div>
  );
}

function OddsPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5">
      <span className="text-sm font-bold text-[#f0f4ff]">{value}</span>
      <span className="text-[10px] text-[#8ca0c0] uppercase tracking-wide">{label}</span>
    </div>
  );
}
