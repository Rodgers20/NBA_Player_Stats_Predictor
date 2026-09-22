import type { PropsResponse, GamesResponse, PlayerChartData, PlayerStats } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function get<T>(path: string, params?: Record<string, string | number | boolean>): Promise<T> {
  const qs = params
    ? "?" + new URLSearchParams(
        Object.entries(params)
          .filter(([, v]) => v !== undefined && v !== null && v !== "")
          .map(([k, v]) => [k, String(v)])
      ).toString()
    : "";
  const res = await fetch(`${BASE}${path}${qs}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  props: (params: Record<string, string | number | boolean> = {}) =>
    get<PropsResponse>("/api/props", params),

  games: () => get<GamesResponse>("/api/games"),

  playerChart: (player: string, stat = "PTS", games = 20) =>
    get<PlayerChartData>(`/api/player/${encodeURIComponent(player)}/chart-data`, { stat, games }),

  playerStats: (player: string) =>
    get<PlayerStats>(`/api/player/${encodeURIComponent(player)}/stats`),

  players: (q = "") =>
    get<{ players: string[] }>("/api/players", q ? { q } : {}),
};
