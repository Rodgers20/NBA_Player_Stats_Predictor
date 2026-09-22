import type { PropsResponse, GamesResponse, PlayerChartData, PlayerStats } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { next: { revalidate: 60 } });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export function fetchProps(params: Record<string, string | number | boolean> = {}): Promise<PropsResponse> {
  const qs = new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== "")
      .map(([k, v]) => [k, String(v)])
  ).toString();
  return get<PropsResponse>(`/api/props${qs ? `?${qs}` : ""}`);
}

export function fetchGames(): Promise<GamesResponse> {
  return get<GamesResponse>("/api/games");
}

export function fetchPlayerChart(player: string, stat = "PTS", games = 20): Promise<PlayerChartData> {
  return get<PlayerChartData>(
    `/api/player/${encodeURIComponent(player)}/chart-data?stat=${stat}&games=${games}`
  );
}

export function fetchPlayerStats(player: string): Promise<PlayerStats> {
  return get<PlayerStats>(`/api/player/${encodeURIComponent(player)}/stats`);
}

export function fetchPlayers(q = ""): Promise<{ players: string[] }> {
  return get<{ players: string[] }>(`/api/players${q ? `?q=${encodeURIComponent(q)}` : ""}`);
}
