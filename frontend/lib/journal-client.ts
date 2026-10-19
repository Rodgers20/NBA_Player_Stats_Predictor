import { localJournalAdd, localJournalList, localJournalSettle } from "./local-journal.mjs";
import { legKey, type SlipLeg } from "./bet-slip.mjs";
import { gradeBet } from "./hit-rate.mjs";
import { api } from "./api";

export type Mode = "paper" | "real";
export type Result = "win" | "loss" | "push" | "void" | "pending";
export type Bet = {
  id: string; league: string; game_date: string; player: string; side: string;
  line: number; stat: string; book: string; price: number; stake_cents: number;
  profit_cents: number | null; result: Result; notes: string;
};
export type Journal = { bets: Bet[]; summary: { settled: number; profit: number; roi: number | null; pending: number } };

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
export const STATIC = process.env.NEXT_PUBLIC_DATA_MODE === "static";
const JSON_HEADERS = { "Content-Type": "application/json" };
const easternToday = () => new Date().toLocaleDateString("en-CA", { timeZone: "America/New_York" });

export class JournalError extends Error {
  constructor(message: string, public validation = false) { super(message); }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (STATIC) {
    try {
      if (!init?.method || init.method === 'GET') return localJournalList(localStorage, new URLSearchParams(path.replace(/^\?/, '')).get('mode') || 'paper') as T;
      const body = JSON.parse(String(init.body || '{}'));
      if (init.method === 'POST') return localJournalAdd(localStorage, body) as T;
      if (init.method === 'PATCH') return localJournalSettle(localStorage, decodeURIComponent(path.slice(1)), body.result) as T;
    } catch (error) {
      throw new JournalError(error instanceof Error ? error.message : 'Could not save your journal.', true);
    }
    throw new JournalError('Unsupported journal operation.');
  }
  const response = await fetch(`${BASE}/api/journal${path}`, { cache: "no-store", ...init });
  const body = await response.json().catch(() => null);
  if (!response.ok || !body) {
    const detail = body?.detail;
    throw new JournalError(typeof detail === "string" ? detail : Array.isArray(detail)
      ? detail.map((item: { loc: string[]; msg: string }) => `${item.loc.slice(1).join(".")}: ${item.msg}`).join(" · ")
      : "The journal service is unavailable. Your change was not confirmed; retry when connected.", response.status === 422);
  }
  return body as T;
}

/**
 * Records each slip leg as its own paper bet. The token is the leg's key, so tapping Track twice
 * (or tracking the same leg again) cannot create a duplicate.
 */
export async function trackLegs(legs: SlipLeg[], stake: number): Promise<{ saved: number; failed: string[] }> {
  const failed: string[] = [];
  let saved = 0;
  for (const leg of legs) {
    try {
      await request<{ id: string }>("", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify({
        token: `slip-${legKey(leg)}`, mode: "paper", league: leg.league, player: leg.player, game_date: leg.date,
        stat: leg.stat, side: leg.direction, line: leg.line, price: leg.price, book: "Paper", stake,
        notes: `From bet slip · ${leg.matchup}` }) });
      saved += 1;
    } catch (error) {
      failed.push(`${leg.player}: ${error instanceof Error ? error.message : "could not save"}`);
    }
  }
  return { saved, failed };
}

/**
 * Grades pending bets whose game has a box score in player history. Bets without a result yet
 * (game not played, or the player did not play) stay pending for you to settle by hand.
 */
export async function autoSettle(bets: Bet[]): Promise<{ settled: number; failed: number }> {
  const today = easternToday();
  const due = bets.filter(bet => bet.result === "pending" && bet.game_date <= today);
  const series = new Map<string, Promise<{ date: string; value: number | null }[]>>();
  const gamesFor = (bet: Bet) => {
    const key = `${bet.league}|${bet.player}|${bet.stat}`;
    if (!series.has(key)) series.set(key, api.playerChart(bet.player, bet.stat, 200, bet.league).then(chart => chart.games));
    return series.get(key)!;
  };
  let settled = 0, failed = 0;
  for (const bet of due) {
    try {
      const result = gradeBet(bet, await gamesFor(bet));
      if (!result) continue;
      await request(`/${encodeURIComponent(bet.id)}`, { method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify({ result }) });
      settled += 1;
    } catch {
      failed += 1;
    }
  }
  return { settled, failed };
}
