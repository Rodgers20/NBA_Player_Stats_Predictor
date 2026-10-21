"use client";

import { useSlip } from "@/store/slip";
import { legKey, type SlipLeg } from "@/lib/bet-slip.mjs";
import { fmtLine, fmtPrice, type MarketPair } from "@/lib/game-markets.mjs";
import { mlLeg, spreadLeg, totalLeg } from "@/lib/team-legs.mjs";
import type { Game, GamePrediction } from "@/lib/types";

/** What a "+" button needs: whether the leg is on the slip, a spoken label, and the toggle. */
export type SlipAdd = { inSlip: boolean; label: string; onToggle: () => void };
export type SideWithAdd = MarketPair[number] & { add?: SlipAdd };

const etDate = (iso?: string | null) => {
  const when = iso ? new Date(iso) : null;
  return when && !Number.isNaN(when.getTime()) ? when.toLocaleDateString("en-CA", { timeZone: "America/New_York" }) : null;
};

/** Attach "+" buttons to the two sides of a market (a missing market stays missing). */
export function withAdds(pair: MarketPair | null, adds: [SlipAdd | undefined, SlipAdd | undefined]): [SideWithAdd, SideWithAdd] | null {
  return pair ? [{ ...pair[0], add: adds[0] }, { ...pair[1], add: adds[1] }] : null;
}

/**
 * Slip buttons for one game's moneyline, spread and total. Moneyline legs carry the real ESPN price and the model's
 * win chance; spread and total legs carry the assumed standard price (see team-legs.mjs). Returns undefined for a
 * market with no line, so no button is drawn for it.
 */
export function useGameSlip(game: Game, league: "nba" | "wnba", date: string | undefined, prediction?: GamePrediction) {
  const legs = useSlip(state => state.legs);
  const toggle = useSlip(state => state.toggle);
  const day = date && /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : etDate(game.start_time);

  const make = (leg: SlipLeg | null, label: string): SlipAdd | undefined => leg
    ? {
        inSlip: legs.some(item => legKey(item) === legKey(leg) && item.direction === leg.direction && item.line === leg.line),
        label,
        onToggle: () => toggle({ ...leg, addedAt: new Date().toISOString() }),
      }
    : undefined;

  if (!day) return { moneyline: {}, spread: {}, total: {} } as const;
  const ctx = { league, date: day };
  const homeChance = prediction?.home_win_probability;
  const chance = typeof homeChance === "number" && homeChance > 0 && homeChance < 1 ? homeChance : null;
  const price = (value: number | null) => fmtPrice(value);
  return {
    moneyline: {
      away: make(mlLeg({ game, side: "away", ...ctx, probability: chance === null ? null : 1 - chance }), `${game.away_team} moneyline ${price(game.away_ml)}`),
      home: make(mlLeg({ game, side: "home", ...ctx, probability: chance }), `${game.home_team} moneyline ${price(game.home_ml)}`),
    },
    spread: {
      away: make(spreadLeg({ game, side: "away", ...ctx }), `${game.away_team} ${fmtLine(-(game.spread ?? 0))} spread`),
      home: make(spreadLeg({ game, side: "home", ...ctx }), `${game.home_team} ${fmtLine(game.spread ?? 0)} spread`),
    },
    total: {
      over: make(totalLeg({ game, direction: "Over", ...ctx }), `Over ${game.total} total`),
      under: make(totalLeg({ game, direction: "Under", ...ctx }), `Under ${game.total} total`),
    },
  } as const;
}
