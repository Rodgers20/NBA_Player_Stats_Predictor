import { api } from "./api";
import type { MatchCentreData } from "@/components/MatchCentre";

/** Loads everything the Match Centre needs in one parallel fan-out. */
export async function loadMatchCentre(matchId: number): Promise<MatchCentreData> {
  const [
    match, stats, events, momentum, shots, passes, topPlayers, standings, heatmap, frames, lineup,
  ] = await Promise.all([
    api.match(matchId),
    api.stats(matchId),
    api.events(matchId),
    api.momentum(matchId),
    api.shots(matchId),
    api.passes(matchId, { limit: 2000 }),
    api.topPlayers(matchId, 5),
    api.standings(),
    api.heatmap(matchId),
    api.freezeFrames(matchId, 240),
    api.lineupPositions(matchId),
  ]);
  return {
    match, stats, events, momentum, shots, passes, topPlayers, standings, heatmap, frames, lineup,
  };
}
