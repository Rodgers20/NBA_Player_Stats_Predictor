import type { PlayerGamelog } from "./types";

/**
 * Gamelogs already opened in this browser tab. The player page fills it; the props board's
 * Details expander reads it, so "last 5" appears only for players already loaded (no per-row fetch).
 */
export const gamelogCache = new Map<string, PlayerGamelog>();
