import type { League } from "./types";

/** The competition the whole app is currently scoped to. */
export const DEFAULT_COMPETITION = "E0";

export function resolveCompetition(
  requested: string | undefined,
  leagues: League[],
): string {
  if (requested && leagues.some((l) => l.code === requested)) return requested;
  if (leagues.some((l) => l.code === DEFAULT_COMPETITION)) return DEFAULT_COMPETITION;
  return leagues[0]?.code ?? DEFAULT_COMPETITION;
}

/** Build an href that keeps the selected competition. */
export const withCompetition = (path: string, code: string) =>
  `${path}${path.includes("?") ? "&" : "?"}competition=${encodeURIComponent(code)}`;

export const seasonLabel = (year: number | null | undefined, code?: string) => {
  if (year == null) return "—";
  if (code === "MLS") return String(year);
  return `${year}/${String(year + 1).slice(2)}`;
};
