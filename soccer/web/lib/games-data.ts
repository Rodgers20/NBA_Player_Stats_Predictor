import { api } from "./api";
import type { FormMatch } from "./types";

/**
 * Recent results for one club, newest first. The snapshot holds the form file at
 * a few limits (10 for the game page, 5-6 from older exports), so ask for the
 * longest first and fall back; a club with none yields [] and the UI shows no dots.
 */
export async function teamFormFor(code: string, team: string): Promise<FormMatch[]> {
  for (const limit of [10, 6, 5]) {
    try {
      const rows = await api.teamForm(code, team, limit);
      if (rows.length) return rows;
    } catch {
      // not snapshotted at this limit; try the next
    }
  }
  return [];
}
