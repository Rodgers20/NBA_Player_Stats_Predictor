export interface CbbGameLogRow {
  id: string; date: string; home: boolean; opp: string; oppLogo: string | null; result: string | null; score: string | null;
  min: number | null; pts: number | null; reb: number | null; ast: number | null; stl: number | null; blk: number | null; tov: number | null;
  fgm: number | null; fga: number | null; tpm: number | null; tpa: number | null;
}
export interface CbbWindow { games: number; min: number | null; pts: number | null; reb: number | null; ast: number | null; stl: number | null; blk: number | null; tov: number | null; tpm: number | null }
export const GAMELOG_URL: string;
export function parseGamelog(json: unknown): CbbGameLogRow[];
export function windowAverages(games: CbbGameLogRow[], n?: number): CbbWindow;
