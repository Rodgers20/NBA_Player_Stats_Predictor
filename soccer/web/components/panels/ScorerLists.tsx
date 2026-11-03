import type { AssistRow, ContributionRow, ScorerRow } from "@/lib/types";

type Row = { player: string; team: string; value: number; sub?: string };

export function LeaderList({ title, rows, accent }: { title: string; rows: Row[]; accent: string }) {
  return (
    <div className="min-w-0">
      <h3 className="mb-2 text-[10px] uppercase tracking-[0.14em] text-ink-muted">{title}</h3>
      <ul className="space-y-1.5">
        {rows.map((r, i) => (
          <li key={`${r.player}-${i}`} className="flex items-center gap-2.5">
            <span className="grid place-items-center w-7 h-7 shrink-0 rounded-full bg-elevated text-[10px] font-semibold text-ink-secondary ring-1 ring-stroke">
              {r.player.split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase()}
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[12px] font-medium">{r.player}</span>
              <span className="block truncate text-[10px] text-ink-muted">{r.team}</span>
            </span>
            <span
              className="shrink-0 rounded px-1.5 py-0.5 font-mono text-[11px] font-semibold tnum"
              style={{ background: `color-mix(in oklab, ${accent} 18%, transparent)`, color: accent }}
            >
              {r.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export const toScorerRows = (r: ScorerRow[]): Row[] =>
  r.map((x) => ({ player: x.player, team: x.team, value: x.goals }));
export const toAssistRows = (r: AssistRow[]): Row[] =>
  r.map((x) => ({ player: x.player, team: x.team, value: x.assists }));
export const toContributionRows = (r: ContributionRow[]): Row[] =>
  r.map((x) => ({ player: x.player, team: x.team, value: x.contributions }));
