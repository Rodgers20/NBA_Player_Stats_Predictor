import { Info } from "lucide-react";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import { TeamCrest } from "@/components/TeamCrest";
import type { PlayerLeaders } from "@/lib/types";

export function LeaderBoard({ leaders }: { leaders: PlayerLeaders }) {
  return (
    <div className="grid gap-8 lg:grid-cols-3">
      <Column
        title="Top Scorers"
        unit="goals"
        accent="var(--color-pitch)"
        rows={leaders.scorers.map((p) => ({
          player: p.player, team: p.team, value: p.goals, playerId: p.player_id,
        }))}
        empty={leaders.note ?? "No scorer table published yet."}
      />
      <Column
        title="Assists"
        unit="assists"
        accent="var(--color-series-4)"
        rows={leaders.assists.map((p) => ({
          player: p.player, team: p.team, value: p.assists, playerId: p.player_id,
        }))}
        empty={leaders.assists_note ?? "No assist totals published yet."}
        footnote={leaders.assists.length ? leaders.assists_note : null}
      />
      <Column
        // The count is the TEAM's — it comes from the scoreline, because no free
        // provider publishes a goalkeeper board. The name shown is the club's
        // number 1, which is the shirt and not the team sheet; clubs whose squad
        // does not settle it fall back to the club's own name.
        title="Clean Sheets"
        unit="clean sheets"
        accent="var(--color-flood)"
        rows={leaders.clean_sheets.map((r) => ({
          player: r.keeper?.name ?? r.team ?? r.player ?? "Unknown",
          team: r.team ?? null,
          value: r.clean_sheets,
          isTeam: !r.keeper,
          playerId: r.keeper?.player_id ? `fd-${r.keeper.player_id}` : undefined,
        }))}
        empty="No matches recorded for this competition yet."
        footnote={leaders.clean_sheets_note ?? null}
      />
    </div>
  );
}

function Column({
  title, rows, accent, unit, empty, footnote,
}: {
  title: React.ReactNode;
  accent: string;
  unit: string;
  empty: string;
  footnote?: string | null;
  rows: { player: string; team: string | null; value: number; playerId?: string; isTeam?: boolean }[];
}) {
  return (
    <div className="min-w-0">
      <h3 className="mb-2.5 text-[10px] uppercase tracking-[0.14em] text-ink-muted">{title}</h3>
      {rows.length === 0 ? (
        <p className="flex items-start gap-2 rounded-lg bg-elevated/50 px-2.5 py-2 text-[11.5px] leading-relaxed text-ink-muted">
          <Info className="mt-[1px] w-3.5 h-3.5 shrink-0" aria-hidden />
          {empty}
        </p>
      ) : (
        <ol className="space-y-1">
          {rows.map((r, i) => (
            <li
              key={`${r.player}-${i}`}
              className="flex items-center gap-2.5 rounded-lg px-1.5 py-1.5 transition-colors duration-150 hover:bg-elevated/70"
            >
              <span className="w-4 shrink-0 text-right font-mono text-[10.5px] tnum text-ink-muted">
                {i + 1}
              </span>
              {r.isTeam
                ? <TeamCrest name={r.player} size={32} />
                : <PlayerAvatar name={r.player} size={32} />}
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[12.5px] font-medium">
                  {r.isTeam ? r.player : (
                    <PlayerLink name={r.player} playerId={r.playerId} />
                  )}
                </span>
                {r.team && (
                  <span className="mt-0.5 flex items-center gap-1.5 text-[10.5px] text-ink-muted">
                    <TeamCrest name={r.team.split(" / ")[0]} size={12} />
                    <span className="truncate">{r.team}</span>
                  </span>
                )}
              </span>
              <span
                className="shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[11.5px] font-semibold tnum"
                style={{ background: `color-mix(in oklab, ${accent} 18%, transparent)`, color: accent }}
                title={`${r.value} ${unit}`}
              >
                {r.value}
              </span>
            </li>
          ))}
        </ol>
      )}
      {footnote && (
        <p className="mt-2 text-[10.5px] leading-relaxed text-ink-muted">{footnote}</p>
      )}
    </div>
  );
}
