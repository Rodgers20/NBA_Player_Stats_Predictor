import { Info } from "lucide-react";
import { FormBadges } from "@/components/charts/FormBadges";
import { TeamLabel } from "@/components/TeamCrest";
import { cn, signed } from "@/lib/format";
import type { LeagueStandings } from "@/lib/types";

export function LeagueTable({
  standings, limit, dense = false,
}: { standings: LeagueStandings; limit?: number; dense?: boolean }) {
  const rows = limit ? standings.rows.slice(0, limit) : standings.rows;

  if (rows.length === 0) {
    return (
      <div className="flex items-start gap-2.5 rounded-lg bg-elevated/60 p-4 text-[12.5px] text-ink-secondary">
        <Info className="mt-0.5 w-4 h-4 shrink-0 text-ink-muted" aria-hidden />
        <p>{standings.note ?? "No completed matches recorded for this season yet."}</p>
      </div>
    );
  }

  return (
    <div>
      {standings.note && (
        <p className="mb-2.5 flex items-start gap-2 rounded-lg bg-elevated/50 px-2.5 py-1.5 text-[11px] text-ink-secondary">
          <Info className="mt-[1px] w-3.5 h-3.5 shrink-0 text-ink-muted" aria-hidden />
          {standings.note}
        </p>
      )}
      <div className="scroll-x -mx-1">
        <table className={cn("w-full text-[12px]", dense ? "min-w-[300px]" : "min-w-[560px]")}>
          <thead>
            <tr className="text-[10px] uppercase tracking-wider text-ink-muted">
              <th className="px-1.5 py-1.5 text-left font-medium w-6">#</th>
              <th className="px-1.5 py-1.5 text-left font-medium">Team</th>
              <th className="px-1.5 py-1.5 text-right font-medium">P</th>
              <th className="px-1.5 py-1.5 text-right font-medium">W</th>
              <th className="px-1.5 py-1.5 text-right font-medium">D</th>
              <th className="px-1.5 py-1.5 text-right font-medium">L</th>
              {!dense && <th className="px-1.5 py-1.5 text-right font-medium">GF</th>}
              {!dense && <th className="px-1.5 py-1.5 text-right font-medium">GA</th>}
              <th className="px-1.5 py-1.5 text-right font-medium">GD</th>
              <th className="px-1.5 py-1.5 text-right font-medium">Pts</th>
              {!dense && <th className="px-1.5 py-1.5 text-left font-medium">Form</th>}
            </tr>
          </thead>
          <tbody className="tnum">
            {rows.map((r) => (
              <tr key={r.team} className="border-t border-stroke/60 hover:bg-elevated/60">
                <td className="px-1.5 py-2 text-ink-muted">{r.position}</td>
                <td className="px-1.5 py-2 font-medium">
                  <TeamLabel name={r.team} size={18} />
                </td>
                <td className="px-1.5 py-2 text-right text-ink-secondary">{r.played}</td>
                <td className="px-1.5 py-2 text-right text-ink-secondary">{r.won}</td>
                <td className="px-1.5 py-2 text-right text-ink-secondary">{r.drawn}</td>
                <td className="px-1.5 py-2 text-right text-ink-secondary">{r.lost}</td>
                {!dense && <td className="px-1.5 py-2 text-right text-ink-secondary">{r.goals_for}</td>}
                {!dense && <td className="px-1.5 py-2 text-right text-ink-secondary">{r.goals_against}</td>}
                <td className="px-1.5 py-2 text-right text-ink-secondary">{signed(r.goal_difference)}</td>
                <td className="px-1.5 py-2 text-right font-semibold">{r.points}</td>
                {!dense && <td className="px-1.5 py-2"><FormBadges form={r.form} /></td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
