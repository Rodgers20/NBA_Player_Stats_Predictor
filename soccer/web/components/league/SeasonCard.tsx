import Link from "next/link";
import { TeamCrest } from "@/components/TeamCrest";
import type { SeasonHonour } from "@/lib/types";

/**
 * One season: who won it, and who finished second.
 *
 * A card with no winner states *why* — the season is still running, or the
 * source never published the final. An empty card would read as a bug, and a
 * guessed winner would be worse: the last match of a Champions League season is
 * not reliably the final, and 2025-26 ends on a drawn semi-final.
 */
export function SeasonCard({
  honour, competitionCode,
}: { honour: SeasonHonour; competitionCode: string }) {
  const href = `/competition?competition=${competitionCode}${
    honour.season_year ? `&season_year=${honour.season_year}` : ""
  }&tab=table`;

  return (
    <Link
      href={href}
      className="flex flex-col gap-3 rounded-lg border border-stroke bg-panel p-4 transition-colors hover:border-stroke-hi hover:bg-elevated"
    >
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="font-display text-[15px] font-bold text-ink">{honour.season}</h3>
        {honour.decided_by === "in_progress" && (
          <span className="rounded-full bg-elevated px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-pitch">
            Live
          </span>
        )}
      </div>

      {honour.winner ? (
        <div className="flex flex-col gap-2.5">
          <Placing team={honour.winner} label={honour.decided_by === "in_progress" ? "Leading" : "Winner"} emphasis />
          {honour.runner_up && (
            <Placing
              team={honour.runner_up}
              label={honour.decided_by === "in_progress" ? "Second" : "Runner-up"}
            />
          )}
        </div>
      ) : (
        <p className="py-2 text-[12px] leading-relaxed text-ink-muted">
          {honour.note ?? "No result recorded."}
        </p>
      )}

      {honour.winner && honour.decided_by === "final" && honour.note && (
        <p className="text-[10.5px] text-ink-muted">{honour.note}</p>
      )}
    </Link>
  );
}

function Placing({
  team, label, emphasis = false,
}: { team: string; label: string; emphasis?: boolean }) {
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <TeamCrest name={team} size={emphasis ? 28 : 24} />
      <span className="min-w-0">
        <span
          className={`block truncate text-[13px] ${
            emphasis ? "font-semibold text-ink" : "text-ink-secondary"
          }`}
        >
          {team}
        </span>
        <span className="block text-[10.5px] text-ink-muted">{label}</span>
      </span>
    </div>
  );
}
