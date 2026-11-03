import Link from "next/link";
import { teamHref } from "@/components/team/TeamLink";
import { TeamCrest } from "@/components/TeamCrest";
import { kickoffTime } from "@/lib/format";
import { FormBadges } from "@/components/charts/FormBadges";
import { LiveScoreCell } from "@/components/league/LiveScoreChip";
import type { LiveMatch, StandingRow } from "@/lib/types";

/**
 * The full competition table, with the qualification bars and Next column.
 *
 * The bar down the left edge encodes what a position *earns* — Champions
 * League, relegation, the knockout play-off round — which is the thing a table
 * is read for and which a bare row order does not convey. Tones are the
 * dashboard's semantic colours rather than each competition's brand palette, so
 * the same bar means the same thing in all three tables, and the legend below
 * names every band actually present.
 */
const TONES: Record<string, string> = {
  primary: "var(--color-pitch)",
  secondary: "var(--color-flood)",
  tertiary: "var(--color-series-3)",
  danger: "var(--color-live)",
};

export function StandingsTable({
  rows, competitionCode, live,
}: {
  rows: StandingRow[];
  competitionCode: string;
  /** In-play matches, so a club playing right now shows its running score. */
  live?: LiveMatch[];
}) {
  if (!rows.length) return null;

  const bands = new Map<string, string>();
  for (const r of rows) {
    if (r.qualification && r.qualification_tone) {
      bands.set(r.qualification, r.qualification_tone);
    }
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] border-collapse text-[13px]">
          <caption className="sr-only">
            League table, ordered by points then goal difference
          </caption>
          <thead>
            <tr className="border-b border-stroke text-[11px] uppercase tracking-wide text-ink-muted">
              <th scope="col" className="w-8 py-2 pl-3 pr-1 text-left font-medium">#</th>
              <th scope="col" className="py-2 pl-1 pr-3 text-left font-medium">Team</th>
              <th scope="col" className="px-2 py-2 text-right font-medium" title="Played">PL</th>
              <th scope="col" className="px-2 py-2 text-right font-medium" title="Won">W</th>
              <th scope="col" className="px-2 py-2 text-right font-medium" title="Drawn">D</th>
              <th scope="col" className="px-2 py-2 text-right font-medium" title="Lost">L</th>
              <th scope="col" className="px-2 py-2 text-right font-medium" title="Goals for and against">+/–</th>
              <th scope="col" className="px-2 py-2 text-right font-medium" title="Goal difference">GD</th>
              <th scope="col" className="px-2 py-2 text-right font-medium" title="Points">PTS</th>
              <th scope="col" className="px-2 py-2 text-left font-medium">Form</th>
              <th scope="col" className="px-2 py-2 text-center font-medium">Next</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const tone = r.qualification_tone ? TONES[r.qualification_tone] : undefined;
              return (
                <tr key={r.team} className="border-b border-stroke/50 last:border-b-0 hover:bg-elevated">
                  <td className="relative py-2 pl-3 pr-1 text-right tabular-nums text-ink-secondary">
                    {tone && (
                      <span
                        aria-hidden
                        className="absolute inset-y-0 left-0 w-[3px] rounded-full"
                        style={{ background: tone }}
                      />
                    )}
                    {r.position}
                  </td>
                  <td className="max-w-[220px] py-2 pl-1 pr-3">
                    <span className="flex min-w-0 items-center gap-2">
                      <Link
                        href={teamHref(r.team, competitionCode)}
                        className="flex min-w-0 items-center gap-2 transition-colors hover:text-pitch"
                      >
                        <TeamCrest name={r.team} size={20} />
                        <span className="truncate">{r.team}</span>
                      </Link>
                      <LiveScoreCell team={r.team} live={live} />
                    </span>
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums text-ink-secondary">{r.played}</td>
                  <td className="px-2 py-2 text-right tabular-nums text-ink-secondary">{r.won}</td>
                  <td className="px-2 py-2 text-right tabular-nums text-ink-secondary">{r.drawn}</td>
                  <td className="px-2 py-2 text-right tabular-nums text-ink-secondary">{r.lost}</td>
                  <td className="whitespace-nowrap px-2 py-2 text-right tabular-nums text-ink-secondary">
                    {r.goals_for}-{r.goals_against}
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums text-ink-secondary">
                    {r.goal_difference > 0 ? `+${r.goal_difference}` : r.goal_difference}
                  </td>
                  <td className="px-2 py-2 text-right font-semibold tabular-nums text-ink">{r.points}</td>
                  <td className="px-2 py-2"><FormBadges form={r.form} /></td>
                  <td className="px-2 py-2">
                    {r.next ? (
                      <span
                        className="flex items-center justify-center"
                        title={`${r.next.at_home ? "vs" : "at"} ${r.next.opponent} · ${
                          kickoffTime(null, r.next.date, r.next.time)
                        }`}
                      >
                        <TeamCrest name={r.next.opponent} size={20} />
                      </span>
                    ) : (
                      <span className="block text-center text-ink-muted">–</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {bands.size > 0 && (
        <ul className="flex flex-wrap gap-x-5 gap-y-1.5 px-3 py-3 text-[11px] text-ink-muted">
          {[...bands].map(([label, tone]) => (
            <li key={label} className="flex items-center gap-1.5">
              <span aria-hidden className="h-1.5 w-1.5 rounded-full"
                    style={{ background: TONES[tone] }} />
              {label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
