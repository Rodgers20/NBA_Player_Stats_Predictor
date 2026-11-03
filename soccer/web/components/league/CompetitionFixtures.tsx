import Link from "next/link";
import { TeamCrest } from "@/components/TeamCrest";
import { kickoffDate, kickoffTime, localDayKey } from "@/lib/format";
import type { Fixture } from "@/lib/types";

/**
 * Upcoming fixtures, grouped by kick-off date.
 *
 * Grouping is by *date* rather than by matchday because a round is spread over
 * three or four days and a reader scanning "what is on" thinks in days.
 *
 * Both the grouping and the times use the viewer's zone, not UTC. A 19:00 UTC
 * kick-off is 3:00 PM in Albany, and a late European game can fall on the
 * previous day locally — so grouping on the raw UTC date would file it under
 * the wrong heading.
 */
export function CompetitionFixtures({
  fixtures, limit,
}: { fixtures: Fixture[]; limit?: number }) {
  const shown = limit ? fixtures.slice(0, limit) : fixtures;
  if (!shown.length) {
    return (
      <p className="px-3 py-6 text-center text-[12.5px] text-ink-muted">
        No upcoming fixtures published for this competition.
      </p>
    );
  }

  const byDate = new Map<string, Fixture[]>();
  for (const f of shown) {
    const key = localDayKey(f.kickoff_utc, f.date, f.time);
    if (!byDate.has(key)) byDate.set(key, []);
    byDate.get(key)!.push(f);
  }

  return (
    <div>
      {[...byDate].map(([date, group]) => (
        <section key={date}>
          <h3 className="sticky top-0 z-10 bg-elevated/95 px-3 py-1.5 text-[11.5px] font-semibold text-ink-secondary backdrop-blur">
            {kickoffDate(group[0]?.kickoff_utc, group[0]?.date, group[0]?.time)}
          </h3>
          <ul>
            {group.map((f) => (
              <li key={`${f.date}-${f.home_team}-${f.away_team}`}>
                <Link
                  href={`/matches?competition=${f.competition_code}`}
                  className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 border-b border-stroke/50 px-3 py-2.5 transition-colors last:border-b-0 hover:bg-elevated"
                >
                  <span className="flex min-w-0 items-center justify-end gap-2 text-right text-[13px]">
                    <span className="truncate">{f.home_team}</span>
                    <TeamCrest name={f.home_team} size={20} />
                  </span>
                  <span className="whitespace-nowrap rounded bg-elevated px-2 py-0.5 text-[11.5px] tabular-nums text-ink-secondary">
                    {kickoffTime(f.kickoff_utc, f.date, f.time)}
                  </span>
                  <span className="flex min-w-0 items-center gap-2 text-[13px]">
                    <TeamCrest name={f.away_team} size={20} />
                    <span className="truncate">{f.away_team}</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
