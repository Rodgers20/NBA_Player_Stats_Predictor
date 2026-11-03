import Link from "next/link";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import { TeamCrest } from "@/components/TeamCrest";

/**
 * A short leaderboard card — top scorers, top assists, best defences.
 *
 * FotMob's third card is "Top rated", which needs its own proprietary match
 * rating. There is no free source for one, and inventing a number to sit beside
 * real goals and assists would be worse than leaving it out, so this project
 * shows clean sheets there instead: computed exactly from the results lake.
 */
export type LeaderEntry = {
  name: string;
  team: string | null;
  value: number | string;
  playerId?: string;
  /** A team board draws a crest and does not link to a player page. */
  isTeam?: boolean;
};

export function StatLeaders({
  title, entries, unit, accent = "var(--color-pitch)", allHref, emptyNote,
}: {
  title: string;
  entries: LeaderEntry[];
  unit: string;
  accent?: string;
  allHref?: string;
  emptyNote?: string;
}) {
  return (
    <div className="flex min-w-0 flex-col">
      <h3 className="border-b border-stroke px-3 py-2.5 font-display text-[13px] font-semibold text-ink">
        {title}
      </h3>

      {entries.length === 0 ? (
        <p className="px-3 py-6 text-center text-[12px] text-ink-muted">
          {emptyNote ?? "Nothing published yet."}
        </p>
      ) : (
        <ol className="flex-1">
          {entries.map((e, i) => (
            <li
              key={`${e.name}-${i}`}
              className="flex items-center gap-2.5 border-b border-stroke/50 px-3 py-2.5 last:border-b-0"
            >
              {e.isTeam
                ? <TeamCrest name={e.name} size={32} />
                : <PlayerAvatar name={e.name} size={32} />}
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[12.5px] font-medium text-ink">
                  {e.isTeam ? e.name : <PlayerLink name={e.name} playerId={e.playerId} />}
                </span>
                {e.team && !e.isTeam && (
                  <span className="mt-0.5 flex items-center gap-1.5 text-[10.5px] text-ink-muted">
                    <TeamCrest name={e.team} size={12} />
                    <span className="truncate">{e.team}</span>
                  </span>
                )}
              </span>
              <span
                className="shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[11.5px] font-semibold tabular-nums"
                style={{
                  background: `color-mix(in oklab, ${accent} 18%, transparent)`,
                  color: accent,
                }}
                title={`${e.value} ${unit}`}
              >
                {e.value}
              </span>
            </li>
          ))}
        </ol>
      )}

      {allHref && entries.length > 0 && (
        <Link
          href={allHref}
          className="border-t border-stroke px-3 py-2 text-center text-[12px] font-medium text-ink-muted transition-colors hover:text-pitch"
        >
          All
        </Link>
      )}
    </div>
  );
}
