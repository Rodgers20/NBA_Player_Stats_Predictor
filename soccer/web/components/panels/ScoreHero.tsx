import type { Match } from "@/lib/types";

/** Crest above name on each side, score centred — the reference arrangement. */
export function ScoreHero({
  match, live, minute, second, homeScore, awayScore,
}: {
  match: Match;
  live?: boolean;
  minute?: number;
  second?: number;
  homeScore?: number;
  awayScore?: number;
}) {
  const hs = homeScore ?? match.home_score;
  const as = awayScore ?? match.away_score;

  return (
    <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-6 py-2">
      <Crest name={match.home_team} accent="var(--color-home)" />

      <div className="text-center px-2">
        <div className="font-display text-[42px] leading-none font-extrabold tracking-tight tnum">
          {hs}
          <span className="mx-3 align-middle text-[26px] font-normal text-ink-muted">-</span>
          {as}
        </div>
        <div className="mt-2.5 flex items-center justify-center gap-2 text-[11px]">
          {live ? (
            <>
              <span className="live-dot" aria-hidden />
              <span className="font-bold tracking-wide text-live">LIVE</span>
              <span className="font-mono tnum text-ink-secondary">
                {minute}:{String(second ?? 0).padStart(2, "0")}
              </span>
            </>
          ) : (
            <span className="rounded bg-elevated px-2 py-0.5 font-medium text-ink-secondary">
              Full Time
            </span>
          )}
        </div>
        <div className="mt-2 text-[11px] text-ink-muted">{match.stadium ?? "—"}</div>
      </div>

      <Crest name={match.away_team} accent="var(--color-away)" />
    </div>
  );
}

function Crest({ name, accent }: { name: string; accent: string }) {
  return (
    <div className="flex flex-col items-center gap-2 min-w-0">
      <div
        className="grid place-items-center w-16 h-16 shrink-0 rounded-2xl text-[15px] font-bold ring-1"
        style={{
          background: `color-mix(in oklab, ${accent} 14%, transparent)`,
          color: accent,
          borderColor: `color-mix(in oklab, ${accent} 42%, transparent)`,
        }}
      >
        {name.slice(0, 3).toUpperCase()}
      </div>
      <div className="max-w-full truncate text-center font-display text-[14px] font-semibold">
        {name}
      </div>
    </div>
  );
}
