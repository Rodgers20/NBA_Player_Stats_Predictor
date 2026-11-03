import { CalendarDays, MapPin, Trophy, UserRound } from "lucide-react";
import { fmtDate } from "@/lib/format";
import type { Match } from "@/lib/types";

export function MatchHero({
  match, live, minute, homeScore, awayScore,
}: {
  match: Match;
  live?: boolean;
  minute?: number;
  homeScore?: number;
  awayScore?: number;
}) {
  const hs = homeScore ?? match.home_score;
  const as = awayScore ?? match.away_score;
  return (
    <div className="panel px-5 py-5">
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-4">
        <TeamSide name={match.home_team} align="right" accent="var(--color-home)" />

        <div className="text-center">
          <div className="font-display text-[38px] leading-none font-extrabold tracking-tight tnum">
            {hs} <span className="mx-1 text-ink-muted font-normal">-</span> {as}
          </div>
          <div className="mt-2 flex items-center justify-center gap-2 text-[11px]">
            {live ? (
              <>
                <span className="live-dot" aria-hidden />
                <span className="font-semibold text-live">LIVE</span>
                <span className="font-mono tnum text-ink-secondary">{minute}&apos;</span>
              </>
            ) : (
              <span className="rounded px-1.5 py-0.5 bg-elevated text-ink-secondary font-medium">
                Full Time
              </span>
            )}
          </div>
          <div className="mt-1.5 text-[11px] text-ink-muted">
            {match.stadium ?? "—"}
          </div>
        </div>

        <TeamSide name={match.away_team} align="left" accent="var(--color-away)" />
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-2 border-t border-stroke pt-4 text-[11.5px] sm:grid-cols-4">
        <Meta icon={Trophy} label="Competition" value={`${match.competition} · ${match.competition_stage}`} />
        <Meta icon={CalendarDays} label="Date" value={fmtDate(match.match_date)} />
        <Meta icon={UserRound} label="Referee" value={match.referee ?? "—"} />
        <Meta icon={MapPin} label="Venue" value={match.stadium ?? "—"} />
      </dl>
    </div>
  );
}

function TeamSide({ name, align, accent }: { name: string; align: "left" | "right"; accent: string }) {
  return (
    <div className={`flex items-center gap-3 ${align === "right" ? "flex-row-reverse text-right" : "text-left"}`}>
      <div
        className="grid place-items-center w-12 h-12 shrink-0 rounded-xl text-[13px] font-bold ring-1"
        style={{
          background: `color-mix(in oklab, ${accent} 15%, transparent)`,
          color: accent,
          borderColor: accent,
        }}
      >
        {name.slice(0, 3).toUpperCase()}
      </div>
      <div className="min-w-0">
        <div className="truncate font-display text-[15px] font-semibold">{name}</div>
      </div>
    </div>
  );
}

function Meta({ icon: Icon, label, value }: { icon: typeof Trophy; label: string; value: string }) {
  return (
    <div className="flex items-start gap-2 min-w-0">
      <Icon className="mt-[2px] w-3.5 h-3.5 shrink-0 text-ink-muted" aria-hidden />
      <div className="min-w-0">
        <dt className="text-[10px] uppercase tracking-wider text-ink-muted">{label}</dt>
        <dd className="truncate text-ink-secondary">{value}</dd>
      </div>
    </div>
  );
}
