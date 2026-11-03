import { CalendarDays, Clock, Trophy, UserRound } from "lucide-react";
import { TeamCrest } from "@/components/TeamCrest";
import { displayTeam, fmtDate } from "@/lib/format";
import type { Fixture, LiveMatch } from "@/lib/types";

/**
 * Match hero for a live or upcoming fixture.
 *
 * There is no in-play feed without an API key, so this shows what is genuinely
 * known before kick-off — the clock, the venue-neutral fixture identity, the
 * referee and the market's pre-match view — and says plainly that the running
 * score is not available, rather than displaying a fabricated 0–0.
 */
export function MatchCentreHero({
  fixture, live, competition,
}: {
  fixture: Fixture;
  live: LiveMatch | null;
  competition: string;
}) {
  const isLive = Boolean(live);
  return (
    <div className="panel px-5 py-5">
      <div className="mb-3 flex items-center gap-1.5 text-[11px]">
        <span className="text-pitch">{competition}</span>
        <span className="text-ink-muted">›</span>
        <span className="text-ink-secondary">
          {displayTeam(fixture.home_team)} vs {displayTeam(fixture.away_team)}
        </span>
      </div>

      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-5">
        <Side name={fixture.home_team} />
        <div className="px-2 text-center">
          {isLive ? (
            <>
              <div className="font-display text-[15px] font-semibold text-ink-secondary">
                {live?.home_goals ?? "—"} – {live?.away_goals ?? "—"}
              </div>
              <div className="mt-2 flex items-center justify-center gap-2 text-[12px]">
                <span className="live-dot" aria-hidden />
                <span className="font-bold tracking-wide text-live">LIVE</span>
                <span className="font-mono tnum text-ink-secondary">
                  {live?.minutes_elapsed != null ? `${live.minutes_elapsed}'` : ""}
                </span>
              </div>
              <div className="mt-1.5 text-[10.5px] text-ink-muted">
                ESPN live score
              </div>
            </>
          ) : (
            <>
              <div className="font-display text-[30px] font-extrabold leading-none tnum">
                {fixture.time ?? "--:--"}
              </div>
              <div className="mt-1.5 text-[10.5px] uppercase tracking-wider text-ink-muted">
                Kick-off
              </div>
            </>
          )}
        </div>
        <Side name={fixture.away_team} />
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-2 border-t border-stroke pt-4 text-[11.5px] sm:grid-cols-4">
        <Meta icon={Trophy} label="Competition" value={competition} />
        <Meta icon={CalendarDays} label="Date" value={fmtDate(fixture.date)} />
        <Meta icon={Clock} label="Kick-off" value={fixture.time ?? "—"} />
        <Meta icon={UserRound} label="Referee" value={fixture.referee ?? "Not announced"} />
      </dl>

      {fixture.odds_home && (
        <div className="mt-3 flex items-center gap-2 border-t border-stroke pt-3 text-[11px]">
          <span className="text-ink-muted">Market</span>
          <Odd label={displayTeam(fixture.home_team)} value={fixture.odds_home} />
          <Odd label="Draw" value={fixture.odds_draw} />
          <Odd label={displayTeam(fixture.away_team)} value={fixture.odds_away} />
        </div>
      )}
    </div>
  );
}

function Side({ name }: { name: string }) {
  return (
    <div className="flex min-w-0 flex-col items-center gap-2.5">
      <TeamCrest name={name} size={56} />
      <span className="max-w-full truncate text-center font-display text-[14px] font-semibold" title={name}>
        {displayTeam(name)}
      </span>
    </div>
  );
}

function Meta({ icon: Icon, label, value }: { icon: typeof Trophy; label: string; value: string }) {
  return (
    <div className="flex min-w-0 items-start gap-2">
      <Icon className="mt-[2px] w-3.5 h-3.5 shrink-0 text-ink-muted" aria-hidden />
      <div className="min-w-0">
        <dt className="text-[10px] uppercase tracking-wider text-ink-muted">{label}</dt>
        <dd className="truncate text-ink-secondary" title={value}>{value}</dd>
      </div>
    </div>
  );
}

function Odd({ label, value }: { label: string; value: number | null }) {
  return (
    <span className="flex items-center gap-1.5 rounded-md bg-elevated px-2 py-1">
      <span className="max-w-[110px] truncate text-ink-secondary">{label}</span>
      <span className="font-mono font-semibold tnum">{value?.toFixed(2) ?? "—"}</span>
    </span>
  );
}
