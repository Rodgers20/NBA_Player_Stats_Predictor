import { CalendarDays, MapPin } from "lucide-react";
import { TeamCrest } from "@/components/TeamCrest";
import { displayTeam, fmtDate } from "@/lib/format";
import type { LeagueMatch } from "@/lib/types";

/** Featured result: crest / score / crest, then a mirrored stat comparison. */
export function FeaturedMatch({ match }: { match: LeagueMatch }) {
  const stats: { label: string; home: number | null; away: number | null; pct?: boolean }[] = [
    { label: "Expected Goals (xG)", home: match.home_xg, away: match.away_xg },
    { label: "Shots", home: match.home_shots, away: match.away_shots },
    { label: "Shots on Target", home: match.home_sot, away: match.away_sot },
    { label: "Corners", home: match.home_corners, away: match.away_corners },
    { label: "Fouls", home: match.home_fouls, away: match.away_fouls },
    { label: "Yellow Cards", home: match.home_yellow, away: match.away_yellow },
    { label: "Red Cards", home: match.home_red, away: match.away_red },
  ].filter((s) => s.home != null || s.away != null);

  return (
    <div>
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
        <div className="flex min-w-0 flex-col items-center gap-2">
          <TeamCrest name={match.home_team} size={40} />
          <span className="max-w-full truncate text-center text-[12.5px] font-medium" title={match.home_team}>
            {displayTeam(match.home_team)}
          </span>
        </div>
        <div className="px-2 text-center">
          <div className="font-display text-[30px] font-extrabold leading-none tnum">
            {match.home_goals}
            <span className="mx-2 align-middle text-[20px] font-normal text-ink-muted">–</span>
            {match.away_goals}
          </div>
          <div className="mt-1.5 text-[10px] text-ink-muted">Full Time</div>
          {match.ht_home_goals != null && (
            <div className="mt-0.5 font-mono text-[10px] tnum text-ink-muted">
              HT {match.ht_home_goals}–{match.ht_away_goals}
            </div>
          )}
        </div>
        <div className="flex min-w-0 flex-col items-center gap-2">
          <TeamCrest name={match.away_team} size={40} />
          <span className="max-w-full truncate text-center text-[12.5px] font-medium" title={match.away_team}>
            {displayTeam(match.away_team)}
          </span>
        </div>
      </div>

      <div className="mt-3 flex items-center justify-center gap-4 border-t border-stroke pt-2.5 text-[11px] text-ink-muted">
        <span className="flex items-center gap-1.5">
          <CalendarDays className="w-3.5 h-3.5" aria-hidden />
          {fmtDate(match.date)}
        </span>
        {match.time && (
          <span className="flex items-center gap-1.5 font-mono tnum">
            <MapPin className="w-3.5 h-3.5" aria-hidden />
            {match.time}
          </span>
        )}
      </div>

      {stats.length > 0 && (
        <div className="mt-3 space-y-2 border-t border-stroke pt-3">
          <div className="mb-1 text-center text-[10px] uppercase tracking-[0.14em] text-ink-muted">
            Match Stats
          </div>
          {stats.map((s) => {
            const h = s.home ?? 0;
            const a = s.away ?? 0;
            const total = h + a;
            const hp = total > 0 ? (h / total) * 100 : 50;
            const ap = total > 0 ? (a / total) * 100 : 50;
            const fmt = (v: number | null) =>
              v == null ? "—" : Number.isInteger(v) ? String(v) : v.toFixed(2);
            return (
              <div key={s.label}>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-10 shrink-0 text-right font-mono tnum">{fmt(s.home)}</span>
                  <span className="flex-1 text-center text-ink-secondary">{s.label}</span>
                  <span className="w-10 shrink-0 font-mono tnum">{fmt(s.away)}</span>
                </div>
                <div className="mt-1 flex items-center gap-[2px]">
                  <div className="flex flex-1 justify-end h-[4px] rounded-l-full bg-stroke/60 overflow-hidden">
                    <div className="h-full rounded-l-full" style={{ width: `${hp}%`, background: "var(--color-home)" }} />
                  </div>
                  <div className="flex flex-1 h-[4px] rounded-r-full bg-stroke/60 overflow-hidden">
                    <div className="h-full rounded-r-full" style={{ width: `${ap}%`, background: "var(--color-away)" }} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
