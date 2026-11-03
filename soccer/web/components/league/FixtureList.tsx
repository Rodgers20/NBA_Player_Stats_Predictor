import { CalendarClock } from "lucide-react";
import { fmtDate } from "@/lib/format";
import { TeamLabel } from "@/components/TeamCrest";
import { kickoffTime } from "@/lib/format";
import type { Fixture } from "@/lib/types";

export function FixtureList({ fixtures, limit }: { fixtures: Fixture[]; limit?: number }) {
  const rows = limit ? fixtures.slice(0, limit) : fixtures;
  if (rows.length === 0) {
    return (
      <p className="flex items-center gap-2 py-6 text-center text-[12px] text-ink-muted">
        <CalendarClock className="w-4 h-4" aria-hidden />
        No scheduled fixtures in the next few days.
      </p>
    );
  }
  const byDate = rows.reduce<Record<string, Fixture[]>>((acc, f) => {
    (acc[f.date] ??= []).push(f);
    return acc;
  }, {});

  return (
    <div className="space-y-3">
      {Object.entries(byDate).map(([date, list]) => (
        <div key={date}>
          <div className="mb-1.5 text-[10.5px] font-medium text-ink-muted">{fmtDate(date)}</div>
          <ul className="space-y-0.5">
            {list.map((f) => (
              <li
                key={f.fixture_key}
                className="grid grid-cols-[44px_1fr_auto_1fr] items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-elevated/70"
              >
                <span className="font-mono text-[10.5px] tnum text-ink-muted">
                  {f.live_status === "live" ? (
                    <span className="flex items-center gap-1 text-live font-semibold">
                      <span className="live-dot !w-1.5 !h-1.5" aria-hidden />LIVE
                    </span>
                  ) : kickoffTime(f.kickoff_utc, f.date, f.time)}
                </span>
                <TeamLabel name={f.home_team} size={16} align="right" className="justify-start text-[12px]" />
                <span className="rounded bg-elevated px-1.5 py-0.5 font-mono text-[10px] text-ink-muted">v</span>
                <TeamLabel name={f.away_team} size={16} className="text-[12px]" />
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
