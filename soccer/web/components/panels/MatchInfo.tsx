import { CalendarDays, CloudSun, Clock, Trophy, UserRound, Users } from "lucide-react";
import { fmtDate } from "@/lib/format";
import type { Match } from "@/lib/types";

/**
 * The "Match Info" panel from the reference design.
 *
 * Weather and attendance are shown because the reference has them, but
 * StatsBomb open data does not carry either field — so they read "Not in feed"
 * rather than being invented. Fabricating them would undermine every other
 * number on the page.
 */
export function MatchInfo({ match }: { match: Match }) {
  const rows: { icon: typeof Trophy; label: string; value: string; muted?: boolean }[] = [
    { icon: Trophy, label: "Competition", value: `${match.competition}` },
    { icon: CalendarDays, label: "Date", value: fmtDate(match.match_date) },
    { icon: Clock, label: "Kick-off", value: match.kick_off?.slice(0, 5) ?? "—" },
    { icon: UserRound, label: "Referee", value: match.referee ?? "—" },
    { icon: CloudSun, label: "Weather", value: "Not in feed", muted: true },
    { icon: Users, label: "Attendance", value: "Not in feed", muted: true },
  ];

  return (
    <dl className="space-y-2.5">
      {rows.map(({ icon: Icon, label, value, muted }) => (
        <div key={label} className="flex items-center gap-2.5 text-[12px]">
          <Icon className="w-3.5 h-3.5 shrink-0 text-ink-muted" aria-hidden />
          <dt className="text-ink-secondary">{label}</dt>
          <dd
            className={`ml-auto truncate text-right ${muted ? "text-ink-muted italic" : "font-medium"}`}
            title={value}
          >
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}
