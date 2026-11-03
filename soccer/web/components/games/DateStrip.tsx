import Link from "next/link";
import { cn } from "@/lib/format";

const fmt = (date: string, opts: Intl.DateTimeFormatOptions) =>
  new Intl.DateTimeFormat("en-US", { ...opts, timeZone: "UTC" }).format(new Date(`${date}T12:00:00Z`));

/** A ribbon of match days for the selected competition; each day is a link, so the page stays server-rendered. */
export function DateStrip({
  days, selected, competition,
}: { days: { date: string; count: number }[]; selected: string; competition: string }) {
  return (
    <nav aria-label="Match days" className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-2" style={{ scrollbarWidth: "thin" }}>
      {days.map((d) => {
        const active = d.date === selected;
        return (
          <Link
            key={d.date}
            href={`/games?competition=${encodeURIComponent(competition)}&date=${d.date}`}
            aria-current={active ? "date" : undefined}
            className={cn(
              "press flex min-w-[64px] shrink-0 flex-col items-center gap-0.5 rounded-lg border px-3 py-2 transition-colors",
              active
                ? "border-teal-400 bg-teal-400 text-bg-root"
                : "border-border-card bg-bg-card text-text-sec hover:bg-surface-hover hover:text-text-pri",
            )}
          >
            <span className="font-mono text-[10px] font-bold uppercase tracking-wider">{fmt(d.date, { weekday: "short" })}</span>
            <span className="font-display text-[24px] font-extrabold leading-none">{fmt(d.date, { day: "numeric" })}</span>
            <span className="font-mono text-[10px] font-bold uppercase">{fmt(d.date, { month: "short" })}</span>
            <span className={cn("mt-0.5 text-[10px] font-semibold", active ? "text-bg-root/80" : "text-text-dim")}>
              {d.count} {d.count === 1 ? "game" : "games"}
            </span>
          </Link>
        );
      })}
    </nav>
  );
}
