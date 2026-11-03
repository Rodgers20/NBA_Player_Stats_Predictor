"use client";

import Link from "next/link";
import { CompetitionCrest } from "@/components/CompetitionCrest";
import { usePathname, useSearchParams } from "next/navigation";
import {
  Activity, BarChart3, Bell, CalendarDays, CircleDot, LayoutDashboard,
  Newspaper, Settings, Shield, Sparkles, Users,
} from "lucide-react";
import { cn } from "@/lib/format";
import { withCompetition } from "@/lib/competition";
import type { League } from "@/lib/types";

export const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/live", label: "Live Matches", icon: Activity, live: true },
  { href: "/matches", label: "Matches", icon: CalendarDays },
  { href: "/teams", label: "Teams", icon: Shield },
  { href: "/players", label: "Players", icon: Users },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/predictions", label: "Predictions", icon: Sparkles },
  { href: "/reports", label: "Reports", icon: Newspaper },
  { href: "/alerts", label: "Match updates", icon: Bell },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar({
  leagues, selected, liveTotal, liveByCompetition,
}: {
  leagues: League[];
  selected: string;
  liveTotal: number;
  liveByCompetition: Record<string, number>;
}) {
  const pathname = usePathname();
  const params = useSearchParams();
  const current = params.get("competition") ?? selected;
  const onCompetitionPage = pathname.startsWith("/competition");

  return (
    <aside className="hidden lg:flex w-[236px] shrink-0 flex-col border-r border-stroke bg-[var(--color-void)]/70 backdrop-blur">
      <Link href={withCompetition("/", current)} className="flex items-center gap-2.5 px-5 h-16 border-b border-stroke hover:bg-elevated/40">
        <div className="grid place-items-center w-8 h-8 rounded-lg bg-pitch/15 ring-1 ring-pitch/30">
          <CircleDot className="w-[18px] h-[18px] text-pitch" />
        </div>
        <div className="leading-none">
          <div className="font-display font-extrabold tracking-tight text-[15px]">FUTBALL</div>
          <div className="text-[9px] uppercase tracking-[0.18em] text-ink-muted mt-1">
            Football Intelligence
          </div>
        </div>
      </Link>

      {/* Plain anchor on purpose: the main site is a different deployment, not a Next route here. */}
      {/* eslint-disable-next-line @next/next/no-html-link-for-pages -- main site is a separate deployment */}
      <a href="/" className="mx-3 mt-3 flex items-center gap-2 rounded-lg px-3 py-2 text-[12px] text-ink-secondary hover:bg-elevated hover:text-ink">
        <span aria-hidden>←</span> Court Vision
      </a>

      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-0.5">
        <SectionLabel>Navigation</SectionLabel>
        {NAV.map(({ href, label, icon: Icon, live }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          const badge = live && liveTotal > 0 ? String(liveTotal) : null;
          return (
            <Link
              key={href}
              href={withCompetition(href, current)}
              aria-current={active ? "page" : undefined}
              className={cn(
                "group relative flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] transition-colors",
                active
                  ? "bg-pitch/10 text-ink font-medium"
                  : "text-ink-secondary hover:bg-elevated hover:text-ink",
              )}
            >
              {active && <span className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-r bg-pitch" />}
              <Icon className={cn("w-4 h-4 shrink-0", active ? "text-pitch" : "text-ink-muted group-hover:text-ink-secondary")} />
              <span className="truncate">{label}</span>
              {badge && (
                <span className="ml-auto flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-semibold bg-live/90 text-white tnum">
                  <span className="live-dot !w-1.5 !h-1.5" aria-hidden />
                  {badge}
                </span>
              )}
            </Link>
          );
        })}

        <SectionLabel className="pt-5">Competitions</SectionLabel>
        {leagues.map((l) => {
          // Highlighted only while its own page is open. These used to re-scope
          // whatever page you were on, which made the list read as a filter
          // duplicating the switcher every page already carries in its header.
          const active = onCompetitionPage && current === l.code;
          const live = liveByCompetition[l.code] ?? 0;
          return (
            <Link
              key={l.code}
              href={withCompetition("/competition", l.code)}
              aria-current={active ? "page" : undefined}
              className={cn(
                "group relative flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] transition-colors",
                active
                  ? "bg-flood/10 text-ink font-medium"
                  : "text-ink-secondary hover:bg-elevated hover:text-ink",
              )}
            >
              {active && <span className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-r bg-flood" />}
              <CompetitionCrest code={l.code} size={18} emblem={l.emblem} />
              <span className="truncate">{l.name}</span>
              {live > 0 ? (
                <span className="ml-auto flex items-center gap-1 text-[10px] font-semibold text-live tnum">
                  <span className="live-dot !w-1.5 !h-1.5" aria-hidden />{live}
                </span>
              ) : (
                <span className="ml-auto text-[10px] text-ink-muted tnum">
                  {l.displayed_season ?? "—"}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-stroke">
        <p className="text-xs font-semibold text-pitch">Built on match evidence</p>
        <p className="mt-2 text-xs leading-relaxed text-ink-secondary">Explore the data behind every forecast.</p>
        <Link href={withCompetition("/predictions", current)} className="mt-2 inline-flex min-h-11 items-center text-xs text-flood transition-colors hover:text-ink">View model performance →</Link>
      </div>
    </aside>
  );
}

function SectionLabel({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("px-3 pb-1.5 text-[10px] uppercase tracking-[0.16em] text-ink-muted", className)}>
      {children}
    </div>
  );
}
