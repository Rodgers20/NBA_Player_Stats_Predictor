"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";
import { Bell, Menu, Search } from "lucide-react";
import { NAV } from "./Sidebar";
import { CompetitionCrest } from "@/components/CompetitionCrest";
import { withCompetition } from "@/lib/competition";
import type { League } from "@/lib/types";

export function TopBar({ title, breadcrumb, action, liveTotal = 0, competition, leagues = [] }: {
  title: string; breadcrumb?: React.ReactNode; action?: React.ReactNode;
  liveTotal?: number; competition: string;
  /** The competition pages, which on desktop live in the sidebar. */
  leagues?: League[];
}) {
  const menu = useRef<HTMLDetailsElement>(null);
  const pathname = usePathname();
  useEffect(() => { if (menu.current) menu.current.open = false; }, [pathname, competition]);
  return (
    <header className="sticky top-0 z-30 flex min-h-16 flex-wrap items-center gap-2 border-b border-stroke bg-base/95 px-4 py-2 backdrop-blur lg:px-6">
      <details ref={menu} className="relative lg:hidden" onKeyDown={(e) => {
        if (e.key === "Escape" && menu.current) { menu.current.open = false; menu.current.querySelector("summary")?.focus(); }
      }}>
        <summary aria-label="Open navigation" className="grid h-11 w-11 cursor-pointer list-none place-items-center rounded-lg hover:bg-elevated"><Menu size={20} aria-hidden /></summary>
        <nav aria-label="Mobile navigation" className="absolute left-0 top-12 z-50 max-h-[75vh] w-64 overflow-y-auto rounded-xl border border-stroke-hi bg-panel p-2 shadow-2xl">
          {/* eslint-disable-next-line @next/next/no-html-link-for-pages -- main site is a separate deployment */}
          <a href="/" className="flex min-h-11 items-center gap-3 rounded-lg px-3 text-sm text-ink-secondary hover:bg-elevated"><span aria-hidden>←</span>Court Vision</a>
          {NAV.map(({href, label, icon: Icon}) => <Link key={href} href={withCompetition(href, competition)} aria-current={pathname === href ? "page" : undefined} className="flex min-h-11 items-center gap-3 rounded-lg px-3 text-sm hover:bg-elevated aria-[current=page]:text-pitch"><Icon size={17} aria-hidden />{label}</Link>)}
          {leagues.length > 0 && (
            <>
              <div className="px-3 pb-1 pt-3 text-[10px] uppercase tracking-[0.16em] text-ink-muted">
                Competitions
              </div>
              {leagues.map((l) => (
                <Link
                  key={l.code}
                  href={withCompetition("/competition", l.code)}
                  aria-current={pathname.startsWith("/competition") && competition === l.code ? "page" : undefined}
                  className="flex min-h-11 items-center gap-3 rounded-lg px-3 text-sm hover:bg-elevated aria-[current=page]:text-pitch"
                >
                  <CompetitionCrest code={l.code} size={17} emblem={l.emblem} />
                  <span className="truncate">{l.name}</span>
                </Link>
              ))}
            </>
          )}
        </nav>
      </details>
      <div className="min-w-0 flex-1">
        <h1 className="truncate font-display text-[16px] font-semibold">{title}</h1>
        {breadcrumb && <div className="text-xs text-ink-muted">{breadcrumb}</div>}
      </div>
      <Link href={withCompetition("/teams", competition)} aria-label="Search teams" className="grid h-11 w-11 place-items-center rounded-lg text-ink-secondary hover:bg-elevated"><Search size={18} aria-hidden /></Link>
      <Link href={withCompetition("/alerts", competition)} aria-label="Match updates" className="relative grid h-11 w-11 place-items-center rounded-lg text-ink-secondary hover:bg-elevated"><Bell size={18} aria-hidden />{liveTotal > 0 && <span className="absolute right-0 top-0 rounded-full bg-live px-1 text-xs text-white">{liveTotal}</span>}</Link>
      {action && <div className="w-full min-w-0 sm:order-none sm:w-auto">{action}</div>}
    </header>
  );
}
