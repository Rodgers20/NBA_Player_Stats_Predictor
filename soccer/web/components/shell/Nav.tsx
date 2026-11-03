"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { CalendarDays, ChartNoAxesCombined, Shield, Sparkles, Trophy } from "lucide-react";
import { withCompetition } from "@/lib/competition";
import type { League } from "@/lib/types";

export const NAV = [
  { href: "/games", label: "Games", icon: CalendarDays },
  { href: "/props", label: "Props", icon: Sparkles },
  { href: "/players", label: "Players", icon: ChartNoAxesCombined },
  { href: "/teams", label: "Teams", icon: Shield },
  { href: "/competition", label: "League", icon: Trophy },
] as const;

/** Short pill labels; a long league name does not fit a 11px mono pill. */
const SHORT: Record<string, string> = { E0: "PREMIER LEAGUE", SP1: "LA LIGA", UCL: "CHAMPIONS LEAGUE", D1: "BUNDESLIGA", MLS: "MLS" };

/** Which nav entry a pathname belongs to (detail pages count as their list). */
const ROOT: Record<string, string> = { games: "/games", props: "/props", players: "/players", team: "/teams", teams: "/teams", match: "/matches", matches: "/matches", competition: "/competition" };
const rootOf = (path: string) => ROOT[path.split("/")[1] ?? ""] ?? "/games";
const isActive = (path: string, href: string) => path === href || path.startsWith(`${href}/`) || (href === "/teams" && path === "/team");

/** Desktop icon rail (hidden below 901px, where the floating pill takes over). */
export function Rail({ competition }: { competition: string }) {
  const path = usePathname();
  return (
    <nav aria-label="Main navigation" className="cv-rail">
      {/* Plain anchor: the main site is a different deployment, not a Next route here. */}
      {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
      <a href="/" aria-label="Court Vision home" className="cv-ring" />
      {NAV.map(({ href, label, icon: Icon }) => (
        <Link key={href} href={withCompetition(href, competition)} aria-current={isActive(path, href) ? "page" : undefined} className="cv-item">
          <Icon size={20} aria-hidden="true" />
          <span>{label}</span>
        </Link>
      ))}
    </nav>
  );
}

export function BottomNav({ competition }: { competition: string }) {
  const path = usePathname();
  return (
    <nav aria-label="Mobile navigation" className="cv-pill">
      {NAV.map(({ href, label, icon: Icon }) => (
        <Link key={href} href={withCompetition(href, competition)} aria-current={isActive(path, href) ? "page" : undefined} className="cv-pill-item">
          <Icon size={19} aria-hidden="true" />
          <span>{label}</span>
        </Link>
      ))}
    </nav>
  );
}

/** Sticky header: brand, competition pills (they re-scope the current page), and the way back to NBA. */
export function Header({ leagues, competition }: { leagues: League[]; competition: string }) {
  const path = usePathname();
  const target = rootOf(path);
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href={withCompetition("/games", competition)} aria-label="Soccer Props AI home" className="site-brand">
          <span className="cv-ring" aria-hidden />
          <span className="site-brand-text">Soccer Props AI</span>
        </Link>
        <div className="site-league-toggle" role="group" aria-label="Competition">
          {leagues.map((l) => (
            <Link
              key={l.code}
              href={withCompetition(target, l.code)}
              aria-current={l.code === competition ? "true" : undefined}
              className={`site-league-pill${l.code === competition ? " active" : ""}`}
            >
              {SHORT[l.code] ?? l.name.toUpperCase()}
            </Link>
          ))}
        </div>
        {/* eslint-disable-next-line @next/next/no-html-link-for-pages -- main site is a separate deployment */}
        <a href="/" className="site-back" aria-label="Back to Court Vision (NBA, WNBA, College)">NBA</a>
      </div>
    </header>
  );
}
