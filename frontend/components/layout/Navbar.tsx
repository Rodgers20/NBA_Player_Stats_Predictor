"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { usePrefs } from "@/store/prefs";

export const NAV = [
  { href: "/analysis", label: "Player Analysis" },
  { href: "/games", label: "Today's Games" },
  { href: "/props", label: "Best Props" },
  { href: "/bets", label: "My Bets" },
];
export const navForLeague = (league: "nba" | "wnba") => league === "wnba"
  ? [...NAV.slice(0, 3), { href: "/hitrates", label: "Hit Rates" }, NAV[3]]
  : NAV;

export function Navbar() {
  const path = usePathname();
  const router = useRouter();
  const { league, setLeague } = usePrefs();

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href="/analysis" aria-label={`${league.toUpperCase()} Props AI home`} className="site-brand">
          {league.toUpperCase()} Props AI
        </Link>
        <nav aria-label="Main navigation" className="site-nav">
          {navForLeague(league).map(({ href, label }) => {
            const active = path === href || (href === "/props" && path === "/");
            return (
              <Link key={href} href={href} aria-current={active ? "page" : undefined} className={`site-nav-link${active ? " active" : ""}`}>
                {label}
              </Link>
            );
          })}
        </nav>
        <div className="site-league-toggle" role="group" aria-label="League">
          {(["nba", "wnba"] as const).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => {
                setLeague(value);
                if (path === "/analysis") router.push(`/analysis?league=${value}`);
              }}
              aria-pressed={league === value}
              className={`site-league-pill${league === value ? " active" : ""}`}
            >
              {value.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
