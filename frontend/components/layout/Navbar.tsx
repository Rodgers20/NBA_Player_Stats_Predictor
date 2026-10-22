"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { usePrefs } from "@/store/prefs";

export const NAV = [
  { href: "/", label: "Tonight" },
  { href: "/analysis", label: "Player Analysis" },
  { href: "/games", label: "Today's Games" },
  { href: "/props", label: "Best Props" },
  { href: "/popular", label: "Popular" },
  { href: "/trends", label: "Trends" },
  { href: "/teams", label: "Teams" },
  { href: "/highlights", label: "Highlights" },
  { href: "/performance", label: "Model Performance" },
  { href: "/bets", label: "My Bets" },
] as const;
export const navForLeague = (league: "nba" | "wnba") => league === "wnba"
  ? [...NAV.slice(0, -1), { href: "/hitrates", label: "Hit Rates" }, NAV[NAV.length - 1]]
  : NAV;

export function Navbar() {
  const path = usePathname();
  const router = useRouter();
  const { league, setLeague } = usePrefs();

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href="/" aria-label={`${league.toUpperCase()} Props AI home`} className="site-brand">
          {league.toUpperCase()} Props AI
        </Link>
        <div className="site-league-toggle" role="group" aria-label="League">
          {(["nba", "wnba"] as const).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => {
                setLeague(value);
                if (path === "/analysis") router.push(`/analysis?league=${value}`);
                if (path.startsWith("/teams")) router.push(`/teams?league=${value}`);
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
