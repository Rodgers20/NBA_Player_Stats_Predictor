"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { usePrefs } from "@/store/prefs";

// Five jobs, five places. Popular, Trends and Hit Rates are views of Props; Teams and Highlights
// are reached from Games; the model record is linked from Tonight and My Bets.
export const NAV = [
  { href: "/", label: "Tonight" },
  { href: "/props", label: "Props" },
  { href: "/games", label: "Games" },
  { href: "/analysis", label: "Players" },
  { href: "/bets", label: "My Bets" },
] as const;

export function Navbar() {
  const path = usePathname();
  const router = useRouter();
  const { league, setLeague } = usePrefs();
  const college = path === "/cbb" || path.startsWith("/cbb/");

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href="/" aria-label={college ? "College basketball home" : `${league.toUpperCase()} Props AI home`} className="site-brand">
          {college ? "College Hoops" : `${league.toUpperCase()} Props AI`}
        </Link>
        <div className="site-league-toggle" role="group" aria-label="League">
          {(["nba", "wnba"] as const).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => {
                setLeague(value);
                if (college) router.push("/");
                if (path === "/analysis") router.push(`/analysis?league=${value}`);
                if (path.startsWith("/teams")) router.push(`/teams?league=${value}`);
              }}
              aria-pressed={!college && league === value}
              className={`site-league-pill${!college && league === value ? " active" : ""}`}
            >
              {value.toUpperCase()}
            </button>
          ))}
          <Link href="/cbb" aria-current={college ? "page" : undefined} className={`site-league-pill${college ? " active" : ""}`}>
            COLLEGE
          </Link>
        </div>
      </div>
    </header>
  );
}
