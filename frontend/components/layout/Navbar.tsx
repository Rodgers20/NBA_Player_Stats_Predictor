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

type Section = "nba" | "wnba" | "nhl" | "college";

// Logos are self-hosted in /public/leagues. Dimensions are the rendered box, not the file size.
const LOGO = {
  nba: { src: "/leagues/nba.svg", w: 9, h: 20 },
  wnba: { src: "/leagues/wnba.png", w: 20, h: 20 },
  nhl: { src: "/leagues/nhl.svg", w: 24, h: 18 },
  college: { src: "/leagues/ncaa.svg", w: 20, h: 20 },
  soccer: { src: "/leagues/soccer.svg", w: 20, h: 20 },
} as const;

function LeagueLogo({ k }: { k: keyof typeof LOGO }) {
  const { src, w, h } = LOGO[k];
  // eslint-disable-next-line @next/next/no-img-element -- tiny static assets, label text names the league
  return <img src={src} width={w} height={h} alt="" className="site-league-logo" />;
}

export function Navbar() {
  const path = usePathname();
  const router = useRouter();
  const { league, setLeague } = usePrefs();
  const inSection = (p: string) => path === p || path.startsWith(`${p}/`);
  // /nhl and /cbb are their own sections; everywhere else the stored NBA/WNBA league is current.
  const section: Section = inSection("/nhl") ? "nhl" : inSection("/cbb") ? "college" : league;
  const offSite = section === "nhl" || section === "college";
  const brand = section === "nhl" ? "NHL Props AI" : section === "college" ? "College Hoops" : `${league.toUpperCase()} Props AI`;
  const brandLabel = section === "nhl" ? "NHL Props AI home" : section === "college" ? "College basketball home" : `${league.toUpperCase()} Props AI home`;
  const pill = (active: boolean) => `site-league-pill${active ? " active" : ""}`;

  const nbaLike = (value: "nba" | "wnba") => (
    <button
      key={value}
      type="button"
      onClick={() => {
        setLeague(value);
        if (offSite) router.push("/");
        if (path === "/analysis") router.push(`/analysis?league=${value}`);
        if (path.startsWith("/teams")) router.push(`/teams?league=${value}`);
      }}
      aria-pressed={section === value}
      className={pill(section === value)}
    >
      <LeagueLogo k={value} />
      <span className="site-league-label">{value.toUpperCase()}</span>
    </button>
  );

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href="/" aria-label={brandLabel} className="site-brand">
          {brand}
        </Link>
        <div className="site-league-toggle" role="group" aria-label="League">
          {nbaLike("nba")}
          {nbaLike("wnba")}
          <Link href="/nhl" aria-current={section === "nhl" ? "page" : undefined} className={pill(section === "nhl")}>
            <LeagueLogo k="nhl" />
            <span className="site-league-label">NHL</span>
          </Link>
          <Link href="/cbb" aria-current={section === "college" ? "page" : undefined} className={pill(section === "college")}>
            <LeagueLogo k="college" />
            <span className="site-league-label">COLLEGE</span>
          </Link>
          {/* Soccer (FUTBALL) is its own deployment proxied at /soccer, so a plain anchor, not a Next <Link>. */}
          <a href="/soccer" className="site-league-pill">
            <LeagueLogo k="soccer" />
            <span className="site-league-label">SOCCER</span>
          </a>
        </div>
      </div>
    </header>
  );
}
