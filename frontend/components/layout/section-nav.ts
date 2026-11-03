"use client";

import { usePathname } from "next/navigation";
import { Target, ChartNoAxesCombined, CalendarDays, Sparkles, Bookmark, Trophy, ListOrdered, type LucideIcon } from "lucide-react";
import { usePrefs } from "@/store/prefs";
import { NAV } from "./Navbar";

export type NavItem = { href: string; label: string; Icon: LucideIcon };
export type Section = "nba" | "wnba" | "nhl" | "college";

const BETS: NavItem = { href: "/bets", label: "My Bets", Icon: Bookmark };

const NBA_LIKE: NavItem[] = NAV.map(({ href, label }) => ({
  href, label,
  Icon: { "/": Target, "/props": Sparkles, "/games": CalendarDays, "/analysis": ChartNoAxesCombined, "/bets": Bookmark }[href],
}));

// Off-site sections keep the bar inside their own routes; My Bets is the one shared page.
const NHL: NavItem[] = [
  { href: "/nhl", label: "Tonight", Icon: Target },
  { href: "/nhl/props", label: "Props", Icon: Sparkles },
  { href: "/nhl/games", label: "Games", Icon: CalendarDays },
  { href: "/nhl/players", label: "Players", Icon: ChartNoAxesCombined },
  BETS,
];
const COLLEGE: NavItem[] = [
  { href: "/cbb", label: "Games", Icon: CalendarDays },
  { href: "/cbb/rankings", label: "Rankings", Icon: Trophy },
  { href: "/cbb/players", label: "Players", Icon: ChartNoAxesCombined },
  { href: "/cbb/standings", label: "Standings", Icon: ListOrdered },
  BETS,
];

const inSection = (path: string, p: string) => path === p || path.startsWith(`${p}/`);

export function useSection(): Section {
  const path = usePathname();
  const league = usePrefs(s => s.league);
  return inSection(path, "/nhl") ? "nhl" : inSection(path, "/cbb") ? "college" : league;
}

export function useSectionNav(): { items: NavItem[]; isActive: (href: string) => boolean } {
  const path = usePathname().replace(/\/$/, "") || "/";
  const section = useSection();
  const items = section === "nhl" ? NHL : section === "college" ? COLLEGE : NBA_LIKE;
  // A section's home also owns its game pages; the NBA root is exact-match only.
  const isActive = (href: string) => {
    if (href === "/nhl/games") return path.startsWith("/nhl/game") || path.startsWith("/nhl/standings") || path.startsWith("/nhl/model");
    if (href === "/cbb") return path === href || path.startsWith("/cbb/game");
    return href === "/" || href === "/nhl" ? path === href : inSection(path, href);
  };
  return { items, isActive };
}
