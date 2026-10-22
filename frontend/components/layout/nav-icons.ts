import { Target, ChartNoAxesCombined, CalendarDays, Sparkles, Bookmark, ChartColumnIncreasing, Shield, Clapperboard, ChartLine, Flame, TrendingUp } from "lucide-react";
import type { NAV } from "./Navbar";

export const navIcons = {
  "/": Target, "/analysis": ChartNoAxesCombined, "/games": CalendarDays, "/props": Sparkles, "/popular": TrendingUp, "/trends": Flame,
  "/hitrates": ChartColumnIncreasing, "/bets": Bookmark, "/teams": Shield, "/highlights": Clapperboard, "/performance": ChartLine,
} satisfies Record<(typeof NAV)[number]["href"] | "/hitrates", typeof CalendarDays>;

export const isActivePath = (path: string, href: string) => (href === "/" ? path === "/" : path === href || path.startsWith(`${href}/`));
