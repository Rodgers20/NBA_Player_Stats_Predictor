import { Target, ChartNoAxesCombined, CalendarDays, Sparkles, Bookmark } from "lucide-react";
import type { NAV } from "./Navbar";

export const navIcons = {
  "/": Target, "/props": Sparkles, "/games": CalendarDays, "/analysis": ChartNoAxesCombined, "/bets": Bookmark,
} satisfies Record<(typeof NAV)[number]["href"], typeof CalendarDays>;

export const isActivePath = (path: string, href: string) => (href === "/" ? path === "/" : path === href || path.startsWith(`${href}/`));
