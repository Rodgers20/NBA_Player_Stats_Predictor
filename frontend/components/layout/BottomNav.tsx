"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChartNoAxesCombined, CalendarDays, Sparkles, Bookmark, ChartColumnIncreasing } from "lucide-react";
import { navForLeague } from "./Navbar";
import { usePrefs } from "@/store/prefs";

const icons = { "/analysis": ChartNoAxesCombined, "/games": CalendarDays, "/props": Sparkles, "/hitrates": ChartColumnIncreasing, "/bets": Bookmark };

export function BottomNav() {
  const path = usePathname();
  const league = usePrefs(state => state.league);

  return (
    <nav aria-label="Mobile navigation" className="mobile-nav safe-area-pb">
      {navForLeague(league).map(({ href, label }) => {
        const Icon = icons[href as keyof typeof icons];
        const active = path === href || (href === "/props" && path === "/");
        return (
          <Link key={href} href={href} aria-current={active ? "page" : undefined} className={`mobile-nav-link${active ? " active" : ""}`}>
            <Icon size={19} aria-hidden="true" />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
