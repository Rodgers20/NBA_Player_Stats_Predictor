"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Ellipsis } from "lucide-react";
import { usePrefs } from "@/store/prefs";
import { navForLeague } from "./Navbar";
import { isActivePath, navIcons } from "./nav-icons";
import styles from "./shell.module.css";

const PRIMARY: Record<string, string> = { "/": "Tonight", "/games": "Games", "/props": "Props", "/trends": "Trends", "/bets": "My Bets" };

/** Floating pill nav for phones: five core pages, everything else under a "More" sheet. */
export function BottomNav() {
  const path = usePathname();
  const league = usePrefs(state => state.league);
  const [open, setOpen] = useState(false);
  const items = navForLeague(league);
  const primary = items.filter(({ href }) => href in PRIMARY);
  const more = items.filter(({ href }) => !(href in PRIMARY));

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <nav aria-label="Mobile navigation" className={styles.pill}>
        {primary.map(({ href }) => {
          const Icon = navIcons[href as keyof typeof navIcons];
          return (
            <Link key={href} href={href} aria-current={isActivePath(path, href) ? "page" : undefined} className={styles.pillItem}>
              <Icon size={19} aria-hidden="true" />
              <span>{PRIMARY[href]}</span>
            </Link>
          );
        })}
        <button type="button" aria-expanded={open} aria-controls="more-sheet" onClick={() => setOpen(value => !value)} className={styles.pillItem}>
          <Ellipsis size={19} aria-hidden="true" />
          <span>More</span>
        </button>
      </nav>
      <div className={styles.scrim} data-open={open} onClick={() => setOpen(false)} aria-hidden="true" />
      <div id="more-sheet" role="dialog" aria-label="More pages" className={styles.sheet} data-open={open} inert={!open}>
        {more.map(({ href, label }) => {
          const Icon = navIcons[href as keyof typeof navIcons];
          return (
            <Link key={href} href={href} onClick={() => setOpen(false)} aria-current={isActivePath(path, href) ? "page" : undefined} className={styles.sheetLink}>
              <Icon size={20} aria-hidden="true" />
              <span>{label}</span>
            </Link>
          );
        })}
      </div>
    </>
  );
}
