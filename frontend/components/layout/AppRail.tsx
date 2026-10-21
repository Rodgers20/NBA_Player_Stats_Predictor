"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { usePrefs } from "@/store/prefs";
import { navForLeague } from "./Navbar";
import { isActivePath, navIcons } from "./nav-icons";
import styles from "./shell.module.css";

/** Desktop icon rail (hidden below 901px, where the floating pill nav takes over). */
export function AppRail() {
  const path = usePathname();
  const league = usePrefs(state => state.league);
  return (
    <nav aria-label="Main navigation" className={styles.rail}>
      <Link href="/" aria-label="Home" className={styles.mark} />
      {navForLeague(league).map(({ href, label }) => {
        const Icon = navIcons[href as keyof typeof navIcons];
        return (
          <Link key={href} href={href} aria-label={label} aria-current={isActivePath(path, href) ? "page" : undefined} className={styles.item}>
            <Icon size={21} aria-hidden="true" />
            <span className={styles.tip} aria-hidden="true">{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
