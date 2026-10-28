"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV } from "./Navbar";
import { isActivePath, navIcons } from "./nav-icons";
import styles from "./shell.module.css";

/** Floating pill nav for phones: the five pages, all visible, no "More" sheet to dig through. */
export function BottomNav() {
  const path = usePathname();
  return (
    <nav aria-label="Mobile navigation" className={styles.pill}>
      {NAV.map(({ href, label }) => {
        const Icon = navIcons[href];
        return (
          <Link key={href} href={href} aria-current={isActivePath(path, href) ? "page" : undefined} className={styles.pillItem}>
            <Icon size={19} aria-hidden="true" />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
