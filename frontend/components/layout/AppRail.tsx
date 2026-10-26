"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV } from "./Navbar";
import { isActivePath, navIcons } from "./nav-icons";
import styles from "./shell.module.css";

/** Desktop icon rail (hidden below 901px, where the floating pill nav takes over). */
export function AppRail() {
  const path = usePathname();
  return (
    <nav aria-label="Main navigation" className={styles.rail}>
      <Link href="/" aria-label="Home" className={styles.mark} />
      {NAV.map(({ href, label }) => {
        const Icon = navIcons[href];
        return (
          <Link key={href} href={href} aria-current={isActivePath(path, href) ? "page" : undefined} className={styles.item}>
            <Icon size={20} aria-hidden="true" />
            <span className={styles.label}>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
