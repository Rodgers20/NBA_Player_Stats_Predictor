"use client";

import Link from "next/link";
import { useSectionNav } from "./section-nav";
import styles from "./shell.module.css";

/** Desktop icon rail (hidden below 901px, where the floating pill nav takes over). */
export function AppRail() {
  const { items, isActive } = useSectionNav();
  return (
    <nav aria-label="Main navigation" className={styles.rail}>
      <Link href="/" aria-label="Home" className={styles.mark} />
      {items.map(({ href, label, Icon }) => {
        return (
          <Link key={href} href={href} aria-current={isActive(href) ? "page" : undefined} className={styles.item}>
            <Icon size={20} aria-hidden="true" />
            <span className={styles.label}>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
