"use client";

import Link from "next/link";
import { useSectionNav } from "./section-nav";
import styles from "./shell.module.css";

/** Floating pill nav for phones: the five pages, all visible, no "More" sheet to dig through. */
export function BottomNav() {
  const { items, isActive } = useSectionNav();
  return (
    <nav aria-label="Mobile navigation" className={styles.pill}>
      {items.map(({ href, label, Icon }) => {
        return (
          <Link key={href} href={href} aria-current={isActive(href) ? "page" : undefined} className={styles.pillItem}>
            <Icon size={19} aria-hidden="true" />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
