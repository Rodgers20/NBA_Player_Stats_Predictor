import type { ReactNode } from "react";
import styles from "./shell.module.css";

/** Sticky right column on wide screens; stacks under the page on narrow ones so nothing is hidden. */
export function ContextSidebar({ children, label = "Context" }: { children: ReactNode; label?: string }) {
  return <aside aria-label={label} className={styles.side}>{children}</aside>;
}
