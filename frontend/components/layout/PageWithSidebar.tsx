import type { ReactNode } from "react";
import { ContextSidebar } from "./ContextSidebar";
import styles from "./shell.module.css";

export function PageWithSidebar({ main, side, sideLabel }: { main: ReactNode; side?: ReactNode; sideLabel?: string }) {
  return (
    <div className={side ? styles.layout : undefined}>
      <div className={styles.main}>{main}</div>
      {side ? <ContextSidebar label={sideLabel}>{side}</ContextSidebar> : null}
    </div>
  );
}
