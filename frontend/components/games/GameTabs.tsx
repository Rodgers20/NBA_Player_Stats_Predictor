"use client";

import type { KeyboardEvent } from "react";
import styles from "./GameDetail.module.css";

export type TabDef<T extends string> = { id: T; label: string };

/** Accessible tab list. Arrow keys move selection and focus instantly (no animation on keyboard use). */
export function Tabs<T extends string>({ tabs, value, onChange, label, className }: { tabs: TabDef<T>[]; value: T; onChange: (id: T) => void; label: string; className?: string }) {
  const move = (event: KeyboardEvent<HTMLDivElement>) => {
    const delta = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!delta) return;
    const index = (tabs.findIndex(tab => tab.id === value) + delta + tabs.length) % tabs.length;
    onChange(tabs[index].id);
    event.currentTarget.querySelectorAll<HTMLElement>('[role="tab"]')[index]?.focus();
  };
  return (
    <div role="tablist" aria-label={label} className={`${styles.tabs}${className ? ` ${className}` : ""}`} onKeyDown={move}>
      {tabs.map(tab => (
        <button key={tab.id} type="button" role="tab" aria-selected={tab.id === value} tabIndex={tab.id === value ? 0 : -1} className={styles.tab} onClick={() => onChange(tab.id)}>
          {tab.label}
        </button>
      ))}
    </div>
  );
}
