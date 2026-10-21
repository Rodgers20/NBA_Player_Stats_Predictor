"use client";

import type { KeyboardEvent } from "react";
import styles from "./GameDetail.module.css";

export type TabDef<T extends string> = { id: T; label: string };

/** Accessible tab list. Arrow keys move selection and focus instantly (no animation on keyboard use). */
export function Tabs<T extends string>({ tabs, value, onChange, label, className, idPrefix }: { tabs: TabDef<T>[]; value: T; onChange: (id: T) => void; label: string; className?: string; idPrefix: string }) {
  const move = (event: KeyboardEvent<HTMLDivElement>) => {
    const current = tabs.findIndex(tab => tab.id === value);
    const index = event.key === "Home" ? 0
      : event.key === "End" ? tabs.length - 1
      : event.key === "ArrowRight" ? (current + 1) % tabs.length
      : event.key === "ArrowLeft" ? (current - 1 + tabs.length) % tabs.length
      : -1;
    if (index < 0) return;
    event.preventDefault();
    onChange(tabs[index].id);
    event.currentTarget.querySelectorAll<HTMLElement>('[role="tab"]')[index]?.focus();
  };
  return (
    <div role="tablist" aria-label={label} className={`${styles.tabs}${className ? ` ${className}` : ""}`} onKeyDown={move}>
      {tabs.map(tab => (
        <button key={tab.id} type="button" role="tab" id={`${idPrefix}-tab-${tab.id}`} aria-controls={`${idPrefix}-panel`} aria-selected={tab.id === value} tabIndex={tab.id === value ? 0 : -1} className={styles.tab} onClick={() => onChange(tab.id)}>
          {tab.label}
        </button>
      ))}
    </div>
  );
}
