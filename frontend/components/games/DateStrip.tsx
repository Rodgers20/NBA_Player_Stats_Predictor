"use client";

import styles from "./DateStrip.module.css";

const label = (day: string, today: string) =>
  day === today ? "Today" : new Date(`${day}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" });
const short = (day: string) => new Date(`${day}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

/** Horizontal day picker. Arrow keys step the selected day; no animation on keyboard use. */
export function DateStrip({ dates, date, today, onChange }: { dates: string[]; date?: string; today: string; onChange: (isoDate: string) => void }) {
  const step = (delta: number) => {
    const next = dates[dates.indexOf(date ?? today) + delta];
    if (next) onChange(next);
  };
  return (
    <nav
      aria-label="Game dates"
      className={styles.strip}
      onKeyDown={event => {
        if (event.key === "ArrowRight") step(1);
        if (event.key === "ArrowLeft") step(-1);
      }}
    >
      {dates.map(day => (
        <button key={day} type="button" aria-pressed={day === date} className={styles.day} onClick={() => onChange(day)}>
          <span>{label(day, today)}</span>
          <strong>{short(day)}</strong>
        </button>
      ))}
    </nav>
  );
}
