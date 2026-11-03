import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

/** Surnames read better in dense tables than full registered names. */
export function shortName(full: string | null | undefined): string {
  if (!full) return "—";
  const parts = full.trim().split(/\s+/);
  if (parts.length === 1) return parts[0];
  return `${parts[0][0]}. ${parts[parts.length - 1]}`;
}

export function clubName(full: string | null | undefined): string {
  return full ?? "—";
}

export const fmtDate = (iso: string) =>
  new Date(iso + "T00:00:00").toLocaleDateString("en-GB", {
    day: "numeric", month: "short", year: "numeric",
  });

/** "7 Sep" — for dense rows where the year is already established. */
export const shortDate = (iso: string) =>
  new Date(iso + "T00:00:00").toLocaleDateString("en-GB", {
    day: "numeric", month: "short",
  });

export const fmtClock = (minute: number, second = 0) =>
  `${minute}:${String(second).padStart(2, "0")}`;

export const pct = (v: number) => `${v.toFixed(0)}%`;

export const signed = (v: number) => (v > 0 ? `+${v}` : `${v}`);

export const SERIES = [
  "var(--color-series-1)", "var(--color-series-2)", "var(--color-series-3)",
  "var(--color-series-4)", "var(--color-series-5)", "var(--color-series-6)",
] as const;

/** Pitch geometry shared by 2D SVG panels and the 3D scenes. */
export const PITCH = { length: 120, width: 80 } as const;

/**
 * Display form of a club name.
 *
 * openfootball writes Champions League clubs in full legal form —
 * "FC Bayern München", "Paris Saint-Germain FC", "Club Atlético de Madrid" —
 * which truncates to uselessness in a narrow column. Stripping the corporate
 * affixes keeps the identifying words. The underlying canonical name is never
 * changed; this only affects what is rendered.
 */
const CLUB_AFFIX = /^(FC|AFC|CF|SC|AC|AS|SS|SK|BC|KV|FK|PAE|OGC|AJ|RCD|CA|UD|SD|RC)\s+|\s+(FC|AFC|CF|SC|AC|SK|BC|KV|FK|SFP|CF)$/gi;

export function displayTeam(name: string | null | undefined): string {
  if (!name) return "—";
  let out = name.trim();
  for (let i = 0; i < 2; i++) out = out.replace(CLUB_AFFIX, " ").trim();
  out = out.replace(/^Club\s+/i, "").replace(/\sde\s/i, " ");
  out = out.replace(/\s{2,}/g, " ").trim();
  return out || name;
}

/**
 * Kick-off times in the viewer's own zone, in 12-hour form.
 *
 * The lake stores UTC, which is right for storage and wrong for reading: a
 * 19:00 Champions League kick-off is 3:00 PM in Albany, and showing "19:00"
 * means every reader does the arithmetic themselves.
 *
 * The zone is a constant rather than the browser's, deliberately. These pages
 * are server-rendered, so reading `Intl`'s runtime zone would format one way on
 * the server and another in the browser and React would report a hydration
 * mismatch on every fixture row. A fixed zone renders identically in both.
 */
export const DISPLAY_TIME_ZONE = "America/New_York";

/**
 * "3:00 PM", or "TBC" when the source published no kick-off time.
 *
 * Some season schedules carry a date only. Those become midnight UTC in the
 * lake, which formats as 8:00 PM the previous evening in Albany — a specific,
 * confident, invented time. A missing time is shown as missing instead.
 */
export function kickoffTime(
  kickoffUtc: string | null | undefined,
  date?: string | null,
  time?: string | null,
): string {
  if (!hasRealTime(kickoffUtc, time)) return "TBC";
  const at = toDate(kickoffUtc, date, time);
  if (!at) return "TBC";
  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: DISPLAY_TIME_ZONE,
  }).format(at);
}

/** "Tuesday, October 13" in the display zone. */
export function kickoffDate(
  kickoffUtc: string | null | undefined,
  date?: string | null,
  time?: string | null,
): string {
  const at = toDate(kickoffUtc, date, time);
  if (!at) return "Date to be confirmed";
  return new Intl.DateTimeFormat("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    timeZone: DISPLAY_TIME_ZONE,
  }).format(at);
}

/** The calendar day in the display zone, as YYYY-MM-DD, for grouping. */
export function localDayKey(
  kickoffUtc: string | null | undefined,
  date?: string | null,
  time?: string | null,
): string {
  const at = toDate(kickoffUtc, date, time);
  if (!at) return date ?? "";
  // en-CA yields ISO order, which sorts correctly as a string.
  return new Intl.DateTimeFormat("en-CA", {
    year: "numeric", month: "2-digit", day: "2-digit",
    timeZone: DISPLAY_TIME_ZONE,
  }).format(at);
}

/** Whether a kick-off time was actually published, rather than defaulted. */
function hasRealTime(
  kickoffUtc: string | null | undefined,
  time?: string | null,
): boolean {
  if (time) return time.slice(0, 5) !== "00:00";
  if (!kickoffUtc) return false;
  // A date-only source row becomes exactly midnight UTC; no European fixture
  // kicks off then, so it is the marker for "time unknown".
  return !/T00:00(:00)?/.test(kickoffUtc);
}

function toDate(
  kickoffUtc: string | null | undefined,
  date?: string | null,
  time?: string | null,
): Date | null {
  if (kickoffUtc) {
    const d = new Date(kickoffUtc);
    if (!Number.isNaN(d.getTime())) return d;
  }
  if (!date) return null;
  // Source times carry no zone and are UTC by convention in this project.
  const d = new Date(`${date.slice(0, 10)}T${(time ?? "00:00").slice(0, 5)}:00Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}
