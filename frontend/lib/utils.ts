import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const STAT_COLORS: Record<string, string> = {
  PTS:  "#14b8a6",
  AST:  "#f97066",
  REB:  "#a78bfa",
  STL:  "#fbbf24",
  BLK:  "#60a5fa",
  FG3M: "#ec4899",
};

export function statColor(stat: string): string {
  const base = stat.split("+")[0].trim().toUpperCase();
  return STAT_COLORS[base] ?? "#2dd4bf";
}

export function fmtOdds(ml: number | null): string {
  if (ml === null || ml === undefined) return "—";
  return ml > 0 ? `+${ml}` : String(ml);
}
