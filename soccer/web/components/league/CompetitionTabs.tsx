"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { cn } from "@/lib/format";

/**
 * Competition page tabs.
 *
 * Each tab is a real link carrying the current competition and season in the
 * query string, so the browser Back button, a refresh and a shared URL all land
 * on the same view. A click-only tab strip would lose all three.
 *
 * A tab is only listed when something can fill it. News reads public RSS from
 * BBC Sport and the Guardian; there is no tab for anything that would be empty.
 */
export const COMPETITION_TABS = [
  { id: "overview", label: "Overview" },
  { id: "table", label: "Table" },
  { id: "fixtures", label: "Fixtures" },
  { id: "players", label: "Player stats" },
  { id: "teams", label: "Team stats" },
  { id: "seasons", label: "Seasons" },
  { id: "news", label: "News" },
] as const;

export type CompetitionTab = (typeof COMPETITION_TABS)[number]["id"];

export function CompetitionTabs({ active }: { active: CompetitionTab }) {
  const pathname = usePathname();
  const params = useSearchParams();

  const href = (tab: string) => {
    const next = new URLSearchParams(params.toString());
    if (tab === "overview") next.delete("tab");
    else next.set("tab", tab);
    const qs = next.toString();
    return qs ? `${pathname}?${qs}` : pathname;
  };

  return (
    <nav aria-label="Competition sections"
         className="-mb-px flex gap-1 overflow-x-auto border-b border-stroke">
      {COMPETITION_TABS.map((t) => {
        const current = t.id === active;
        return (
          <Link
            key={t.id}
            href={href(t.id)}
            aria-current={current ? "page" : undefined}
            className={cn(
              "whitespace-nowrap border-b-2 px-3 py-2.5 text-[13px] font-medium transition-colors",
              current
                ? "border-pitch text-ink"
                : "border-transparent text-ink-muted hover:text-ink",
            )}
          >
            {t.label}
          </Link>
        );
      })}
    </nav>
  );
}
