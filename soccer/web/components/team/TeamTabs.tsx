"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { cn } from "@/lib/format";

/**
 * Club page tabs, as real links so Back, refresh and sharing all work.
 *
 * Transfers is absent: no free source publishes fees or dealings, and a tab
 * that can only ever be empty is worse than one that is not there.
 */
export const TEAM_TABS = [
  { id: "overview", label: "Overview" },
  { id: "table", label: "Table" },
  { id: "fixtures", label: "Fixtures" },
  { id: "squad", label: "Squad" },
  { id: "stats", label: "Stats" },
  { id: "history", label: "History" },
  { id: "news", label: "News" },
] as const;

export type TeamTab = (typeof TEAM_TABS)[number]["id"];

export function TeamTabs({ active }: { active: TeamTab }) {
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
    <nav aria-label="Club sections"
         className="-mb-px flex gap-1 overflow-x-auto border-b border-stroke">
      {TEAM_TABS.map((t) => {
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
