"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { TeamLogo } from "@/components/teams/TeamLogo";
import type { CbbTeam } from "@/lib/cbb-types";

const TABS = [
  { href: "/cbb", label: "Games" },
  { href: "/cbb/rankings", label: "Power rankings" },
  { href: "/cbb/standings", label: "Standings" },
  { href: "/cbb/players", label: "Players" },
  { href: "/cbb/draft", label: "Draft" },
  { href: "/cbb/model", label: "Model record" },
] as const;

/** Page title plus the college sub-nav. */
export function CbbHeader({ title, sub, children }: { title: string; sub?: ReactNode; children?: ReactNode }) {
  const path = usePathname().replace(/\/$/, "") || "/";
  return (
    <header className="mb-5 space-y-4">
      <nav aria-label="College basketball" className="flex gap-2 overflow-x-auto pb-1 [scrollbar-width:none]">
        {TABS.map(tab => {
          const active = tab.href === "/cbb" ? path === "/cbb" || path.startsWith("/cbb/game") : path.startsWith(tab.href);
          return <Link key={tab.href} href={tab.href} aria-current={active ? "page" : undefined}
            className={`ql-chip shrink-0 whitespace-nowrap ${active ? "ql-chip-active" : ""}`}>{tab.label}</Link>;
        })}
      </nav>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="ql-kicker">Men&apos;s college basketball</p>
          <h1 className="ql-heading mt-1">{title}</h1>
          {sub && <p className="ql-subtitle mt-2 max-w-2xl">{sub}</p>}
        </div>
        {children}
      </div>
    </header>
  );
}

export function CbbLogo({ team, size = 32 }: { team: Pick<CbbTeam, "abbr" | "name" | "logo">; size?: number }) {
  return <TeamLogo team={{ abbrev: team.abbr, name: team.name, logo: team.logo }} size={size} />;
}

export function RankTag({ rank }: { rank: number | null | undefined }) {
  return rank == null ? null : <span className="mr-1 font-mono text-[10px] font-bold text-lock-gold" title={`AP Top 25: No. ${rank}`}>{rank}</span>;
}

const MARKS = {
  hit: { glyph: "✓", cls: "text-teal-300 border-teal-400/50 bg-teal-400/10", label: "Correct" },
  miss: { glyph: "✗", cls: "text-rose-300 border-rose-400/50 bg-rose-400/10", label: "Missed" },
  push: { glyph: "–", cls: "text-text-sec border-border-card bg-bg-root", label: "Push" },
} as const;

/** Hit / miss / push badge for a graded call. Renders nothing for an ungraded one. */
export function Mark({ kind }: { kind: "hit" | "miss" | "push" | null | undefined }) {
  if (!kind) return null;
  const m = MARKS[kind];
  return <span role="img" aria-label={m.label} title={m.label} className={`inline-flex h-4 min-w-4 items-center justify-center rounded border px-1 text-[10px] font-bold leading-none ${m.cls}`}>{m.glyph}</span>;
}

export function Loading({ label }: { label: string }) {
  return <div className="ql-panel h-64 animate-pulse" aria-busy="true" aria-label={label} />;
}

export function ErrorBox({ onRetry, children }: { onRetry?: () => void; children: ReactNode }) {
  return <div role="alert" className="ql-panel p-5 text-sm text-text-sec">{children} {onRetry && <button type="button" onClick={onRetry} className="underline">Try again</button>}</div>;
}
