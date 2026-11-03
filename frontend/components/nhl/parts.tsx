"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import { TeamLogo } from "@/components/teams/TeamLogo";
import type { NhlTeamRef } from "@/lib/nhl-types";

export { ErrorBox, Loading, Mark } from "@/components/cbb/parts";

const TABS = [
  { href: "/nhl", label: "Games" },
  { href: "/nhl/props", label: "Props" },
  { href: "/nhl/players", label: "Players" },
  { href: "/nhl/standings", label: "Standings" },
  { href: "/nhl/model", label: "Model record" },
] as const;

/** Page title plus the NHL sub-nav. */
export function NhlHeader({ title, sub, children }: { title: string; sub?: ReactNode; children?: ReactNode }) {
  const path = usePathname().replace(/\/$/, "") || "/";
  return (
    <header className="mb-5 space-y-4">
      <nav aria-label="NHL" className="flex gap-2 overflow-x-auto pb-1 [scrollbar-width:none]">
        {TABS.map(tab => {
          const active = tab.href === "/nhl" ? path === "/nhl" || path.startsWith("/nhl/game") : path.startsWith(tab.href);
          return <Link key={tab.href} href={tab.href} aria-current={active ? "page" : undefined}
            className={`ql-chip shrink-0 whitespace-nowrap ${active ? "ql-chip-active" : ""}`}>{tab.label}</Link>;
        })}
      </nav>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="ql-kicker">National Hockey League</p>
          <h1 className="ql-heading mt-1">{title}</h1>
          {sub && <p className="ql-subtitle mt-2 max-w-2xl">{sub}</p>}
        </div>
        {children}
      </div>
    </header>
  );
}

/** Team logo from the data's own SVG url (the dark-background variant), initials when it fails. */
export function NhlLogo({ team, size = 32 }: { team: Pick<NhlTeamRef, "abbr" | "name" | "logo" | "logo_dark">; size?: number }) {
  return <TeamLogo team={{ abbrev: team.abbr, name: team.name, logo: team.logo_dark }} size={size} />;
}

/** Same logo from just an abbreviation, for rows that only carry the team code. */
export function AbbrLogo({ abbr, size = 20 }: { abbr: string; size?: number }) {
  return <TeamLogo team={{ abbrev: abbr, name: abbr, logo: `https://assets.nhle.com/logos/nhl/svg/${abbr}_dark.svg` }} size={size} />;
}

const initials = (name: string) => name.split(/\s+/).filter(Boolean).slice(0, 2).map(w => w[0]).join("").toUpperCase();

/** NHL mug; initials when the photo is missing. */
export function Headshot({ src, name, size = 40 }: { src: string | null | undefined; name: string; size?: number }) {
  const [failed, setFailed] = useState(false);
  const box = { width: size, height: size };
  if (!src || failed) {
    return <span aria-hidden="true" style={{ ...box, fontSize: Math.max(10, size * 0.34) }}
      className="inline-flex shrink-0 items-center justify-center rounded-full border border-border-card bg-surface-sunk font-display font-bold text-text-sec">{initials(name)}</span>;
  }
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt="" width={size} height={size} loading="lazy" style={box} onError={() => setFailed(true)}
    className="shrink-0 rounded-full border border-border-card bg-surface-sunk object-cover object-top" />;
}

/** The honesty note shown wherever a prop probability appears. */
export function FairOddsNote({ extra }: { extra?: ReactNode }) {
  return (
    <div role="note" className="ql-panel border-lock-gold/40 p-4">
      <p className="ql-kicker !text-lock-gold">Read this first</p>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-text-pri">
        These are model probabilities and fair odds, not sportsbook lines. There are no free sportsbook prop lines in this data, so nothing here is an
        offer you can bet at. Compare the fair odds with your book&apos;s price: a bet only has positive edge when the book pays <b>better</b> than fair
        (for example, fair is -150 and your book offers -130 or +100).{extra}
      </p>
    </div>
  );
}
