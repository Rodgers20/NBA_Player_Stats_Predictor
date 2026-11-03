"use client";

import { useState, type ReactNode } from "react";
import { TeamLogo } from "@/components/teams/TeamLogo";
import type { NhlTeamRef } from "@/lib/nhl-types";

export { ErrorBox, Loading, Mark } from "@/components/cbb/parts";

/** NBA-style page header: kicker, heading, subtitle, optional right-hand controls. Navigation lives in the bottom bar and rail. */
export function NhlHeader({ kicker, title, sub, children }: { kicker: string; title: string; sub?: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <p className="ql-kicker">{kicker} / NHL</p>
        <h1 className="ql-heading mt-1">{title}</h1>
        {sub && <p className="ql-subtitle mt-2 max-w-[80ch]">{sub}</p>}
      </div>
      {children}
    </header>
  );
}

/** Same page frame the NBA pages use. */
export const FRAME = "ql-page mx-auto max-w-[1440px] px-4 pb-12 pt-7 sm:px-8";

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
    <p role="note" className="ql-panel border-lock-gold/40 px-4 py-3 text-xs leading-relaxed text-text-sec">
      <b className="text-lock-gold">Model odds, not book lines.</b> Probabilities and fair odds only; there are no free NHL prop lines. A bet has an edge only when your book pays better than fair.{extra}
    </p>
  );
}
