"use client";

import { useState } from "react";
import { headshot } from "@/lib/cbb-players.mjs";

const initials = (name: string) => name.split(/\s+/).filter(Boolean).slice(0, 2).map(w => w[0]).join("").toUpperCase();

/** ESPN college headshot; falls back to initials only when the photo does not exist (many freshmen have none yet). */
export function Headshot({ id, name, size = 40 }: { id: string | null | undefined; name: string; size?: number }) {
  const [failed, setFailed] = useState(false);
  const box = { width: size, height: size };
  if (!id || failed) {
    return <span aria-hidden="true" style={{ ...box, fontSize: Math.max(10, size * 0.34) }}
      className="inline-flex shrink-0 items-center justify-center rounded-full border border-border-card bg-surface-sunk font-display font-bold text-text-sec">{initials(name)}</span>;
  }
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={headshot(id)} alt="" width={size} height={size} loading="lazy" style={box} onError={() => setFailed(true)}
    className="shrink-0 rounded-full border border-border-card bg-surface-sunk object-cover object-top" />;
}
