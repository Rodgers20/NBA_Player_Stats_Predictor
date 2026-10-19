"use client";

import { useState } from "react";

export function TeamLogo({ team, size = 40 }: { team: { abbrev: string; name: string; logo?: string }; size?: number }) {
  const [failed, setFailed] = useState(false);
  return <span className="inline-flex shrink-0 items-center justify-center" style={{ width: size, height: size }}>
    {team.logo && !failed
      // ESPN logos are remote and sized explicitly; native images work with static exports.
      // eslint-disable-next-line @next/next/no-img-element
      ? <img src={team.logo} alt={`${team.name} logo`} width={size} height={size} className="h-full w-full object-contain" onError={() => setFailed(true)} />
      : <span className="font-display font-bold text-text-sec">{team.abbrev}</span>}
  </span>;
}
