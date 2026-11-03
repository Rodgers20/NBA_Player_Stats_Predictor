"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/format";

/**
 * Makes a player's name open their page from anywhere — leader boards, match
 * lineups, top performers.
 *
 * Most of those surfaces know a player only by name, so the id is resolved on
 * demand through search. Until it resolves the name still renders, just not as
 * a link: a name that silently does nothing when clicked is worse than one that
 * was never clickable.
 */
export function PlayerLink({
  name, playerId, className, children,
}: {
  name: string;
  /** Pass when the caller already knows it; skips the lookup entirely. */
  playerId?: string | null;
  className?: string;
  children?: React.ReactNode;
}) {
  const [resolved, setResolved] = useState<string | null>(playerId ?? null);

  useEffect(() => {
    if (playerId || !name) return;

    const controller = new AbortController();
    let cancelled = false;
    (async () => {
      try {
        const hits = await api.searchPlayers(name, 5, controller.signal);
        if (cancelled) return;
        const exact = hits.find(
          (h) => h.name.toLowerCase() === name.toLowerCase(),
        );
        setResolved(exact?.player_id ?? null);
      } catch {
        // A failed lookup just leaves the name unlinked.
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [name, playerId]);

  const body = children ?? name;
  if (!resolved) {
    return <span className={className}>{body}</span>;
  }
  return (
    <Link
      href={`/players/${encodeURIComponent(resolved)}`}
      className={cn("transition-colors hover:text-pitch", className)}
    >
      {body}
    </Link>
  );
}
