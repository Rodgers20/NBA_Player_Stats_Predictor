"use client";

import Image from "next/image";
import { useState } from "react";
import { api } from "@/lib/api";
import { useClubCrest } from "@/lib/crests";
import Link from "next/link";
import { teamHref } from "@/components/team/TeamLink";
import { cn, displayTeam } from "@/lib/format";

/**
 * Club badge with a deterministic fallback.
 *
 * The official crest comes from football-data.org's directory, which covers
 * every club in the three competitions. Initials remain the fallback for a club
 * outside it — but they used to be the *common* case, which is how Barça became
 * "BA" and Celta became "C".
 */
export function TeamCrest({
  name, size = 20, className,
}: { name: string; size?: number; className?: string }) {
  const [failed, setFailed] = useState(false);
  const { url: official, loading } = useClubCrest(name);

  const initials = name
    .replace(/[^A-Za-z\s']/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0])
    .join("")
    .toUpperCase();

  // Hold the slot while the directory resolves: rendering the legacy endpoint
  // now would 404 and latch the initials fallback before the real crest lands.
  if (loading) {
    return (
      <span
        className={cn("shrink-0 rounded-full bg-elevated", className)}
        style={{ width: size, height: size }}
        aria-hidden
      />
    );
  }

  if (failed) {
    // Stable hue per club so the same team always looks the same.
    let hash = 0;
    for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) | 0;
    const hue = Math.abs(hash) % 360;
    return (
      <span
        className={cn("grid shrink-0 place-items-center rounded-full font-semibold", className)}
        style={{
          width: size, height: size,
          fontSize: Math.max(7, size * 0.36),
          background: `oklch(0.36 0.07 ${hue})`,
          color: `oklch(0.86 0.09 ${hue})`,
        }}
        aria-hidden
      >
        {initials}
      </span>
    );
  }

  return (
    <Image unoptimized
      src={official ?? api.crestUrl(name)}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      onError={() => { if (!official) setFailed(true); }}
      className={cn("shrink-0 object-contain", className)}
      style={{ width: size, height: size }}
    />
  );
}

/** Club badge plus name, the pairing used in tables and match rows. */
export function TeamLabel({
  name, size = 18, align = "left", className, truncate = true,
  competition, link = true,
}: {
  name: string;
  size?: number;
  align?: "left" | "right";
  className?: string;
  truncate?: boolean;
  /** Which league the club page should open on. Optional; it picks one otherwise. */
  competition?: string | null;
  /** Set false where this sits inside another link — anchors cannot nest. */
  link?: boolean;
}) {
  const body = (
    <>
      <TeamCrest name={name} size={size} />
      <span className={cn(truncate && "truncate")} title={name}>
        {displayTeam(name)}
      </span>
    </>
  );
  const shape = cn(
    "flex min-w-0 items-center gap-2",
    align === "right" && "flex-row-reverse",
    className,
  );

  // A club's name is an address everywhere it appears; the crest travels with
  // it so the whole label is one target rather than a word beside a dead badge.
  if (!link) return <span className={shape}>{body}</span>;
  return (
    <Link href={teamHref(name, competition)} className={cn(shape, "transition-colors hover:text-pitch")}>
      {body}
    </Link>
  );
}
