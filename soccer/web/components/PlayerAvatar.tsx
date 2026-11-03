"use client";

import Image from "next/image";
import { useState } from "react";
import { api } from "@/lib/api";

/**
 * Player portrait with a graceful fallback.
 *
 * Portraits resolve through the API — a uniform API-Football studio crop where
 * one exists, otherwise initials — and are cached on disk, so a
 * miss is normal rather than an error. When there is no image the initials
 * stand in and nothing shifts layout.
 *
 * The studio crops are transparent PNGs, so the circle carries its own light
 * backing plate: without one the subject is a head floating on the page, and
 * dark hair disappears into a dark panel.
 */
export function PlayerAvatar({
  name, photo, size = 32, variant = "avatar",
}: { name: string; photo?: string; size?: number; variant?: "avatar" | "portrait" }) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const src = photo || api.headshotUrl(name);
  const initials = name
    .split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase();

  if (failedSrc === src) {
    return (
      <span
        className="grid shrink-0 place-items-center rounded-full bg-elevated text-[10px] font-semibold text-ink-secondary ring-1 ring-stroke"
        style={{ width: size, height: size }}
        aria-hidden
      >
        {initials}
      </span>
    );
  }
  return (
    <Image unoptimized
      src={src}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      onError={() => setFailedSrc(src)}
      className={variant === "portrait" ? "shrink-0 rounded-t-[3rem] object-contain object-bottom" : "shrink-0 rounded-full object-cover object-top ring-1 ring-stroke"}
      style={{
        width: size,
        height: size,
        // A touch lighter than the panel so a transparent portrait reads as a
        // portrait rather than a cut-out.
        background: variant === "portrait" ? "transparent" : "linear-gradient(180deg, rgba(255,255,255,.14), rgba(255,255,255,.05))",
      }}
    />
  );
}
