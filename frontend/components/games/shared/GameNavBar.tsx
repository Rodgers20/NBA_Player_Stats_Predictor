"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";

export type NavTarget = { href: string; title: string; sub?: string };

/** Back-to-slate link plus previous / next game, also on the arrow keys, so a reader can step through a day. */
export function GameNavBar({ backHref, backLabel, position, prev, next }: { backHref: string; backLabel: string; position?: string; prev?: NavTarget; next?: NavTarget }) {
  const router = useRouter();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if ((t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) || t?.closest('[role="tablist"]') || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowLeft" && prev) router.push(prev.href);
      if (e.key === "ArrowRight" && next) router.push(next.href);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [prev, next, router]);
  const link = (t: NavTarget | undefined, dir: "prev" | "next") => t ? (
    <Link href={t.href} className={`ql-chip min-w-0 flex-1 gap-2 sm:flex-none ${dir === "next" ? "!justify-end" : "!justify-start"}`} aria-label={`${dir === "prev" ? "Previous" : "Next"} game: ${t.title}`}>
      {dir === "prev" && <ChevronLeft size={14} aria-hidden className="shrink-0" />}<span className="truncate">{t.title}</span>{dir === "next" && <ChevronRight size={14} aria-hidden className="shrink-0" />}
    </Link>) : null;
  return (
    <nav aria-label="Games on this day" className="mb-4 flex flex-wrap items-center justify-between gap-2">
      <Link href={backHref} className="inline-flex items-center gap-1.5 text-sm text-text-sec hover:text-text-pri"><ArrowLeft size={16} aria-hidden /> {backLabel}{position ? ` · ${position}` : ""}</Link>
      {(prev || next) && <div className="flex w-full gap-2 sm:w-auto">{link(prev, "prev")}{link(next, "next")}</div>}
    </nav>
  );
}
