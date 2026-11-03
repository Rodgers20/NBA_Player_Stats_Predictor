"use client";

import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { nhlApi } from "@/lib/api";
import { NhlPlayerAnalysis } from "./NhlPlayerAnalysis";

/** Full-screen NHL player analysis. Owns one history entry so the back button closes it, like the NBA and college sheets. */
export function NhlPlayerSheet({ playerId, onClose }: { playerId: number; onClose: () => void }) {
  const back = useRef<HTMLButtonElement>(null);
  const closeRef = useRef(onClose);
  useEffect(() => { closeRef.current = onClose; }, [onClose]);
  useEffect(() => {
    const returnFocus = document.activeElement as HTMLElement | null;
    const { overflow } = document.documentElement.style;
    document.documentElement.style.overflow = "hidden";
    window.history.pushState({ ...window.history.state, nhlSheet: true }, "");
    const onPop = () => closeRef.current();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") window.history.back(); };
    window.addEventListener("popstate", onPop);
    window.addEventListener("keydown", onKey);
    back.current?.focus({ preventScroll: true });
    return () => {
      window.removeEventListener("popstate", onPop);
      window.removeEventListener("keydown", onKey);
      document.documentElement.style.overflow = overflow;
      returnFocus?.focus({ preventScroll: true });
    };
  }, []);

  const q = useQuery({ queryKey: ["nhl-player", playerId], queryFn: () => nhlApi.player(playerId), staleTime: 300_000, retry: false });
  return createPortal(
    <div role="dialog" aria-modal="true" aria-label={`${q.data?.player.name ?? "Player"} analysis`} className="ql-enter fixed inset-0 z-[91] overflow-y-auto overscroll-contain bg-[var(--color-bg-root)]">
      <div className="sticky top-0 z-20 flex items-center gap-3 border-b border-ink/10 bg-bg-root/85 px-3 py-2.5 backdrop-blur-xl sm:px-6">
        <button ref={back} type="button" onClick={() => window.history.back()}
          className="inline-flex items-center gap-2 rounded-full px-3 py-2 text-sm font-semibold text-text-pri hover:bg-ink/5 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
          <ArrowLeft size={18} aria-hidden="true" /> Back
        </button>
        <p className="min-w-0 flex-1 truncate text-sm font-bold text-text-sec">{q.data?.player.name}</p>
        <Link href={`/nhl/players?player=${playerId}`} className="shrink-0 text-xs font-semibold text-teal-300 underline">Full page</Link>
      </div>
      <div className="ql-page space-y-6">
        {q.isPending && <div className="ql-panel h-64 animate-pulse" aria-busy="true" aria-label="Loading player" />}
        {q.isError && <p role="alert" className="ql-panel p-5 text-sm text-text-sec">This player isn&apos;t in the data. <button type="button" className="underline" onClick={() => void q.refetch()}>Try again</button></p>}
        {q.data && <NhlPlayerAnalysis d={q.data} />}
      </div>
    </div>,
    document.body,
  );
}
