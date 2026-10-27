"use client";

import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { motion, useReducedMotion } from "motion/react";
import { ArrowLeft, Maximize2 } from "lucide-react";
import { PlayerAnalysis, type AnalysisFocus } from "@/components/analysis/PlayerAnalysis";

// Critically damped (no overshoot): the sheet was tapped open, not flung, so it should arrive and settle.
const SPRING = { type: "spring", bounce: 0, duration: 0.5 } as const;

interface Props {
  player: string;
  league: "nba" | "wnba";
  focus?: AnalysisFocus;
  /** Where the tap happened, so the sheet grows out of the row and shrinks back into it. */
  origin?: { x: number; y: number } | null;
  backLabel?: string;
  onClose: () => void;
}

/**
 * Full-screen player analysis presented over the current page. It owns one history entry, so the browser's
 * back button (and swipe-back on phones) closes it like a real page, and every close goes through history.
 */
export function PlayerAnalysisSheet({ player, league, focus, origin, backLabel = "Back to Props", onClose }: Props) {
  const reduce = useReducedMotion();
  const backButton = useRef<HTMLButtonElement>(null);
  const closeRef = useRef(onClose);
  useEffect(() => { closeRef.current = onClose; }, [onClose]);

  useEffect(() => {
    const returnFocus = document.activeElement as HTMLElement | null;
    const { overflow } = document.documentElement.style;
    document.documentElement.style.overflow = "hidden"; // the board underneath must not scroll
    window.history.pushState({ ...window.history.state, playerSheet: true }, "");
    const onPop = () => closeRef.current();
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") window.history.back(); };
    window.addEventListener("popstate", onPop);
    window.addEventListener("keydown", onKey);
    backButton.current?.focus({ preventScroll: true });
    return () => {
      window.removeEventListener("popstate", onPop);
      window.removeEventListener("keydown", onKey);
      document.documentElement.style.overflow = overflow;
      returnFocus?.focus({ preventScroll: true });
    };
  }, []);

  const transformOrigin = origin ? `${origin.x}px ${origin.y}px` : "50% 30%";
  const hidden = reduce ? { opacity: 0 } : { opacity: 0, scale: 0.9, y: 24, borderRadius: 28 };
  const shown = reduce ? { opacity: 1 } : { opacity: 1, scale: 1, y: 0, borderRadius: 0 };

  return createPortal(
    <>
      <motion.div className="fixed inset-0 z-[90] bg-black/60 backdrop-blur-sm" aria-hidden="true"
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }} />
      <motion.div role="dialog" aria-modal="true" aria-label={`${player} player analysis`}
        className="fixed inset-0 z-[91] overflow-y-auto overscroll-contain bg-[var(--color-bg-root)] will-change-transform"
        style={{ transformOrigin: reduce ? undefined : transformOrigin }}
        initial={hidden} animate={shown} exit={hidden} transition={reduce ? { duration: 0.2 } : SPRING}>
        <div className="sticky top-0 z-20 flex items-center justify-between gap-3 border-b border-ink/10 bg-bg-root/80 px-3 py-2.5 backdrop-blur-xl sm:px-6">
          <button ref={backButton} type="button" onClick={() => window.history.back()}
            className="inline-flex items-center gap-2 rounded-full px-3 py-2 text-sm font-semibold text-text-pri transition-transform hover:bg-ink/5 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
            <ArrowLeft size={18} aria-hidden="true" /> {backLabel}
          </button>
          <p className="truncate text-sm font-bold text-text-sec">{player}</p>
          <Link href={`/analysis?league=${league}&player=${encodeURIComponent(player)}`} prefetch={false}
            className="inline-flex items-center gap-2 rounded-full border border-ink/10 px-3 py-2 text-xs font-semibold text-text-sec hover:text-text-pri">
            <Maximize2 size={14} aria-hidden="true" /> <span className="hidden sm:inline">Open as page</span>
          </Link>
        </div>
        <PlayerAnalysis linkedPlayer={player} league={league} focus={focus} embedded />
      </motion.div>
    </>,
    document.body,
  );
}
