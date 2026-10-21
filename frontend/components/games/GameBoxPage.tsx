"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { usePrefs } from "@/store/prefs";
import { BoxScore } from "./BoxScore";

export function GameBoxPage() {
  const params = useSearchParams();
  const preferred = usePrefs(s => s.league);
  const requested = params.get("league");
  const league = requested === "wnba" || requested === "nba" ? requested : preferred;
  const id = params.get("id") ?? "";
  const valid = /^\d{6,12}$/.test(id);
  return <main className="ql-page mx-auto max-w-[1440px] px-4 pb-16 pt-8 sm:px-8">
    <Link href={`/teams?league=${league}`} className="mb-5 inline-flex items-center gap-2 text-sm text-teal-400"><ArrowLeft size={16} /> Back to teams</Link>
    <p className="ql-kicker">{league.toUpperCase()} / GAME REPORT</p>
    {valid ? <BoxScore league={league} gameId={id} /> : <p role="alert" className="glass mt-4 p-6 text-text-sec">That game link is not valid.</p>}
  </main>;
}
