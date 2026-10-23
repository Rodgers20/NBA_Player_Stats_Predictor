"use client";

import { Suspense, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { usePrefs } from "@/store/prefs";
import { PlayerAnalysis } from "@/components/analysis/PlayerAnalysis";

export default function AnalysisPage() {
  return <Suspense fallback={<div className="mx-auto max-w-[1700px] p-8 text-text-sec">Loading player analysis…</div>}><AnalysisSelection /></Suspense>;
}

function AnalysisSelection() {
  const params = useSearchParams();
  const savedLeague = usePrefs(s => s.league);
  const setLeague = usePrefs(s => s.setLeague);
  const requestedLeague = params.get("league");
  const league = requestedLeague === "nba" || requestedLeague === "wnba" ? requestedLeague : savedLeague;
  useEffect(() => {
    if (requestedLeague === "nba" || requestedLeague === "wnba") setLeague(requestedLeague);
  }, [requestedLeague, setLeague]);
  const linkedPlayer = params.get("player") ?? "";
  return <PlayerAnalysis key={`${league}:${linkedPlayer}`} linkedPlayer={linkedPlayer} league={league} />;
}
