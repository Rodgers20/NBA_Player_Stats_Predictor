"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { TeamOfTheWeek } from "./TeamOfTheWeek";

export function TeamOfTheWeekPanel({ code, year }: { code: string; year?: number }) {
  const [round, setRound] = useState<number>();
  const query = useQuery({ queryKey: ["team-of-week", code, year, round], queryFn: () => api.teamOfTheWeek(code, round, year), staleTime: 15 * 60 * 1000 });
  if (query.isPending) return <p role="status" className="p-5 text-sm text-ink-muted">Loading round performances…</p>;
  if (query.isError) return <div className="p-5 text-sm text-ink-muted">Match data could not be loaded. <button onClick={() => query.refetch()} className="text-pitch underline">Retry</button></div>;
  return <TeamOfTheWeek key={`${code}-${year}-${query.data.match_week}`} data={query.data} onRoundChange={setRound} />;
}
