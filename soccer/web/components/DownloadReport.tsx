"use client";
import type { LeagueMatch } from "@/lib/types";
export function DownloadReport({rows, name}: {rows: LeagueMatch[]; name: string}) {
  function download() {
    const quote = (v: unknown) => '"' + String(v ?? "").replace(/^[=+@-]/, "'$&").replaceAll('"', '""') + '"';
    const columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"] as const;
    const csv = [columns.join(","), ...rows.map(r => columns.map(c => quote(r[c])).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], {type:"text/csv;charset=utf-8"}));
    const a = document.createElement("a"); a.href = url; a.download = `${name}-results.csv`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <button disabled={!rows.length} onClick={download} className="min-h-11 rounded-lg bg-pitch px-4 text-sm font-semibold text-[#04160C] hover:brightness-110 disabled:opacity-50">Download results CSV</button>;
}
