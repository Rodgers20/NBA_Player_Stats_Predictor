import { Suspense } from "react";
import { NhlSlate } from "@/components/nhl/NhlSlate";

export const metadata = { title: "NHL games | Court Vision" };

export default function NhlGamesPage() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading games…</p>}><NhlSlate /></Suspense>;
}
