import { Suspense } from "react";
import { GameBoxPage } from "@/components/games/GameBoxPage";

export const metadata = { title: "Box Score | Court Vision" };

export default function BoxScorePage() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading box score…</p>}><GameBoxPage /></Suspense>;
}
