import { Suspense } from "react";
import { GameDetailPage } from "@/components/games/GameDetailPage";

export const metadata = { title: "Game | Court Vision" };

export default function GameDetailRoute() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading game…</p>}><GameDetailPage /></Suspense>;
}
