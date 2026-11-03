import { Suspense } from "react";
import { NhlGamePage } from "@/components/nhl/NhlGamePage";

export const metadata = { title: "Game | NHL" };

export default function NhlGameRoute() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading game…</p>}><NhlGamePage /></Suspense>;
}
