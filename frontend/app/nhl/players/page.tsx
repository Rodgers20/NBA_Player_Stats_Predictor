import { Suspense } from "react";
import { NhlPlayers } from "@/components/nhl/NhlPlayers";

export const metadata = { title: "NHL players | Court Vision" };

export default function NhlPlayersRoute() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading players…</p>}><NhlPlayers /></Suspense>;
}
