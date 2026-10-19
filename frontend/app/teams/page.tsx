import { Suspense } from "react";
import { TeamsBrowser } from "@/components/teams/TeamsBrowser";

export const metadata = { title: "Teams | Court Vision" };

export default function TeamsPage() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading teams…</p>}><TeamsBrowser /></Suspense>;
}
