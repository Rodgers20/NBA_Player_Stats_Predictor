import { Suspense } from "react";
import { TeamsBrowser } from "@/components/teams/TeamsBrowser";

export const metadata = { title: "Team Profile | Court Vision" };

export default function TeamDetailPage() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading team…</p>}><TeamsBrowser detail /></Suspense>;
}
