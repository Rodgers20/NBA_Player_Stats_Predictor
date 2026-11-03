import { Suspense } from "react";
import { NhlSlate } from "@/components/nhl/NhlSlate";

export const metadata = { title: "NHL | Court Vision" };

export default function NhlPage() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading games…</p>}><NhlSlate /></Suspense>;
}
