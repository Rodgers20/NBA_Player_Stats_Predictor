import { Suspense } from "react";
import { NhlTonight } from "@/components/nhl/NhlTonight";

export const metadata = { title: "NHL tonight | Court Vision" };

export default function NhlPage() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading tonight…</p>}><NhlTonight /></Suspense>;
}
