import { Suspense } from "react";
import { NhlProps } from "@/components/nhl/NhlProps";

export const metadata = { title: "NHL props | Court Vision" };

export default function NhlPropsRoute() {
  return <Suspense fallback={<p className="p-8 text-text-sec">Loading props…</p>}><NhlProps /></Suspense>;
}
